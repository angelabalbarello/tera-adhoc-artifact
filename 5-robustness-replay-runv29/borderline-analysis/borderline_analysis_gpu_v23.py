# -*- coding: utf-8 -*-
"""
borderline_analysis_gpu.py  (v23)
══════════════════════════════════════════════════════════════════════
Análise qualitativa dos episódios borderline (Atenção + Alerta) para
a Seção 5.5 do artigo FGCS.

Protocolo:
  1. Gera dataset de treino (Normal + Crítico) via synthetic_driver_risk_v7
  2. Carrega pesos AF-KD do run_v23.py (REUSE_TRAINED_MODELS=True) OU
     re-treina com os MESMOS hiperparâmetros do experimento principal.
     IMPORTANTE: use sempre REUSE_TRAINED_MODELS=True para garantir que
     os resultados borderline refletem o MESMO modelo do artigo.
  3. Gera episódios borderline (Atenção + Alerta) — NUNCA vistos no treino
  4. Roda inferência frame-a-frame em todos os episódios borderline
  5. Reporta estatísticas prontas para inserir na Seção 5.5
  6. Salva CSV de resultados + figura para o artigo

v23 — novidades:
  [C1] REUSE_TRAINED_MODELS: carrega checkpoints de modelos_salvos/ gerados
       pelo run_v23.py. Garante consistência com os resultados principais
       e evita re-treino desnecessário (cada seed poupa ~horas de CPU).
  [C2] Auditoria de recipe_id por split impressa no log (igual ao run_v23.py).
  [C3] Treinamento salva checkpoints em modelos_salvos/ se REUSE=False.

Requisitos:
  - synthetic_driver_risk_v7.py no mesmo diretório
  - modelos_salvos/student_seed{42..45}.pt  (gerados pelo run_v23.py)
  - pip install torch numpy pandas matplotlib scikit-learn

Uso:
  python borderline_analysis_gpu.py                  # usa checkpoints
  python borderline_analysis_gpu.py --no_reuse       # re-treina do zero
  python borderline_analysis_gpu.py --seeds 42 43 --per_recipe 80 --theta 0.10
══════════════════════════════════════════════════════════════════════
"""

import argparse
import math
import random
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Importa o gerador v6 ────────────────────────────────────────────────────
try:
    import synthetic_driver_risk_v7 as gen
    print("✓ synthetic_driver_risk_v7 carregado")
except ImportError:
    print("ERRO: synthetic_driver_risk_v7.py não encontrado no diretório atual.")
    print(f"  Diretório atual: {os.getcwd()}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTES — idênticas ao run_v22_v6.py
# ══════════════════════════════════════════════════════════════════════════════
SEEDS        = [42, 43, 44, 45]
T            = 96        # frames por episódio
D            = 31        # dimensão de entrada
H            = 64        # hidden size LSTM
K_AGG        = 6         # agregador causal (últimos k frames)
M_DETECT     = 3         # m consecutivos para detecção estável
THETA        = 0.10      # limiar operacional do artigo
FRAME_THR    = 0.35      # threshold do y_frame_clean → label binário
PER_RECIPE   = 80        # episódios por receita
LAT_WARMUP   = 100       # warm-up de latência

# Hiperparâmetros de treino — idênticos ao run_v22_v6.py
TEACHER_EPOCHS  = 60
BASELINE_EPOCHS = 60
STUDENT_EPOCHS  = 80
STUDENT_LR      = 5e-4
STUDENT_BETA    = 0.35
STUDENT_TEMP    = 2.0
STUDENT_LAM_S   = 0.20
STUDENT_LAM_E   = 1.00
EARLY_ONSET_THR = 0.35
EARLY_TARGET_HI = 0.90

BORDERLINE_CATS = {"Atencao", "Alerta"}
OUT_PREFIX      = "borderline"

# ── v23: checkpoints ─────────────────────────────────────────────────────────
# MODEL_DIR deve apontar para o mesmo diretório usado pelo run_v23.py.
# Por padrão ambos os scripts rodam na mesma pasta de trabalho.
MODEL_DIR            = Path("modelos_salvos")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
# REUSE_TRAINED_MODELS=True → carrega student_seed{N}.pt do run_v23.py
# REUSE_TRAINED_MODELS=False → re-treina do zero e salva novos checkpoints
# ATENÇÃO: use True para garantir consistência com os resultados do artigo.
REUSE_TRAINED_MODELS = True


# ══════════════════════════════════════════════════════════════════════════════
# 2. UTILITÁRIOS
# ══════════════════════════════════════════════════════════════════════════════
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_pos_weight(y: np.ndarray) -> float:
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    return max(neg / pos, 1.0) if pos > 0 else 1.0


def build_loss_objects(yfr: np.ndarray, yep: np.ndarray, device: torch.device):
    pw_fr = torch.tensor([make_pos_weight(yfr.reshape(-1))], device=device).float()
    pw_ep = torch.tensor([make_pos_weight(yep.reshape(-1))], device=device).float()
    return (nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw_fr),
            nn.BCEWithLogitsLoss(pos_weight=pw_ep))


def make_prefix_window(x_seq: np.ndarray, t: int, window: int = T) -> np.ndarray:
    """Janela causal até t, com pad à esquerda pelo primeiro frame."""
    x_slice = x_seq[:t + 1, :]
    if x_slice.shape[0] < window:
        pad_len   = window - x_slice.shape[0]
        pad_frame = x_slice[0:1, :] if x_slice.shape[0] > 0 else np.zeros(
            (1, x_seq.shape[1]), dtype=x_seq.dtype)
        x_slice = np.concatenate([np.repeat(pad_frame, pad_len, axis=0), x_slice], axis=0)
    else:
        x_slice = x_slice[-window:]
    return x_slice.astype(np.float32, copy=False)


def episode_probs_from_frames(frame_probs: np.ndarray, k: int = K_AGG) -> np.ndarray:
    return np.mean(frame_probs[:, -k:], axis=1)


def binary_entropy(p: float) -> float:
    p = float(np.clip(p, 1e-7, 1 - 1e-7))
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def first_stable_detection(probs: np.ndarray, thr: float, m: int) -> int:
    """Primeiro frame onde m consecutivos >= thr. Retorna -1 se não detecta."""
    count = 0
    for t, p in enumerate(probs):
        if p >= thr:
            count += 1
            if count >= m:
                return t - m + 1
        else:
            count = 0
    return -1


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODELO — idêntico ao run_v22_v6.py
# ══════════════════════════════════════════════════════════════════════════════
class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim: int = D, hidden_dim: int = H, bi: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                            bidirectional=bi, num_layers=2, dropout=0.1)
        d = hidden_dim * 2 if bi else hidden_dim
        self.fc_fr = nn.Linear(d, 1)
        self.fc_ep = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor):
        h, _ = self.lstm(x)
        fr = self.fc_fr(h)           # (B, T, 1)
        ep = self.fc_ep(h.mean(1))   # (B, 1)
        return fr, ep


# ══════════════════════════════════════════════════════════════════════════════
# 4. TREINO — idêntico ao run_v22_v6.py
# ══════════════════════════════════════════════════════════════════════════════
def train_teacher(model: nn.Module, X: np.ndarray, yfr: np.ndarray,
                  yep: np.ndarray, device: torch.device,
                  epochs: int = TEACHER_EPOCHS) -> nn.Module:
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    bce_fr, bce_ep = build_loss_objects(yfr, yep, device)
    Xt    = torch.tensor(X,   device=device).float()
    yfr_t = torch.tensor(yfr, device=device).float()
    yep_t = torch.tensor(yep, device=device).float().unsqueeze(1)
    for e in range(epochs):
        opt.zero_grad()
        fr_log, ep_log = model(Xt)
        fr_log = fr_log.squeeze(-1)
        loss = bce_fr(fr_log, yfr_t).mean() + 0.5 * bce_ep(ep_log, yep_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (e + 1) % 20 == 0:
            print(f"    teacher  [{e+1:3d}/{epochs}]  loss={loss.item():.4f}")
    return model


def train_afkd(student: nn.Module, teacher: nn.Module,
               X: np.ndarray, yfr_main: np.ndarray, yep: np.ndarray,
               device: torch.device,
               yfr_soft: Optional[np.ndarray] = None,   # ← y_frame_clean CONTÍNUO
               epochs: int         = STUDENT_EPOCHS,
               beta_max: float     = STUDENT_BETA,
               temp: float         = STUDENT_TEMP,
               lam_start: float    = STUDENT_LAM_S,
               lam_end: float      = STUDENT_LAM_E,
               early_onset_thr: float = EARLY_ONSET_THR,
               early_target_hi: float = EARLY_TARGET_HI,
               warmup_epochs: int  = 12,
               rampup_epochs: int  = 16) -> nn.Module:
    """AF-KD — idêntico ao run_v22_v6.py (todos os FIX/OPT preservados).
    yfr_soft deve ser y_frame_clean CONTÍNUO (0.0–1.0), não o label binário.
    """
    student.to(device).train()
    teacher.to(device).eval()
    opt = torch.optim.AdamW(student.parameters(), lr=STUDENT_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=STUDENT_LR * 0.1)
    bce_fr, bce_ep = build_loss_objects(yfr_main, yep, device)

    Xt        = torch.tensor(X,        device=device).float()
    yfr_t     = torch.tensor(yfr_main, device=device).float()
    yep_t     = torch.tensor(yep,      device=device).float().unsqueeze(1)
    yep_bool  = torch.tensor(yep > 0,  device=device)
    # BUG FIX: usar y_frame_clean CONTÍNUO como soft target (não o binário)
    # No runner original: yfr_soft=yfrc_tr_fc = y_fr_cont[tr_idx] (float 0-1)
    if yfr_soft is None:
        yfr_soft = yfr_main.astype(np.float32)
    yfr_soft_t = torch.tensor(yfr_soft, device=device).float()

    t_len    = X.shape[1]
    base_idx = torch.arange(t_len, device=device).float()

    # Parâmetros da early-loss (idênticos ao run_v22_v6.py)
    K           = 50
    p_min_start = 0.50
    p_min_end   = 0.78
    delta_start = 0.30
    delta_end   = 2.80
    gamma_ep    = 0.25
    ramp_end_epoch = warmup_epochs + rampup_epochs

    for e in range(epochs):
        # ── Fase e parâmetros ────────────────────────────────────────────────
        if e < warmup_epochs:
            beta_curr  = 0.0
            phase      = "WARMUP"
            warm_frac  = (e + 1) / max(warmup_epochs, 1)
            delta      = delta_start * warm_frac
            p_min      = p_min_start
            alpha_curr = 1.0
        elif e < ramp_end_epoch:
            ramp_frac  = (e - warmup_epochs + 1) / rampup_epochs
            beta_curr  = beta_max * ramp_frac
            phase      = "RAMPUP"
            delta      = delta_start + (1.20 - delta_start) * ramp_frac
            p_min      = p_min_start + (0.68 - p_min_start) * ramp_frac
            alpha_curr = 0.85 - 0.25 * ramp_frac
        else:
            beta_curr  = beta_max
            phase      = "REFINE"
            ref_frac   = (e - ramp_end_epoch + 1) / max(epochs - ramp_end_epoch, 1)
            delta      = 1.20 + (delta_end - 1.20) * ref_frac
            p_min      = 0.68 + (p_min_end - 0.68) * ref_frac
            alpha_curr = 0.60 - 0.20 * ref_frac

        alpha_curr = max(alpha_curr, 0.50)
        frac = min((e + 1) / max(ramp_end_epoch, 1), 1.0)
        lam  = lam_start + (lam_end - lam_start) * frac
        w    = (1.0 + lam * (base_idx / max(t_len - 1, 1))).unsqueeze(0)

        student.train()
        opt.zero_grad()
        s_fr, s_ep = student(Xt)
        s_fr = s_fr.squeeze(-1)

        with torch.no_grad():
            t_fr, _ = teacher(Xt)
            t_p     = torch.sigmoid(t_fr.squeeze(-1) / temp)

        # ── Perdas ──────────────────────────────────────────────────────────
        l_hard = (bce_fr(s_fr, yfr_t) * w).mean()
        l_ep   = bce_ep(s_ep, yep_t)

        # KD via MSE gateado em críticos (FIX-11)
        if beta_curr > 0:
            crit_idx = yep_bool.nonzero(as_tuple=True)[0]
            l_soft = F.mse_loss(torch.sigmoid(s_fr[crit_idx]), t_p[crit_idx]) \
                     if len(crit_idx) > 0 else torch.tensor(0.0, device=device)
        else:
            l_soft = torch.tensor(0.0, device=device)

        s_p = torch.sigmoid(s_fr)

        # Early-loss somente em críticos (FIX-6)
        has_onset  = (yfr_soft_t.max(dim=1).values >= early_onset_thr) & yep_bool
        onset_idx  = (yfr_soft_t >= early_onset_thr).float().argmax(dim=1)
        early_track = early_floor = early_mono = early_push = torch.tensor(0.0, device=device)
        count = 0
        for b in range(s_p.shape[0]):
            if not bool(has_onset[b].item()):
                continue
            t0b   = int(onset_idx[b].item())
            t1b   = min(t_len, t0b + K)
            if t1b <= t0b:
                continue
            seg_prob = s_p[b, t0b:t1b]
            seg_soft = yfr_soft_t[b, t0b:t1b]
            steps    = torch.arange(seg_prob.shape[0], device=device).float()
            soft_norm   = torch.clamp(
                (seg_soft - early_onset_thr) / max(early_target_hi - early_onset_thr, 1e-6),
                0.0, 1.0)
            target_curve = 0.30 + 0.60 * soft_norm
            time_focus   = 1.35 / (1.0 + 0.08 * steps)

            mse_track  = ((seg_prob - target_curve) ** 2) * time_focus
            floor_curve = torch.maximum(torch.full_like(target_curve, p_min),
                                        target_curve * 0.95)
            margin_pen = torch.relu(floor_curve - seg_prob) * time_focus

            if seg_prob.shape[0] >= 3:
                diffs   = seg_prob[1:] - seg_prob[:-1]
                desired = 0.015 + torch.relu(target_curve[1:] - target_curve[:-1])
                early_mono = early_mono + (torch.relu(desired - diffs) * time_focus[1:]).mean()

            head_len = min(8, seg_prob.shape[0])
            push_target = torch.linspace(0.45, min(0.80, p_min + 0.10),
                                         head_len, device=device)
            early_push  = early_push + torch.relu(push_target - seg_prob[:head_len]).mean()
            early_track = early_track + mse_track.mean()
            early_floor = early_floor + margin_pen.mean()
            count += 1

        # Tail penalisation para críticos (FIX-9b)
        tail_pen = torch.tensor(0.0, device=device)
        tail_count = 0
        for b in range(s_p.shape[0]):
            if not bool(yep_bool[b].item()):
                continue
            tail_pen   = tail_pen + torch.relu(0.40 - s_p[b, -K_AGG:]).mean()
            tail_count += 1
        if tail_count > 0:
            tail_pen = tail_pen / tail_count

        # Supressão cauda não-críticos (FIX-13a / FIX-18a)
        noncrit_pen = torch.tensor(0.0, device=device)
        nc_count = 0
        for b in range(s_p.shape[0]):
            if bool(yep_bool[b].item()):
                continue
            noncrit_pen = noncrit_pen + torch.relu(s_p[b, -K_AGG:] - 0.04).mean() * 3.0
            nc_count += 1
        if nc_count > 0:
            noncrit_pen = noncrit_pen / nc_count

        if count > 0:
            early_track = early_track / count
            early_floor = early_floor / count
            early_mono  = early_mono  / count
            early_push  = early_push  / count
            early_pen   = 0.60 * early_track + 1.60 * early_floor + \
                          1.20 * early_mono   + 1.50 * early_push
        else:
            early_pen = torch.tensor(0.0, device=device)

        loss = (alpha_curr * l_hard
                + beta_curr  * l_soft
                + gamma_ep   * l_ep
                + delta      * early_pen
                + 1.20       * tail_pen
                + 2.00       * noncrit_pen)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if (e + 1) % 10 == 0:
            print(f"    afkd [{e+1:3d}/{epochs}] [{phase}]  "
                  f"loss={loss.item():.4f}  "
                  f"(hard={l_hard.item():.3f} soft={l_soft.item():.3f} "
                  f"ep={l_ep.item():.3f} early={early_pen.item():.3f} "
                  f"tail+={tail_pen.item():.3f} tail-={noncrit_pen.item():.3f})")
    return student


# ══════════════════════════════════════════════════════════════════════════════
# 5. INFERÊNCIA FRAME-A-FRAME
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def warmup_model(model: nn.Module, x_ref: np.ndarray, device: torch.device,
                 steps: int = LAT_WARMUP):
    model.eval()
    xt = torch.tensor(x_ref, device=device).float().unsqueeze(0)
    for _ in range(steps):
        model(xt)


@torch.no_grad()
def infer_episodes(model: nn.Module, X: np.ndarray,
                   device: torch.device) -> np.ndarray:
    """
    Inferência frame-a-frame estritamente causal (política fixa).
    Retorna frame_probs shape (N, T).
    """
    model.eval()
    n, t_len, _ = X.shape
    frame_probs = np.zeros((n, t_len), dtype=np.float32)

    x_ref = make_prefix_window(X[0], min(5, t_len - 1))
    warmup_model(model, x_ref, device)

    for i in range(n):
        for t in range(t_len):
            x_slice = make_prefix_window(X[i], t)
            xt = torch.tensor(x_slice, device=device).float().unsqueeze(0)
            fr_log, _ = model(xt)
            frame_probs[i, t] = float(torch.sigmoid(fr_log[0, -1, 0]).item())

    return frame_probs


# ══════════════════════════════════════════════════════════════════════════════
# 6. ANÁLISE BORDERLINE
# ══════════════════════════════════════════════════════════════════════════════
def analyse_borderline(frame_probs: np.ndarray,
                       metas: List[dict],
                       theta: float = THETA,
                       m: int = M_DETECT) -> pd.DataFrame:
    rows = []
    for i, meta in enumerate(metas):
        fp  = frame_probs[i]
        cat = str(meta.get("categoria", meta.get("class_name", "")))
        det = first_stable_detection(fp, theta, m)
        rows.append({
            "seed":             meta.get("_seed", -1),
            "episode_idx":      i,
            "categoria":        cat,
            "nome":             meta.get("nome", ""),
            "recipe_id":        meta.get("recipe_id", ""),
            "risk_target":      float(meta.get("risk_target", 0)),
            "risk_episode":     float(meta.get("risk_episode", 0)),
            "max_prob":         round(float(np.max(fp)), 4),
            "mean_prob":        round(float(np.mean(fp)), 4),
            "ep_prob_k6":       round(float(np.mean(fp[-K_AGG:])), 4),
            "frac_above_theta": round(float(np.mean(fp >= theta)), 4),
            "detection_frame":  det,
            "would_alert":      int(det >= 0),
            "theta":            theta,
        })
    return pd.DataFrame(rows)


def aggregate(ep_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, cat), grp in ep_df.groupby(["seed", "categoria"]):
        rows.append({
            "seed":             seed,
            "categoria":        cat,
            "n":                len(grp),
            "max_prob_mean":    round(grp["max_prob"].mean(), 4),
            "max_prob_std":     round(grp["max_prob"].std(), 4),
            "mean_prob_mean":   round(grp["mean_prob"].mean(), 4),
            "mean_prob_std":    round(grp["mean_prob"].std(), 4),
            "frac_above_mean":  round(grp["frac_above_theta"].mean(), 4),
            "pct_would_alert":  round(grp["would_alert"].mean() * 100, 1),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURAS
# ══════════════════════════════════════════════════════════════════════════════
def plot_results(ep_df: pd.DataFrame,
                 frame_probs_all: Dict,
                 theta: float,
                 out_prefix: str = OUT_PREFIX):
    cats   = ["Atencao", "Alerta"]
    colors = {"Atencao": "#2196F3", "Alerta": "#FF9800"}
    labels = {"Atencao": "Atenção\n(risk 0.25–0.30)", "Alerta": "Alerta\n(risk 0.72–0.78)"}

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── Linha 1: violin plots ───────────────────────────────────────────────
    metrics = [
        ("max_prob",  "Prob. máxima por episódio"),
        ("mean_prob", "Prob. média por episódio"),
        ("frac_above_theta", f"Fração de frames ≥ θ={theta:.2f}"),
    ]
    for col_idx, (col, title) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, col_idx])
        data = [ep_df.loc[ep_df["categoria"] == c, col].values for c in cats]
        vp = ax.violinplot(data, positions=range(len(cats)),
                           showmedians=True, showextrema=True)
        for pc, c in zip(vp["bodies"], cats):
            pc.set_facecolor(colors[c])
            pc.set_alpha(0.50)
        for i, (c, vals) in enumerate(zip(cats, data)):
            jitter = np.random.default_rng(42).uniform(-0.07, 0.07, size=len(vals))
            ax.scatter(i + jitter, vals, color=colors[c], alpha=0.40, s=14, zorder=3)
        if col != "frac_above_theta":
            ax.axhline(theta, color="red", lw=1.2, ls="--",
                       label=f"θ={theta:.2f}", alpha=0.85)
            ax.legend(fontsize=8)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([labels[c] for c in cats], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", alpha=0.30)

    # ── Linha 2: trajetórias médias frame-a-frame ───────────────────────────
    for col_idx, cat in enumerate(cats):
        ax = fig.add_subplot(gs[1, col_idx])
        sub    = ep_df[ep_df["categoria"] == cat]
        traces = [frame_probs_all[k]
                  for k in zip(sub["seed"], sub["episode_idx"])
                  if k in frame_probs_all]
        if traces:
            arr     = np.stack(traces)
            t_axis  = np.arange(arr.shape[1])
            mean_tr = arr.mean(axis=0)
            std_tr  = arr.std(axis=0)
            color   = colors[cat]
            for tr in arr:
                ax.plot(t_axis, tr, color=color, alpha=0.06, lw=0.5)
            ax.plot(t_axis, mean_tr, color=color, lw=2.2,
                    label=f"Média (n={len(traces)})")
            ax.fill_between(t_axis, mean_tr - std_tr, mean_tr + std_tr,
                            color=color, alpha=0.20, label="±1σ")
        ax.axhline(theta, color="red", lw=1.2, ls="--",
                   label=f"θ={theta:.2f}", alpha=0.85)
        ax.set_title(f"Trajetória: {labels[cat]}", fontsize=10)
        ax.set_xlabel("Frame (t)")
        ax.set_ylabel("P(risco | AF-KD)")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.30)

    # ── Linha 2, col 3: % que dispararia alerta ─────────────────────────────
    ax  = fig.add_subplot(gs[1, 2])
    agg = ep_df.groupby("categoria")["would_alert"].mean() * 100
    bar_colors = [colors.get(c, "gray") for c in agg.index]
    bars = ax.bar(range(len(agg)), agg.values, color=bar_colors,
                  edgecolor="black", linewidth=0.6, alpha=0.80)
    for bar, val in zip(bars, agg.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5, f"{val:.1f}%",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([labels[c] for c in agg.index], fontsize=9)
    ax.set_ylabel("% de episódios que disparariam alerta")
    ax.set_title(f"Taxa de alerta com θ={theta:.2f}, m={M_DETECT}", fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.30)

    fig.suptitle(
        f"AF-KD sobre episódios borderline — análise qualitativa pós-treino\n"
        f"(seeds 42–45, θ={theta:.2f}, m={M_DETECT}, K_AGG={K_AGG})",
        fontsize=13, y=1.01
    )
    out = f"{out_prefix}_figure.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Figura salva: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. RELATÓRIO PARA O ARTIGO
# ══════════════════════════════════════════════════════════════════════════════
def print_report(ep_df: pd.DataFrame, agg_df: pd.DataFrame, theta: float):
    print("\n" + "═" * 72)
    print("  RESULTADOS — ANÁLISE BORDERLINE (AF-KD v22_v6, seeds 42–45)")
    print("═" * 72)

    consolidated = {}
    for cat, grp in agg_df.groupby("categoria"):
        consolidated[cat] = {
            "n":    int(grp["n"].sum()),
            "mm":   round(grp["max_prob_mean"].mean(), 3),
            "ms":   round(grp["max_prob_std"].mean(), 3),
            "mp":   round(grp["mean_prob_mean"].mean(), 3),
            "fa":   round(grp["frac_above_mean"].mean(), 3),
            "pct":  round(grp["pct_would_alert"].mean(), 1),
        }

    for cat, d in consolidated.items():
        print(f"\n  [{cat.upper()}]  (n={d['n']} episódios, soma 4 seeds)")
        print(f"    Prob. máxima média    : {d['mm']:.3f} ± {d['ms']:.3f}")
        print(f"    Prob. média por ep.   : {d['mp']:.3f}")
        print(f"    Frames ≥ θ={theta:.2f}     : {d['fa']*100:.1f}%")
        print(f"    Dispararia alerta (m={M_DETECT}): {d['pct']:.1f}%")

    # ── Texto pronto para Seção 5.5 ─────────────────────────────────────────
    at = consolidated.get("Atencao", {})
    al = consolidated.get("Alerta",  {})
    print("\n" + "─" * 72)
    print("  TEXTO PARA INSERIR NA SEÇÃO 5.5 DO ARTIGO:")
    print("─" * 72)
    print(f"""
To assess the model's behaviour on intermediate-risk episodes excluded
from training, we applied the trained AF-KD model to {at.get('n','?')} attention-level
and {al.get('n','?')} alert-level episodes generated by the Safe-Drive Generator
(Atenção: risk target 0.25–0.30; Alerta: risk target 0.72–0.78).
These episodes were never seen during training.

For attention-level episodes — low-severity distractions at low speed
(parked phone use, grooming, conversation) — the model produced a mean
maximum probability of {at.get('mm','?'):.3f} ± {at.get('ms','?'):.3f}, with {at.get('fa','?')*100:.1f}% of
frames exceeding θ = {theta:.2f}, and {at.get('pct','?'):.1f}% of episodes triggering a
stable alert (m = {M_DETECT} consecutive frames). These values indicate that the
model does not systematically alarm on genuinely low-risk behaviour.

For alert-level episodes — transient high-severity events (hard braking
in rain, sharp acceleration in dense traffic, object retrieval at speed)
with risk targets 0.72–0.78 — the model produced a mean maximum
probability of {al.get('mm','?'):.3f} ± {al.get('ms','?'):.3f}, with {al.get('fa','?')*100:.1f}% of frames exceeding
θ = {theta:.2f}, and {al.get('pct','?'):.1f}% triggering a stable alert. The higher alert
rate for these episodes is consistent with their objective severity
(risk target near the critical boundary); the decision to exclude them
from training is methodological — to obtain an operationally valid
FPR measured exclusively over genuinely safe Normal driving — not an
empirical judgement that such events are risk-free.

Taken together, these results suggest that the AF-KD model trained on
the binary Normal–Critical separation generalises coherently to the
intermediate risk continuum: it does not over-alarm on low-severity
attention episodes while responding proportionally to the higher
objective risk of alert-level events.
""")
    print("═" * 72)


# ══════════════════════════════════════════════════════════════════════════════
# 9. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
def main(seeds: List[int], per_recipe: int, theta: float, reuse: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_ep_rows   = []
    frame_probs_all = {}   # {(seed, ep_idx): np.array (T,)}

    for SEED in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {SEED}")
        print(f"{'='*60}")
        set_seed(SEED)

        # ── 1) Gerar dataset de TREINO (Normal + Crítico) ─────────────────
        print(f"\n[1/4] Gerando dataset de treino (seed {SEED})...")
        X_all, y_ep_all, y_fr_clean, _, metas_list = gen.build_dataset(
            gen.RECIPES_TRAIN,
            window     = T,
            per_recipe = per_recipe,
            seed       = SEED,
            shuffle    = True,
            SENSOR_LEVEL = 4,
        )
        meta_df = pd.DataFrame(metas_list)
        cat_col = next((c for c in ("class_name", "categoria") if c in meta_df.columns), None)
        cat     = meta_df[cat_col].astype(str).values
        y_ep    = (cat == "Critico").astype(np.int64)

        # Label frame binário
        y_fr = ((y_fr_clean > FRAME_THR).astype(np.int64) * y_ep[:, None]).astype(np.int64)
        y_fr_fc = y_fr.copy()

        n_crit = int(y_ep.sum())
        n_norm = int((y_ep == 0).sum())
        print(f"  Total: {len(y_ep)} | Crítico: {n_crit} | Normal: {n_norm}")

        # ── 2) Splits 70/15/15 estratificados por recipe_id ──────────────
        idx = np.arange(len(y_ep), dtype=np.int64)
        strat_key = np.where(
            meta_df[cat_col].astype(str).values == "Critico",
            meta_df["recipe_id"].astype(str).values,
            meta_df[cat_col].astype(str).values,
        )
        tr_idx, tmp = train_test_split(idx, test_size=0.30,
                                       random_state=SEED, stratify=strat_key)
        va_idx, te_idx = train_test_split(tmp, test_size=0.50,
                                          random_state=SEED, stratify=strat_key[tmp])
        X_tr        = X_all[tr_idx]
        yep_tr      = y_ep[tr_idx]
        yfr_tr      = y_fr_fc[tr_idx]       # binário — hard target
        yfr_soft_tr = y_fr_clean[tr_idx]    # CONTÍNUO — soft target (BUG FIX)
        X_va        = X_all[va_idx]
        yep_va      = y_ep[va_idx]
        print(f"  Split — tr:{len(tr_idx)} val:{len(va_idx)} te:{len(te_idx)}")

        # [v23-audit] Distribuição de recipe_id por split (igual ao run_v23.py)
        for split_name, split_idx in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
            sub = meta_df.iloc[split_idx]
            crit_sub = sub[sub[cat_col].astype(str) == "Critico"]
            if len(crit_sub) > 0:
                rc = crit_sub["recipe_id"].value_counts().sort_index()
                counts_str = "  ".join(f"{r}={n}" for r, n in rc.items())
                print(f"  [v23-audit] {split_name:5s} crit={len(crit_sub):3d} | {counts_str}")

        # ── 3) Carregar ou treinar modelos ────────────────────────────────
        ckpt_student  = MODEL_DIR / f"student_seed{SEED}.pt"
        ckpt_teacher  = MODEL_DIR / f"teacher_seed{SEED}.pt"

        use_reuse = reuse and REUSE_TRAINED_MODELS and ckpt_student.exists()
        if use_reuse:
            # ── Carregar pesos do run_v23.py — GARANTE CONSISTÊNCIA ──────
            print(f"\n[v23] Carregando student_seed{SEED}.pt de {MODEL_DIR}/ ...")
            student = MultiTaskLSTM(D, H, bi=False).to(device)
            student.load_state_dict(torch.load(ckpt_student, map_location=device))
            student.eval()
            print(f"  ✓ student carregado — resultados consistentes com artigo")
        else:
            if reuse and not ckpt_student.exists():
                print(f"\n⚠️  Checkpoint não encontrado: {ckpt_student}")
                print(f"   Execute run_v23.py primeiro para gerar os pesos,")
                print(f"   ou use --no_reuse para re-treinar do zero.")
                print(f"   Continuando com re-treino (resultados podem divergir do artigo).\n")

            # ── Treinar professor BiLSTM ──────────────────────────────────
            print(f"\n[2/4] Treinando professor BiLSTM...")
            teacher = MultiTaskLSTM(D, H, bi=True)
            yfr_tr_teacher = np.zeros_like(yfr_tr)
            yfr_tr_teacher[:, :-8] = yfr_tr[:, 8:]
            teacher = train_teacher(teacher, X_tr, yfr_tr_teacher, yep_tr, device)
            torch.save(teacher.state_dict(), ckpt_teacher)
            print(f"  professor salvo: {ckpt_teacher}")

            # ── Treinar aluno AF-KD ───────────────────────────────────────
            print(f"\n[3/4] Treinando aluno AF-KD...")
            student = MultiTaskLSTM(D, H, bi=False)
            student = train_afkd(student, teacher, X_tr, yfr_tr, yep_tr, device,
                                 yfr_soft=yfr_soft_tr)
            torch.save(student.state_dict(), ckpt_student)
            print(f"  student salvo: {ckpt_student}")
            del teacher

        # ── 4) Gerar episódios BORDERLINE para esta seed ──────────────────
        print(f"\n[{'reuse' if use_reuse else '4'}/4] Gerando e inferindo episódios borderline...")
        set_seed(SEED)   # mesma seed → reprodutibilidade
        X_b, _, _, _, metas_b = gen.build_dataset(
            gen.RECIPES_BORDERLINE,
            window     = T,
            per_recipe = per_recipe,
            seed       = SEED,
            shuffle    = False,   # mantém ordem por recipe para análise
            SENSOR_LEVEL = 4,
        )
        n_atencao = sum(1 for m in metas_b if m.get("categoria") == "Atencao")
        n_alerta  = sum(1 for m in metas_b if m.get("categoria") == "Alerta")
        print(f"  Borderline gerado: {len(metas_b)} episódios "
              f"(Atenção:{n_atencao} Alerta:{n_alerta})")

        # ── 6) Inferência frame-a-frame em todos os borderlines ───────────
        print(f"  Rodando inferência (frame-a-frame, causal)...")
        t0 = time.perf_counter()
        fp = infer_episodes(student, X_b, device)
        t1 = time.perf_counter()
        print(f"  Inferência concluída em {t1-t0:.1f}s")

        # ── 7) Montar DataFrame de resultados ─────────────────────────────
        for i, meta in enumerate(metas_b):
            meta["_seed"] = SEED
            meta["episode_idx"] = i
        ep_df_seed = analyse_borderline(fp, metas_b, theta, M_DETECT)
        ep_df_seed["seed"] = SEED
        all_ep_rows.append(ep_df_seed)

        for i in range(len(metas_b)):
            frame_probs_all[(SEED, i)] = fp[i]

        del student  # libera VRAM entre seeds

    # ── Consolidar e salvar ───────────────────────────────────────────────
    ep_df  = pd.concat(all_ep_rows, ignore_index=True)
    agg_df = aggregate(ep_df)

    ep_out  = f"{OUT_PREFIX}_episode_level.csv"
    agg_out = f"{OUT_PREFIX}_summary.csv"
    ep_df.to_csv(ep_out,  index=False)
    agg_df.to_csv(agg_out, index=False)
    print(f"\n  → {ep_out}  ({len(ep_df)} linhas)")
    print(f"  → {agg_out}")

    # ── Figura ────────────────────────────────────────────────────────────
    plot_results(ep_df, frame_probs_all, theta)

    # ── Relatório terminal ────────────────────────────────────────────────
    print_report(ep_df, agg_df, theta)


# ══════════════════════════════════════════════════════════════════════════════
# 10. ENTRADA
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análise borderline AF-KD v23 — GPU")
    parser.add_argument("--seeds",      type=int, nargs="+", default=SEEDS)
    parser.add_argument("--per_recipe", type=int, default=PER_RECIPE)
    parser.add_argument("--theta",      type=float, default=THETA)
    parser.add_argument("--no_reuse",   action="store_true",
                        help="Ignora checkpoints e re-treina do zero (não recomendado)")
    args = parser.parse_args()

    print(f"Seeds       : {args.seeds}")
    print(f"per_recipe  : {args.per_recipe}")
    print(f"θ           : {args.theta}")
    print(f"m (detecção): {M_DETECT}")
    print(f"K_AGG       : {K_AGG}")
    print(f"Reuse models: {not args.no_reuse}")

    main(args.seeds, args.per_recipe, args.theta, reuse=not args.no_reuse)
