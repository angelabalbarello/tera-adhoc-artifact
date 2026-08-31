# -*- coding: utf-8 -*-
"""
run_ablation_teacher_study.py
ABLAÇÃO DE TEACHER — compatibilizada com o TERA Pipeline v1.0

OBJETIVO
Responder à questão de ablação do artigo (Seção 4.4):
  "O ganho vem da supervisão temporal orientada ao onset (princípio
   do AF-TKD) ou do poder representacional do teacher BiLSTM em
   particular?"

MODELOS AVALIADOS (5 configurações × 4 seeds = 20 treinos)
  ┌──────────────────────────┬─────────────┬──────────────────────────────────┐
  │ Config                   │ Teacher     │ Pergunta                         │
  ├──────────────────────────┼─────────────┼──────────────────────────────────┤
  │ LSTM-Baseline            │ —           │ baseline causal sem KD           │
  │ LSTM-AF-KD  ★ proposto   │ BiLSTM      │ teacher recorrente -> LSTM causal │
  │ LSTM-TransKD             │ Transformer │ teacher atencional -> LSTM causal │
  │ Transformer-Baseline     │ —           │ baseline causal Transformer      │
  │ Transformer-AF-KD        │ BiLSTM      │ teacher recorrente -> Trans causal │
  └──────────────────────────┴─────────────┴──────────────────────────────────┘

  Ablação central: LSTM-AF-KD vs LSTM-TransKD
    -> Student (LSTM causal) idêntico; apenas o teacher muda.
    -> Se LSTM-TransKD ≈ LSTM-AF-KD: o princípio de supervisão temporal
      é teacher-agnóstico.
    -> Se LSTM-AF-KD > LSTM-TransKD: a natureza recorrente do professor
      importa para transferência temporal ao aluno causal.

COMPATIBILIDADE COM O TERA PIPELINE
  · Usa MultiTaskLSTM de tera_train.py (mesma arquitetura, ~58K params)
  · Usa aftkd_loss() de tera_train.py (protocolo 3 fases do artigo)
  · Usa synthetic_driver_risk_v7 via tera_gen.py (mesmo gerador)
  · Mesmas seeds, splits estratificados, K_AGG=6, FAIL_BUDGET=0.05
  · TTDef = TTD_bruto + FR × T_EP=10s  (alinhado com macros LaTeX)
  · Adiciona TransformerStudent + TransformerTeacher sem alterar o
    protocolo canônico do TERA para as configs LSTM-*

DIFERENÇAS DELIBERADAS em relação ao run_ablation_v8.py original
  · aftkd_loss() substituído pelo do tera_train.py (protocolo do artigo)
    com parâmetros do experiment_config.yaml
  · MultiTaskLSTM (tera_train) em lugar de LSTMStudent (run_ablation_v8)
    — mesma capacidade, mesma contagem de params, interface compatível
  · Transformer usa TransformerStudent/TransformerTeacher do v8 intocados
    (são novos; não existem no TERA original)
  · Temperature scaling mantido do v8 (pós-treino, 20 passos)
  · Splits: usa tera_gen.DatasetManager para garantir que os splits
    sejam exatamente iguais ao experimento principal

SAÍDAS
  resultados_ablacao_teacher/
  ├─ results_all_seeds.csv            valores por seed × modelo
  ├─ summary.csv                      média ± std por configuração
  ├─ table_teacher_ablation.tex       tabela LaTeX pronta para colar
  └─ table_teacher_ablation_full.tex  tabela completa com TTDef/TTD_bruto

EXECUÇÃO
  # Requer: synthetic_driver_risk_v7.py na raiz
  python run_ablation_teacher_study.py           # todas as seeds
  python run_ablation_teacher_study.py --seed 42 # apenas seed 42 (debug)
  python run_ablation_teacher_study.py --skip-train  # só análise (se já treinou)

TEMPO ESTIMADO
  GPU:  ~25–35 min  (20 treinos × ~1–2 min cada)
  CPU:  ~120–180 min
"""

import argparse
import json
import math
import random
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# 0) CONFIGURAÇÃO GLOBAL
#    Mantida 100% alinhada com experiment_config.yaml e tera_train.py

SEEDS       = [42, 43, 44, 45]
OUTPUT_DIR  = Path("resultados_ablacao_teacher")

# Protocolo canônico do TERA (idêntico ao experiment_config.yaml)
T           = 96       # window
D           = 31       # input_dim
H           = 64       # hidden_dim
K_AGG       = 6        # agregação causal (tera_eval.py linha 40)
TTD_M       = 3        # m consecutivos para detecção estável
FAIL_BUDGET = 0.05     # FR máximo (calibration.fail_rate_budget)
T_EP        = 10.0     # penalidade TTDef (evaluation.t_max_ef)
FRAME_THR   = 0.35     # limiar binário frame-level (dataset.frame_thr)
KIN         = [12, 13, 14]  # model.kinematic_indices
LAT_WARMUP  = 100      # lat_warmup_steps

# Hiperparâmetros AF-TKD do experiment_config.yaml (training.student_aftkd)
AFTKD_CFG = {
    "onset_exp_alpha":       0.15,
    "temp_fixed":            2.0,
    "lambda_late":           3.5,
    "theta_late":            0.15,
    "pre_onset_push_weight": 3.0,
    "hidden_kd_window":      8,
    "warmup_epochs":         10,
    "rampup_epochs":         17,
    "epochs":                80,
    "lr":                    0.001,
    "batch_size":            64,
    "lam_start":             0.20,
    "lam_end":               1.00,
}

# Hiperparâmetros do Transformer causal (novos — não existem no TERA original)
TRANSFORMER_CFG = {
    "input_dim":       D,
    "d_model":         96,
    "nhead":           4,
    "num_layers":      3,
    "dim_feedforward": 192,
    "dropout":         0.10,
    "max_len":         128,
}

# Grade de θ para calibração (idêntica ao run_ablation_v8)
THETA_GRID = list(np.concatenate([
    np.array([0.10, 0.12, 0.15, 0.18, 0.20], dtype=float),
    np.linspace(0.25, 0.90, 14),
]))
THR_EP_GRID = np.concatenate([
    np.linspace(0.05, 0.40, 15),
    np.linspace(0.40, 0.95, 12),
])

DATA_DIR = Path("dados_sinteticos")

# Reprodutibilidade

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 1) MODELOS

# 1a. MultiTaskLSTM — idêntico ao tera_train.py

class MultiTaskLSTM(nn.Module):
    """
    LSTM canônico do TERA Pipeline.
    Idêntico ao MultiTaskLSTM de tera_train.py — mesma interface,
    mesmos ~58K params (H=64, D=31, 2 camadas).
    bi=True -> professor BiLSTM; bi=False -> aluno causal embarcado.
    """
    def __init__(self, input_dim: int = D, hidden_dim: int = H,
                 num_layers: int = 2, dropout: float = 0.1,
                 bi: bool = False):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.bidirectional = bi
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bi,
        )
        factor = 2 if bi else 1
        self.fc_fr = nn.Linear(hidden_dim * factor, 1)
        self.fc_ep = nn.Linear(hidden_dim * factor, 1)

    def forward(self, x: torch.Tensor,
                h: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x, h)
        fr = self.fc_fr(out)           # (B, T, 1)
        ep = self.fc_ep(out.mean(1))   # (B, 1)
        return fr, ep


# 1b. Transformer causal — novo para a ablação

def _causal_mask(T_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(T_len, T_len, dtype=torch.bool, device=device), diagonal=1
    )

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


def _make_encoder(cfg: dict) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=cfg["d_model"], nhead=cfg["nhead"],
        dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"],
        activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=cfg["num_layers"])


class TransformerStudent(nn.Module):
    """
    Transformer causal com atenção mascarada — aluno para ablação.
    Não reside no dispositivo embarcado; serve apenas para comparação.
    ~180K params (d_model=96, 4 heads, 3 layers).
    """
    def __init__(self, cfg: dict = TRANSFORMER_CFG):
        super().__init__()
        self.in_proj = nn.Linear(cfg["input_dim"], cfg["d_model"])
        self.pos_enc = PositionalEncoding(cfg["d_model"], cfg["max_len"])
        self.encoder = _make_encoder(cfg)
        self.norm    = nn.LayerNorm(cfg["d_model"])
        self.fc_fr   = nn.Linear(cfg["d_model"], 1)
        self.fc_ep   = nn.Linear(cfg["d_model"], 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h    = self.pos_enc(self.in_proj(x))
        mask = _causal_mask(x.size(1), x.device)
        h    = self.norm(self.encoder(h, mask=mask))
        return self.fc_fr(h), self.fc_ep(h[:, -1:, :]).squeeze(1)


class TransformerTeacher(nn.Module):
    """
    Transformer full-sequence (sem máscara causal) — oráculo atencional.
    Usado como teacher em LSTM-TransKD.
    Mesmos hiperparâmetros do TransformerStudent para comparação justa.
    """
    def __init__(self, cfg: dict = TRANSFORMER_CFG):
        super().__init__()
        self.in_proj = nn.Linear(cfg["input_dim"], cfg["d_model"])
        self.pos_enc = PositionalEncoding(cfg["d_model"], cfg["max_len"])
        self.encoder = _make_encoder(cfg)
        self.norm    = nn.LayerNorm(cfg["d_model"])
        self.fc_fr   = nn.Linear(cfg["d_model"], 1)
        self.fc_ep   = nn.Linear(cfg["d_model"], 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.pos_enc(self.in_proj(x))
        h = self.norm(self.encoder(h))          # SEM máscara — full-sequence
        return self.fc_fr(h), self.fc_ep(h[:, -1:, :]).squeeze(1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 2) LOSS AF-TKD — importada do tera_train.py via adaptador
#    Usa exatamente os parâmetros do experiment_config.yaml

class _AFTKDConfig:
    """Adaptador que expõe AFTKD_CFG como atributos aninhados (interface tera_train)."""
    class _Sub:
        def __init__(self, d: dict):
            self.onset_exp_alpha       = d["onset_exp_alpha"]
            self.temp_fixed            = d["temp_fixed"]
            self.pre_onset_push_weight = d["pre_onset_push_weight"]
            self.lambda_late           = d["lambda_late"]
            self.theta_late            = d["theta_late"]
            self.hidden_kd_window      = d["hidden_kd_window"]
    def __init__(self):
        self.af_tkd = self._Sub(AFTKD_CFG)

_AFTKD_CONFIG = _AFTKDConfig()


def aftkd_loss(
    student_fr:   torch.Tensor,
    student_ep:   torch.Tensor,
    teacher_fr:   torch.Tensor,
    y_episode:    torch.Tensor,
    onset_frames: torch.Tensor,
    lam_kd:       float = 0.5,
    window:       int   = T,
) -> torch.Tensor:
    """
    Perda AF-TKD idêntica à de tera_train.py.
    Parâmetros injetados de AFTKD_CFG (experiment_config.yaml).
    """
    student_logits = student_fr.squeeze(-1)
    teacher_logits = teacher_fr.squeeze(-1)
    ep_logit       = student_ep.squeeze(-1)

    B    = student_logits.shape[0]
    T_   = student_logits.shape[1]
    cfg  = _AFTKD_CONFIG

    alpha      = cfg.af_tkd.onset_exp_alpha
    temp       = cfg.af_tkd.temp_fixed
    pre_push_w = cfg.af_tkd.pre_onset_push_weight
    lam_late   = cfg.af_tkd.lambda_late
    theta_late = cfg.af_tkd.theta_late
    kd_window  = cfg.af_tkd.hidden_kd_window

    y_ep_float = y_episode.float().squeeze(-1) if y_episode.ndim > 1 else y_episode.float()
    l_hard = F.binary_cross_entropy_with_logits(ep_logit, y_ep_float)

    p_teacher = torch.sigmoid(teacher_logits / temp)
    p_student = torch.sigmoid(student_logits / temp)
    kl = F.binary_cross_entropy(p_student, p_teacher, reduction="none")

    weights   = torch.ones_like(kl)
    crit_mask = (y_ep_float == 1)

    if crit_mask.sum() > 0:
        for i in range(B):
            if crit_mask[i]:
                t0 = int(onset_frames[i].item())
                if t0 > 0:
                    w_ep = torch.ones(T_, device=student_logits.device)
                    for t in range(t0, T_):
                        w_ep[t] = torch.exp(
                            torch.tensor(-alpha * (t - t0),
                                         device=student_logits.device))
                        if torch.sigmoid(student_logits[i, t]) < theta_late:
                            w_ep[t] *= lam_late
                    win_start = max(0, t0 - kd_window)
                    for t in range(win_start, t0):
                        w_ep[t] = pre_push_w
                    w_ep = w_ep / (w_ep.sum() + 1e-8)
                    weights[i] = w_ep

    l_soft = (temp ** 2) * (kl * weights).sum(dim=1).mean()
    return (1 - lam_kd) * l_hard + lam_kd * l_soft


# 3) DATASET — usa o mesmo gerador do TERA principal

def _load_generator():
    import importlib
    try:
        return importlib.import_module("synthetic_driver_risk_v7")
    except ModuleNotFoundError:
        raise ImportError(
            "synthetic_driver_risk_v7.py não encontrado na raiz do projeto.\n"
            "Certifique-se de que o arquivo está presente antes de executar."
        )


def load_dataset(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Carrega (ou gera) dataset via synthetic_driver_risk_v7.
    Interface idêntica à usada pelo DatasetManager do TERA.
    Retorna X, y_frame_bin, y_episode_bin, meta_df.
    """
    npz_path = DATA_DIR / f"dataset_sintetico_seed{seed}.npz"
    csv_path = DATA_DIR / f"episodios_seed{seed}.csv"

    if not (npz_path.exists() and csv_path.exists()):
        print(f"  Gerando dataset seed={seed}...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        gen = _load_generator()
        gen.quick_generate(
            out_npz=str(npz_path),
            out_csv=str(csv_path),
            seed=seed,
            window=T,
            per_recipe=80,
        )

    npz     = np.load(npz_path)
    meta_df = pd.read_csv(csv_path, sep=";")
    X       = npz["X"].astype(np.float32)

    y_fr_cont = npz.get("y_frame_clean", npz.get("y_frame")).astype(np.float32)
    y_fr_bin  = (y_fr_cont >= FRAME_THR).astype(np.float32)

    # Classe: Critico | Normal (filtro binário idêntico ao run_ablation_v8 linha 1518)
    class_col = next(c for c in ("class_name", "categoria", "class") if c in meta_df.columns)
    cat       = meta_df[class_col].astype(str).values
    binary_mask = (cat == "Normal") | (cat == "Critico")
    X        = X[binary_mask]
    y_fr_bin = y_fr_bin[binary_mask]
    meta_df  = meta_df[binary_mask].reset_index(drop=True)
    cat      = cat[binary_mask]

    y_ep_bin  = (y_fr_bin.max(axis=1) > 0).astype(np.float32)

    n_norm = int((cat == "Normal").sum())
    n_crit = int((cat == "Critico").sum())
    print(f"  [dataset] seed={seed}  Normal={n_norm}  Crítico={n_crit}  Total={len(meta_df)}")
    return X, y_fr_bin, y_ep_bin, meta_df


def make_splits(
    X: np.ndarray,
    y_fr: np.ndarray,
    y_ep: np.ndarray,
    meta: pd.DataFrame,
    seed: int,
) -> Dict[str, Any]:
    """
    Splits 70/15/15 estratificados por recipe_id nos críticos —
    protocolo idêntico ao DatasetManager de tera_gen.py.
    """
    class_col = next(c for c in ("class_name", "categoria", "class") if c in meta.columns)
    cat = meta[class_col].astype(str).values
    strat_key = np.where(
        cat == "Critico",
        meta["recipe_id"].astype(str).values if "recipe_id" in meta.columns else cat,
        cat,
    )
    idx = np.arange(len(X))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.30,
                                        random_state=seed, stratify=strat_key)
    va_idx, te_idx  = train_test_split(tmp_idx, test_size=0.50,
                                        random_state=seed, stratify=strat_key[tmp_idx])

    progressive = (
        meta["progressive"].fillna(0).astype(int).values.astype(bool)
        if "progressive" in meta.columns else np.zeros(len(meta), dtype=bool)
    )
    progressive = progressive & (y_ep == 1)

    return {
        "tr": (tr_idx, X[tr_idx], y_fr[tr_idx], y_ep[tr_idx], progressive[tr_idx]),
        "va": (va_idx, X[va_idx], y_fr[va_idx], y_ep[va_idx], progressive[va_idx]),
        "te": (te_idx, X[te_idx], y_fr[te_idx], y_ep[te_idx], progressive[te_idx]),
    }


# 4) TREINAMENTO

def _onset_frames_tensor(y_fr: np.ndarray, window: int = T) -> torch.Tensor:
    onsets = []
    for yt in y_fr:
        idx = np.where(yt > 0.5)[0]
        onsets.append(int(idx[0]) if len(idx) > 0 else window - 1)
    return torch.tensor(onsets, dtype=torch.long)


def train_supervised(
    model: nn.Module,
    X_tr: np.ndarray,
    y_fr_tr: np.ndarray,
    y_ep_tr: np.ndarray,
    device: torch.device,
    epochs: int = 60,
    lr: float = 0.001,
    batch_size: int = 64,
    tag: str = "",
) -> nn.Module:
    """
    Treinamento supervisionado padrão (sem KD).
    Usado para: teacher BiLSTM, LSTM-Baseline, Transformer-Baseline,
    TransformerTeacher.
    Perda: BCE frame + BCE episódio (idêntico ao tera_train.py linha 294).
    """
    from torch.utils.data import DataLoader, TensorDataset
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xt  = torch.tensor(X_tr,    dtype=torch.float32)
    yft = torch.tensor(y_fr_tr, dtype=torch.float32)
    yet = torch.tensor(y_ep_tr, dtype=torch.float32)

    loader = DataLoader(TensorDataset(Xt, yft, yet),
                        batch_size=batch_size, shuffle=True)

    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb, yf, ye in loader:
            xb, yf, ye = xb.to(device), yf.to(device), ye.to(device)
            fr_log, ep_log = model(xb)
            fr_log = fr_log.squeeze(-1)
            ep_log = ep_log.squeeze(-1)
            loss = (F.binary_cross_entropy_with_logits(fr_log, yf)
                    + F.binary_cross_entropy_with_logits(ep_log, ye)) / 2.0
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if (ep + 1) % 20 == 0:
            print(f"    {tag} ep={ep+1:3d}/{epochs}  loss={total/len(loader):.4f}")
    return model


def train_aftkd_student(
    student: nn.Module,
    teacher: nn.Module,
    X_tr: np.ndarray,
    y_fr_tr: np.ndarray,
    y_ep_tr: np.ndarray,
    device: torch.device,
    tag: str = "",
) -> nn.Module:
    """
    Treinamento AF-TKD com protocolo 3 fases.
    Usa aftkd_loss() com parâmetros de AFTKD_CFG (experiment_config.yaml).
    Compatível com qualquer par (teacher, student):
      - BiLSTM -> LSTM          (LSTM-AF-KD, config canônica do artigo)
      - TransformerTeacher -> LSTM  (LSTM-TransKD, ablação)
      - BiLSTM -> TransformerStudent  (Transformer-AF-KD, ablação)
    """
    from torch.utils.data import DataLoader, TensorDataset
    cfg = AFTKD_CFG

    student.to(device).train()
    teacher.to(device).eval()
    opt = torch.optim.Adam(student.parameters(), lr=cfg["lr"])

    onset_t = _onset_frames_tensor(y_fr_tr)
    Xt  = torch.tensor(X_tr,    dtype=torch.float32)
    yft = torch.tensor(y_fr_tr, dtype=torch.float32)
    yet = torch.tensor(y_ep_tr, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(Xt, yft, yet, onset_t),
        batch_size=cfg["batch_size"], shuffle=True,
    )

    warmup_ep = cfg["warmup_epochs"]
    rampup_ep = cfg["rampup_epochs"]
    total_ep  = cfg["epochs"]
    lam_start = cfg["lam_start"]
    lam_end   = cfg["lam_end"]

    for ep in range(total_ep):
        if ep < warmup_ep:
            lam_kd = 0.0
            phase  = "warmup"
        elif ep < warmup_ep + rampup_ep:
            frac   = (ep - warmup_ep) / rampup_ep
            lam_kd = lam_start + frac * (lam_end - lam_start)
            phase  = "rampup"
        else:
            lam_kd = lam_end
            phase  = "steady"

        student.train()
        total = 0.0
        for xb, yf, ye, ons in loader:
            xb, yf, ye, ons = xb.to(device), yf.to(device), ye.to(device), ons.to(device)
            s_fr, s_ep = student(xb)
            with torch.no_grad():
                t_fr, _ = teacher(xb)
            loss = aftkd_loss(s_fr, s_ep, t_fr, ye, ons, lam_kd=lam_kd, window=T)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()

        if (ep + 1) % 20 == 0:
            print(f"    {tag} ep={ep+1:3d}/{total_ep} [{phase:6s}]  "
                  f"λ={lam_kd:.3f}  loss={total/len(loader):.4f}")
    return student


# Temperature scaling pós-treino (idêntico ao run_ablation_v8)

class TScaledModel(nn.Module):
    def __init__(self, base: nn.Module, temperature: float):
        super().__init__()
        self.base = base
        self.T    = float(temperature)

    def forward(self, x: torch.Tensor):
        fr_log, ep_log = self.base(x)
        return fr_log / self.T, ep_log


def temperature_scale(
    model: nn.Module,
    X_val: np.ndarray,
    y_fr_val: np.ndarray,
    device: torch.device,
    n_steps: int = 20,
    lr: float = 0.01,
) -> TScaledModel:
    model.to(device).eval()
    T_param = nn.Parameter(torch.tensor(1.5, device=device, dtype=torch.float32))
    opt = torch.optim.Adam([T_param], lr=lr)
    Xv  = torch.tensor(X_val,   device=device, dtype=torch.float32)
    yv  = torch.tensor(y_fr_val, device=device, dtype=torch.float32)
    with torch.no_grad():
        fr_log, _ = model(Xv)
        fr_log = fr_log.squeeze(-1)
    for _ in range(n_steps):
        opt.zero_grad()
        Tc = T_param.clamp(0.5, 5.0)
        loss = F.binary_cross_entropy_with_logits(fr_log / Tc, yv)
        loss.backward(); opt.step()
    T_final = float(T_param.clamp(0.5, 5.0).detach().cpu())
    print(f"    temperature_scale: T={T_final:.4f}")
    return TScaledModel(model, T_final)


# 5) INFERÊNCIA E AVALIAÇÃO

def infer_fixed(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    warmup: int = LAT_WARMUP,
) -> Tuple[np.ndarray, float]:
    """
    Inferência episódio-a-episódio (batch=1).
    Retorna (frame_probs, lat_ms).
    Protocolo idêntico ao tera_infer.infer_stream_fixed com make_prefix_window.
    """
    model.to(device).eval()
    N, T_, D_ = X.shape
    all_probs = np.zeros((N, T_), dtype=np.float32)
    lat_times: List[float] = []

    def _make_prefix(x_seq: np.ndarray, t: int) -> np.ndarray:
        sl = x_seq[:t + 1, :]
        if sl.shape[0] < T_:
            pad = np.repeat(sl[:1, :], T_ - sl.shape[0], axis=0)
            sl  = np.concatenate([pad, sl], axis=0)
        else:
            sl = sl[-T_:]
        return sl.astype(np.float32)

    # warm-up
    with torch.no_grad():
        x_ref = torch.tensor(_make_prefix(X[0], 5), dtype=torch.float32,
                              device=device).unsqueeze(0)
        for _ in range(warmup):
            _ = model(x_ref)
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for ep in range(N):
            for t in range(T_):
                xt = torch.tensor(_make_prefix(X[ep], t), dtype=torch.float32,
                                   device=device).unsqueeze(0)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0_ = time.perf_counter()
                fr_log, _ = model(xt)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                lat_times.append(time.perf_counter() - t0_)
                all_probs[ep, t] = float(torch.sigmoid(fr_log[0, -1, 0]).item())

    lat_ms = float(np.mean(lat_times)) * 1000.0
    return all_probs, lat_ms


def episode_probs_k6(frame_probs: np.ndarray, k: int = K_AGG) -> np.ndarray:
    """Agregação causal: média dos últimos K_AGG frames (tera_eval linha 108)."""
    return np.mean(frame_probs[:, -k:], axis=1)


def fail_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    pos = int((y_true == 1).sum())
    if pos == 0:
        return 0.0
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return float(fn / pos)


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    pf, tf = probs.flatten(), targets.flatten()
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1],
                      np.linspace(0, 1, n_bins + 1)[1:]):
        mask = (pf >= lo) & (pf < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(pf) * abs(tf[mask].mean() - pf[mask].mean())
    return float(ece)


def _first_stable(probs: np.ndarray, onset: int, thr: float, m: int) -> Optional[int]:
    count = 0
    for t in range(int(onset), len(probs)):
        if probs[t] >= thr:
            count += 1
            if count >= m:
                return t - m + 1
        else:
            count = 0
    return None


def select_theta(
    frame_probs_val: np.ndarray,
    y_fr_val: np.ndarray,
    m: int = TTD_M,
) -> float:
    """
    Calibra θ_TTD: minimiza TTD com FailRate ≤ FAIL_BUDGET.
    Protocolo idêntico ao select_theta_per_model do run_ablation_v8.
    """
    dt = 10.0 / T
    feasible, fallback = [], []
    for theta in THETA_GRID:
        ttds, misses, positives = [], 0, 0
        for yt, yp in zip(y_fr_val, frame_probs_val):
            onset_arr = np.where(yt > 0)[0]
            if len(onset_arr) == 0:
                continue
            positives += 1
            t0  = int(onset_arr[0])
            det = _first_stable(yp, t0, float(theta), m)
            if det is None:
                misses += 1
                ttds.append((T - t0) * dt)
            else:
                ttds.append((det - t0) * dt)
        fr  = float(misses / max(positives, 1))
        ttd = float(np.mean(ttds)) if ttds else 1e9
        prox = abs(float(theta) - 0.30)
        fallback.append((fr, ttd, prox, float(theta)))
        if fr <= FAIL_BUDGET:
            feasible.append((ttd, fr, prox, float(theta)))
    if feasible:
        feasible.sort()
        return float(feasible[0][3])
    fallback.sort()
    return float(fallback[0][3]) if fallback else 0.5


def select_thr_ep(
    frame_probs_val: np.ndarray,
    y_ep_val: np.ndarray,
) -> float:
    """
    Calibra thr_ep no VAL: maximiza F1 com FR ≤ FAIL_BUDGET.
    Idêntico ao tera_eval.select_thr_ep.
    """
    ep_probs = episode_probs_k6(frame_probs_val)
    feasible, fallback = [], []
    for thr in THR_EP_GRID:
        pred = (ep_probs >= float(thr)).astype(np.int64)
        fr_v = fail_rate(y_ep_val, pred)
        f1v  = float(f1_score(y_ep_val, pred, zero_division=0))
        fallback.append((fr_v, -f1v, abs(float(thr) - 0.30), float(thr)))
        if fr_v <= FAIL_BUDGET:
            feasible.append((-f1v, fr_v, abs(float(thr) - 0.30), float(thr)))
    if feasible:
        feasible.sort()
        return float(feasible[0][3])
    fallback.sort()
    return float(fallback[0][3]) if fallback else 0.5


def evaluate_model(
    frame_probs: np.ndarray,
    lat_ms: float,
    y_ep: np.ndarray,
    y_fr: np.ndarray,
    progressive: np.ndarray,
    thr_ep: float,
    theta: float,
    m: int = TTD_M,
) -> Dict[str, float]:
    """
    Calcula F1, FR, TTDef, TTD_bruto, ECE, Lat_ms.
    TTDef = TTD_bruto + FR × T_EP  (alinhado com macros LaTeX do artigo).
    """
    dt = 10.0 / T
    ep_probs = episode_probs_k6(frame_probs)
    ep_hat   = (ep_probs >= thr_ep).astype(np.int64)

    f1v = float(f1_score(y_ep.astype(int), ep_hat, zero_division=0))
    fr  = fail_rate(y_ep.astype(int), ep_hat)
    ece = compute_ece(ep_probs, y_ep.astype(float))

    # TTD bruto (somente detectados — análise auxiliar)
    ttds_det = []
    # TTD_efetivo via penalidade (todos os positivos)
    ttds_all, ttds_prog, ttds_abr = [], [], []
    misses, positives = 0, 0
    abrupt = (~progressive) & (y_ep.astype(bool))

    for i, (yt, yp) in enumerate(zip(y_fr, frame_probs)):
        onset_arr = np.where(yt > 0)[0]
        if len(onset_arr) == 0:
            continue
        positives += 1
        t0  = int(onset_arr[0])
        det = _first_stable(yp, t0, theta, m)
        pen = (T - t0) * dt
        if det is not None:
            ttd = (det - t0) * dt
            ttds_det.append(ttd)
        else:
            misses += 1
            ttd = pen
        ttds_all.append(ttd)
        if progressive[i]:
            ttds_prog.append(ttd)
        elif abrupt[i]:
            ttds_abr.append(ttd)

    ttd_bruto = float(np.mean(ttds_det)) if ttds_det else float("nan")
    ttd_ef    = float(np.mean(ttds_all)) if ttds_all else float("nan")
    ttdef     = (round(ttd_bruto + fr * T_EP, 4)
                 if not math.isnan(ttd_bruto) else float("nan"))

    return {
        "F1":              round(f1v, 4),
        "FailRate":        round(fr,  4),
        "ECE":             round(ece, 4),
        "TTDef":           ttdef,
        "TTD_efetivo":     round(ttd_ef,    4) if not math.isnan(ttd_ef)    else float("nan"),
        "TTD_bruto":       round(ttd_bruto, 4) if not math.isnan(ttd_bruto) else float("nan"),
        "TTD_Progressive": round(float(np.mean(ttds_prog)), 4) if ttds_prog else float("nan"),
        "TTD_Abrupt":      round(float(np.mean(ttds_abr)),  4) if ttds_abr  else float("nan"),
        "N_Progressive":   int(progressive.sum()),
        "N_Abrupt":        int(abrupt.sum()),
        "Lat_ms":          round(lat_ms, 4),
    }


# 6) ORQUESTRAÇÃO — UMA SEED

def run_seed(seed: int, device: torch.device, models_cache: Path) -> List[Dict]:
    """
    Executa protocolo completo para uma seed.
    Retorna lista de dicts com métricas por modelo.
    """
    set_seed(seed)
    print(f"\n{'='*72}")
    print(f"  SEED {seed}")
    print(f"{'='*72}")

    # 1. Dataset
    print("\n1. Dataset...")
    X, y_fr, y_ep, meta = load_dataset(seed)
    splits = make_splits(X, y_fr, y_ep, meta, seed)

    _, X_tr,  y_fr_tr,  y_ep_tr,  prog_tr  = splits["tr"]
    _, X_va,  y_fr_va,  y_ep_va,  prog_va  = splits["va"]
    _, X_te,  y_fr_te,  y_ep_te,  prog_te  = splits["te"]

    print(f"   Train={len(X_tr)}  Val={len(X_va)}  Test={len(X_te)}")

    def _ckpt(name: str) -> Path:
        p = models_cache / f"{name}_seed{seed}.pt"
        return p

    def _save(model: nn.Module, name: str):
        p = _ckpt(name)
        # TScaledModel: salva estado do base + temperatura
        if isinstance(model, TScaledModel):
            torch.save({"base": model.base.state_dict(), "T": model.T}, p)
        else:
            torch.save(model.state_dict(), p)

    def _exists(name: str) -> bool:
        return _ckpt(name).exists()

    # 2. Teacher BiLSTM
    print("\n2. Teacher BiLSTM...")
    bilstm = MultiTaskLSTM(D, H, bi=True)
    if _exists("bilstm_teacher"):
        bilstm.load_state_dict(torch.load(_ckpt("bilstm_teacher"), map_location=device))
        print("   [cache]")
    else:
        bilstm = train_supervised(bilstm, X_tr, y_fr_tr, y_ep_tr, device,
                                   epochs=60, tag="BiLSTM-teacher")
        _save(bilstm, "bilstm_teacher")
    bilstm.to(device).eval()

    # 3. TransformerTeacher
    print("\n3. TransformerTeacher (oráculo full-sequence)...")
    trans_teacher = TransformerTeacher()
    if _exists("trans_teacher"):
        trans_teacher.load_state_dict(torch.load(_ckpt("trans_teacher"), map_location=device))
        print("   [cache]")
    else:
        trans_teacher = train_supervised(trans_teacher, X_tr, y_fr_tr, y_ep_tr,
                                          device, epochs=60, tag="Trans-teacher")
        _save(trans_teacher, "trans_teacher")
    trans_teacher.to(device).eval()

    # 4. LSTM-Baseline
    print("\n4. LSTM-Baseline...")
    lstm_base = MultiTaskLSTM(D, H, bi=False)
    if _exists("lstm_base_ts"):
        ckpt = torch.load(_ckpt("lstm_base_ts"), map_location=device)
        lstm_base.load_state_dict(ckpt["base"])
        lstm_base = TScaledModel(lstm_base, ckpt["T"])
        print("   [cache]")
    else:
        lstm_base = train_supervised(lstm_base, X_tr, y_fr_tr, y_ep_tr, device,
                                      epochs=60, tag="LSTM-Baseline")
        lstm_base = temperature_scale(lstm_base, X_va, y_fr_va, device)
        _save(lstm_base, "lstm_base_ts")
    lstm_base.to(device).eval()

    # 5. LSTM-AF-KD (BiLSTM -> LSTM) ★ canônico do artigo
    print("\n5. LSTM-AF-KD [BiLSTM -> LSTM]...")
    lstm_afkd = MultiTaskLSTM(D, H, bi=False)
    if _exists("lstm_afkd_ts"):
        ckpt = torch.load(_ckpt("lstm_afkd_ts"), map_location=device)
        lstm_afkd.load_state_dict(ckpt["base"])
        lstm_afkd = TScaledModel(lstm_afkd, ckpt["T"])
        print("   [cache]")
    else:
        lstm_afkd = train_aftkd_student(lstm_afkd, bilstm, X_tr, y_fr_tr, y_ep_tr,
                                         device, tag="LSTM-AF-KD")
        lstm_afkd = temperature_scale(lstm_afkd, X_va, y_fr_va, device)
        _save(lstm_afkd, "lstm_afkd_ts")
    lstm_afkd.to(device).eval()

    # 6. LSTM-TransKD (Transformer -> LSTM) ★ ablação central
    print("\n6. LSTM-TransKD [Transformer -> LSTM]  ★ ablação central...")
    lstm_transkd = MultiTaskLSTM(D, H, bi=False)
    if _exists("lstm_transkd_ts"):
        ckpt = torch.load(_ckpt("lstm_transkd_ts"), map_location=device)
        lstm_transkd.load_state_dict(ckpt["base"])
        lstm_transkd = TScaledModel(lstm_transkd, ckpt["T"])
        print("   [cache]")
    else:
        lstm_transkd = train_aftkd_student(lstm_transkd, trans_teacher,
                                            X_tr, y_fr_tr, y_ep_tr,
                                            device, tag="LSTM-TransKD")
        lstm_transkd = temperature_scale(lstm_transkd, X_va, y_fr_va, device)
        _save(lstm_transkd, "lstm_transkd_ts")
    lstm_transkd.to(device).eval()

    # 7. Transformer-Baseline
    print("\n7. Transformer-Baseline (causal)...")
    trans_base = TransformerStudent()
    if _exists("trans_base_ts"):
        ckpt = torch.load(_ckpt("trans_base_ts"), map_location=device)
        trans_base.load_state_dict(ckpt["base"])
        trans_base = TScaledModel(trans_base, ckpt["T"])
        print("   [cache]")
    else:
        trans_base = train_supervised(trans_base, X_tr, y_fr_tr, y_ep_tr,
                                       device, epochs=60, tag="Trans-Baseline")
        trans_base = temperature_scale(trans_base, X_va, y_fr_va, device)
        _save(trans_base, "trans_base_ts")
    trans_base.to(device).eval()

    # 8. Transformer-AF-KD (BiLSTM -> Transformer causal)
    print("\n8. Transformer-AF-KD [BiLSTM -> Transformer]...")
    trans_afkd = TransformerStudent()
    if _exists("trans_afkd_ts"):
        ckpt = torch.load(_ckpt("trans_afkd_ts"), map_location=device)
        trans_afkd.load_state_dict(ckpt["base"])
        trans_afkd = TScaledModel(trans_afkd, ckpt["T"])
        print("   [cache]")
    else:
        trans_afkd = train_aftkd_student(trans_afkd, bilstm,
                                          X_tr, y_fr_tr, y_ep_tr,
                                          device, tag="Trans-AF-KD")
        trans_afkd = temperature_scale(trans_afkd, X_va, y_fr_va, device)
        _save(trans_afkd, "trans_afkd_ts")
    trans_afkd.to(device).eval()

    # 9. Calibração: θ_TTD e thr_ep por modelo
    print("\n9. Calibração (VAL)...")
    named_models = {
        "LSTM-Baseline":       lstm_base,
        "LSTM-AF-KD":          lstm_afkd,
        "LSTM-TransKD":        lstm_transkd,
        "Transformer-Baseline": trans_base,
        "Transformer-AF-KD":   trans_afkd,
    }

    theta_map: Dict[str, float]  = {}
    thr_ep_map: Dict[str, float] = {}
    val_probs: Dict[str, np.ndarray] = {}

    for name, mdl in named_models.items():
        fp_va, _ = infer_fixed(mdl, X_va, device, warmup=50)
        val_probs[name]  = fp_va
        theta_map[name]  = select_theta(fp_va, y_fr_va)
        thr_ep_map[name] = select_thr_ep(fp_va, y_ep_va)
        print(f"   {name:28s}  θ={theta_map[name]:.3f}  thr_ep={thr_ep_map[name]:.3f}")

    # 10. Inferência e avaliação no TEST
    print("\n10. Inferência e avaliação (TEST)...")
    rows = []
    for name, mdl in named_models.items():
        print(f"   Avaliando {name}...")
        fp_te, lat_ms = infer_fixed(mdl, X_te, device, warmup=LAT_WARMUP)
        metrics = evaluate_model(
            frame_probs=fp_te,
            lat_ms=lat_ms,
            y_ep=y_ep_te,
            y_fr=y_fr_te,
            progressive=prog_te,
            thr_ep=thr_ep_map[name],
            theta=theta_map[name],
        )
        # Identifica Student e Teacher
        parts   = name.split("-")
        student = parts[0]
        method  = "-".join(parts[1:]) if len(parts) > 1 else "Baseline"
        teacher_map = {
            "LSTM-Baseline":       "None",
            "LSTM-AF-KD":          "BiLSTM",
            "LSTM-TransKD":        "Transformer",
            "Transformer-Baseline": "None",
            "Transformer-AF-KD":   "BiLSTM",
        }
        row = {
            "Seed":    seed,
            "Config":  name,
            "Student": student,
            "Metodo":  method,
            "Teacher": teacher_map[name],
            "Params":  count_params(mdl.base if isinstance(mdl, TScaledModel) else mdl),
            **metrics,
        }
        rows.append(row)
        print(f"     F1={metrics['F1']:.4f}  FR={metrics['FailRate']:.4f}  "
              f"TTDef={metrics['TTDef']:.3f}s  Lat={lat_ms:.3f}ms/q")

    return rows


# 7) AGREGAÇÃO E EXPORTAÇÃO LATEX

# Ordem canônica para as tabelas do artigo
_TABLE_ORDER = [
    "LSTM-Baseline",
    "LSTM-AF-KD",
    "LSTM-TransKD",
    "Transformer-Baseline",
    "Transformer-AF-KD",
]

# Labels para exibição na tabela LaTeX
_LABELS = {
    "LSTM-Baseline":        ("LSTM",        "Baseline",  "—"),
    "LSTM-AF-KD":           ("LSTM",        "AF-KD",     "BiLSTM"),
    "LSTM-TransKD":         ("LSTM",        "TransKD",   "Transformer"),
    "Transformer-Baseline": ("Transformer", "Baseline",  "—"),
    "Transformer-AF-KD":    ("Transformer", "AF-KD",     "BiLSTM"),
}


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["F1", "FailRate", "ECE", "TTDef", "TTD_efetivo",
                   "TTD_bruto", "TTD_Progressive", "TTD_Abrupt", "Lat_ms", "Params"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    rows = []
    for config in _TABLE_ORDER:
        sub = df[df["Config"] == config]
        if sub.empty:
            continue
        rec = {"Config": config, "N_Seeds": len(sub)}
        for mc in metric_cols:
            rec[f"{mc}_mean"] = round(float(sub[mc].mean()), 4)
            rec[f"{mc}_std"]  = round(float(sub[mc].std()),  4)
        rows.append(rec)
    return pd.DataFrame(rows)


def export_latex_compact(df_agg: pd.DataFrame, out_path: Path) -> None:
    """
    Tabela compacta para colar diretamente no artigo (Seção 4.4).
    Colunas: Student | Teacher | F1 | FR | TTDef | ECE
    """
    lines = [
        "% Ablação de Teacher — gerado automaticamente por run_ablation_teacher_study.py",
        "% Student idêntico (LSTM causal) nas três primeiras linhas.",
        "% Ablação central: LSTM-AF-KD vs LSTM-TransKD (teacher varia; student fixo).",
        "\\begin{tabular}{lllcccc}",
        "\\toprule",
        "\\textbf{Student} & \\textbf{Teacher} & \\textbf{Método}"
        " & \\textbf{F1}$\\uparrow$"
        " & \\textbf{FR}$\\downarrow$"
        " & \\textbf{TTD$_{\\text{ef}}$(s)}$\\downarrow$"
        " & \\textbf{ECE}$\\downarrow$ \\\\",
        "\\midrule",
    ]

    idx = df_agg.set_index("Config")
    for i, config in enumerate(_TABLE_ORDER):
        if config not in idx.index:
            continue
        r   = idx.loc[config]
        st, meth, teach = _LABELS[config]

        # Negrito na linha canônica do artigo (LSTM-AF-KD)
        bold_open  = "\\textbf{" if config == "LSTM-AF-KD" else ""
        bold_close = "}"         if config == "LSTM-AF-KD" else ""

        # Separador visual entre famílias LSTM / Transformer
        if config == "Transformer-Baseline":
            lines.append("\\midrule")

        f1_s   = f"{r['F1_mean']:.3f}\\,{{\\tiny$\\pm${r['F1_std']:.3f}}}"
        fr_s   = f"{r['FailRate_mean']:.3f}\\,{{\\tiny$\\pm${r['FailRate_std']:.3f}}}"
        ttd_s  = f"{r['TTDef_mean']:.3f}\\,{{\\tiny$\\pm${r['TTDef_std']:.3f}}}"
        ece_s  = f"{r['ECE_mean']:.3f}\\,{{\\tiny$\\pm${r['ECE_std']:.3f}}}"

        lines.append(
            f"{bold_open}{st}{bold_close} & "
            f"{bold_open}{teach}{bold_close} & "
            f"{bold_open}{meth}{bold_close} & "
            f"{bold_open}{f1_s}{bold_close} & "
            f"{bold_open}{fr_s}{bold_close} & "
            f"{bold_open}{ttd_s}{bold_close} & "
            f"{bold_open}{ece_s}{bold_close} \\\\"
        )

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "",
        "% Nota: TTD$_{\\text{ef}}$ = TTD$_{\\text{bruto}}$ + FR $\\times$ 10\\,s",
        "% (penalidade para episódios não detectados; Eq.~\\ref{eq:ttdef_sys}).",
        "% $n{=}4$ sementes. Negrito: configuração canônica do artigo.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Tabela compacta (artigo): {out_path}")


def export_latex_full(df_agg: pd.DataFrame, out_path: Path) -> None:
    """
    Tabela completa com todas as métricas — para material suplementar ou revisão.
    """
    lines = [
        "% Ablação de Teacher — tabela completa",
        "\\begin{tabular}{lllccccccc}",
        "\\toprule",
        "\\textbf{Student} & \\textbf{Teacher} & \\textbf{Método}"
        " & \\textbf{F1} & \\textbf{FR}"
        " & \\textbf{TTD$_{\\text{ef}}$} & \\textbf{TTD$_b$}"
        " & \\textbf{TTD$_p$} & \\textbf{ECE} & \\textbf{ms/q} \\\\",
        "\\midrule",
    ]
    idx = df_agg.set_index("Config")
    for config in _TABLE_ORDER:
        if config not in idx.index:
            continue
        r = idx.loc[config]
        st, meth, teach = _LABELS[config]
        if config == "Transformer-Baseline":
            lines.append("\\midrule")

        def _fmt(col: str) -> str:
            m = r.get(f"{col}_mean", float("nan"))
            s = r.get(f"{col}_std",  float("nan"))
            if math.isnan(m):
                return "---"
            return f"{m:.3f}{{\\tiny$\\pm${s:.3f}}}"

        lines.append(
            f"{st} & {teach} & {meth} & "
            f"{_fmt('F1')} & {_fmt('FailRate')} & "
            f"{_fmt('TTDef')} & {_fmt('TTD_bruto')} & "
            f"{_fmt('TTD_Progressive')} & {_fmt('ECE')} & {_fmt('Lat_ms')} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Tabela completa: {out_path}")


def print_terminal_summary(df_agg: pd.DataFrame) -> None:
    """Imprime resumo formatado no terminal após execução."""
    sep = "─" * 80
    print(f"\n{'═'*80}")
    print("  ABLAÇÃO DE TEACHER — RESULTADO FINAL")
    print(f"{'═'*80}")
    print(f"  {'Config':<28} {'F1':>7} {'FR':>7} {'TTDef':>8} {'ECE':>7} {'ms/q':>7}")
    print(sep)
    idx = df_agg.set_index("Config")
    for config in _TABLE_ORDER:
        if config not in idx.index:
            continue
        r   = idx.loc[config]
        tag = " ★" if config == "LSTM-AF-KD" else \
              " ▲" if config == "LSTM-TransKD" else "  "
        marker = " [ablação central]" if config in ("LSTM-AF-KD", "LSTM-TransKD") else ""
        f1    = f"{r['F1_mean']:.3f}±{r['F1_std']:.3f}"
        fr    = f"{r['FailRate_mean']:.3f}±{r['FailRate_std']:.3f}"
        ttdef = f"{r['TTDef_mean']:.3f}±{r['TTDef_std']:.3f}"
        ece   = f"{r['ECE_mean']:.3f}±{r['ECE_std']:.3f}"
        lat   = f"{r['Lat_ms_mean']:.3f}"
        print(f"{tag} {config:<28} {f1:>11} {fr:>11} {ttdef:>12} {ece:>11} {lat:>7}{marker}")
    print(sep)
    print("\n  ★ = configuração canônica do artigo (LSTM + BiLSTM teacher)")
    print("  ▲ = ablação central (mesmo student, teacher atencional)")
    print("  Pergunta: LSTM-AF-KD ≈ LSTM-TransKD? -> princípio teacher-agnóstico")
    print(f"{'═'*80}\n")


# 8) MAIN

def main():
    parser = argparse.ArgumentParser(
        description="Ablação de Teacher — TERA Pipeline compatível"
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Rodar apenas esta seed (debug). Ex: --seed 42")
    parser.add_argument("--skip-train", action="store_true",
                        help="Pular treino; usar CSV já salvo.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models_cache = OUTPUT_DIR / "models_cache"
    models_cache.mkdir(parents=True, exist_ok=True)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds_run  = [args.seed] if args.seed is not None else SEEDS

    print("═" * 72)
    print("  ABLAÇÃO DE TEACHER — TERA Pipeline v1.0")
    print(f"  Device: {device}  |  Seeds: {seeds_run}")
    print(f"  AF-TKD config: α={AFTKD_CFG['onset_exp_alpha']}  "
          f"T={AFTKD_CFG['temp_fixed']}  "
          f"λ_late={AFTKD_CFG['lambda_late']}")
    print("═" * 72)

    csv_path = OUTPUT_DIR / "results_all_seeds.csv"

    # Fase 1: Treino e avaliação
    if not args.skip_train:
        all_rows: List[Dict] = []
        for seed in seeds_run:
            rows = run_seed(seed, device, models_cache)
            all_rows.extend(rows)

        df = pd.DataFrame(all_rows)
        df.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"\nresults_all_seeds.csv -> {csv_path}")
    else:
        if not csv_path.exists():
            print(f"✗ {csv_path} não encontrado. Execute sem --skip-train primeiro.")
            sys.exit(1)
        df = pd.read_csv(csv_path)
        print(f"Resultados carregados: {csv_path}")

    # Fase 2: Agregação
    df_agg = aggregate(df)
    df_agg.to_csv(OUTPUT_DIR / "summary.csv", index=False, float_format="%.6f")
    print(f"summary.csv -> {OUTPUT_DIR / 'summary.csv'}")

    # Fase 3: Exportação LaTeX
    export_latex_compact(df_agg, OUTPUT_DIR / "table_teacher_ablation.tex")
    export_latex_full(df_agg,    OUTPUT_DIR / "table_teacher_ablation_full.tex")

    # Fase 4: Resumo no terminal
    print_terminal_summary(df_agg)

    print(f"Saídas em: {OUTPUT_DIR}/")
    print(f"  ├─ results_all_seeds.csv")
    print(f"  ├─ summary.csv")
    print(f"  ├─ table_teacher_ablation.tex      <- colar na Seção 4.4 do artigo")
    print(f"  └─ table_teacher_ablation_full.tex <- versão completa")


if __name__ == "__main__":
    main()
