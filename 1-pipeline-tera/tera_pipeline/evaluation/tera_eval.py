# -*- coding: utf-8 -*-
"""
tera_pipeline/evaluation/tera_eval.py
═══════════════════════════════════════════════════════════════════════════════
Engine de avaliação canônica do TERA Pipeline.

CORREÇÕES APLICADAS (vs versão anterior):
  [FIX-2] ep_probs = mean(frame_probs[:, -K_AGG:]) com K_AGG=6.
          Versão anterior: probs.max(axis=1) → F1≈1, FR≈0 trivialmente.
          Adicionado episode_probs_from_frames() idêntico ao run_v29.

  [FIX-3] select_thr_ep() calibra thr_ep no VAL maximizando F1 com
          FR ≤ FAIL_BUDGET=0.05. Versão anterior usava thr_ep=0.050 fixo
          (confundindo FAIL_BUDGET com thr_ep).
          _calibrate_thr_ep() usa VAL frame_probs salvas pelo tera_infer [FIX-3].

Referência: run_v29_ablacao_ttdef_ajuste_gatting.py
  episode_probs_from_frames (l.584), select_thr_ep (l.968), fail_rate (l.528).
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


# ── Constantes canônicas ─────────────────────────────────────────────────────
T_MAX_EF    = 10.0   # penalidade por episódio crítico não detectado (s)
TTD_M       = 3      # m = 3 consecutivos acima de theta_ttd
FAIL_BUDGET = 0.05   # restrição de FR na calibração (≠ thr_ep!)

# [FIX-2] K_AGG=6 — idêntico ao run_v29 linha 189.
K_AGG = 6

# [FIX-3] Grid de thresholds para select_thr_ep — idêntico ao run_v29 linha 216.
THR_EP_GRID = np.concatenate([
    np.linspace(0.05, 0.40, 15),
    np.linspace(0.40, 0.95, 12),
])


@dataclass
class EpisodeResult:
    episode_id:  int
    is_critical: bool
    is_progressive: bool
    detected:    bool
    ttd_frames:  Optional[int]
    ttd_seconds: Optional[float]


@dataclass
class ConfigResult:
    seed:        int
    config_id:   str
    model_name:  str
    gating:      bool
    f1:          float
    precision:   float
    recall:      float
    ece:         float
    fail_rate:   float
    ttd:         float
    ttdef:       float
    lat_ms:      float
    cost_ms_per_frame: float
    skip_pct:    float
    fail_rate_progressive: float
    fail_rate_abrupt:      float
    ttd_progressive:       float
    ttd_abrupt:            float
    ttdef_progressive:     float
    ttdef_abrupt:          float
    n_progressive:         int
    n_abrupt:              int


@dataclass
class AggregatedResult:
    config_id: str
    n_seeds:   int
    seeds:     List[int]
    metrics:   Dict[str, float] = field(default_factory=dict)
    stds:      Dict[str, float] = field(default_factory=dict)


# ── Funções de protocolo ──────────────────────────────────────────────────────

def binary_entropy(p: float) -> float:
    p = float(np.clip(p, 1e-7, 1 - 1e-7))
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


# [FIX-2] Idêntico ao run_v29 linha 584.
def episode_probs_from_frames(frame_probs: np.ndarray,
                               k: int = K_AGG) -> np.ndarray:
    """
    Probabilidade episódica = média dos últimos k frames (K_AGG=6).
    NÃO use probs.max(axis=1): tornaria F1≈1 e FR≈0 trivialmente.
    """
    return np.mean(frame_probs[:, -k:], axis=1)


def _fail_rate_vec(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """FR = FN/(TP+FN) — idêntico ao run_v29 linha 528."""
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        p = tp + fn
        return float(fn / p) if p > 0 else 0.0
    except Exception:
        pos = int((y_true == 1).sum())
        fn  = int(((y_true == 1) & (y_pred == 0)).sum())
        return float(fn / pos) if pos > 0 else 0.0


# [FIX-3] Idêntico ao run_v29 linha 968.
def select_thr_ep(frame_probs_val: np.ndarray,
                  y_true_ep_val: np.ndarray) -> float:
    """
    Calibra thr_ep no VAL: maximiza F1 com FR <= FAIL_BUDGET=0.05.

    NÃO confunda FAIL_BUDGET (restrição=0.05) com thr_ep (threshold resultante).
    thr_ep típico: 0.10–0.40 dependendo do modelo.
    """
    ep_probs = episode_probs_from_frames(frame_probs_val, k=K_AGG)
    feasible, fallback = [], []
    for thr in THR_EP_GRID:
        pred = (ep_probs >= float(thr)).astype(np.int64)
        fr   = _fail_rate_vec(y_true_ep_val, pred)
        f1v  = float(f1_score(y_true_ep_val, pred, zero_division=0))
        item = (fr, -f1v, abs(float(thr) - 0.30), float(thr))
        fallback.append(item)
        if fr <= FAIL_BUDGET:
            feasible.append((-f1v, fr, abs(float(thr) - 0.30), float(thr)))
    if feasible:
        feasible.sort()
        return float(feasible[0][3])
    fallback.sort()
    return float(fallback[0][3]) if fallback else 0.5


def first_stable_detection(probs: np.ndarray, onset: int,
                            thr: float, m: int = TTD_M) -> Optional[int]:
    if onset >= len(probs):
        return None
    count = 0
    for t in range(int(onset), len(probs)):
        if probs[t] >= thr:
            count += 1
            if count >= m:
                return t - m + 1
        else:
            count = 0
    return None


def first_stable_detection_full(probs: np.ndarray,
                                 thr: float, m: int = TTD_M) -> Optional[int]:
    """
    [v27-FIX] Busca a primeira detecção estável desde o frame 0 (não desde onset).
    Usada para calcular antecipação pré-onset em episódios progressivos.
    Idêntica ao run_v29 linha 692.
    """
    count = 0
    for t in range(len(probs)):
        if probs[t] >= thr:
            count += 1
            if count >= m:
                return t - m + 1
        else:
            count = 0
    return None


def compute_ttd_list(y_true_fr: np.ndarray, frame_probs: np.ndarray,
                     thr: float, window: int = 96, dt: float = 10.0/96,
                     m: int = TTD_M, frame_thr: float = 0.35) -> List[EpisodeResult]:
    results = []
    for ep_id, (yt, yp) in enumerate(zip(y_true_fr, frame_probs)):
        onset_frames = np.where(yt > frame_thr)[0]
        if len(onset_frames) == 0:
            continue
        onset = int(onset_frames[0])
        det   = first_stable_detection(yp, onset, thr=thr, m=m)
        if det is not None:
            detected   = True
            ttd_frames  = det - onset
            ttd_seconds = ttd_frames * dt
        else:
            detected   = False
            ttd_frames  = None
            ttd_seconds = None
        results.append(EpisodeResult(
            episode_id=ep_id, is_critical=True, is_progressive=False,
            detected=detected, ttd_frames=ttd_frames, ttd_seconds=ttd_seconds))
    return results


def compute_ttdef(ttd_det: float, fail_rate: float,
                  t_max: float = T_MAX_EF) -> float:
    return ttd_det + fail_rate * t_max


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray,
                                n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_pred >= lo) & (y_pred < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(float(y_true[mask].mean()) -
                                 float(y_pred[mask].mean()))
    return float(ece / max(len(y_true), 1))


def cliff_delta(group_a: List[float], group_b: List[float]) -> float:
    n_a, n_b = len(group_a), len(group_b)
    if n_a == 0 or n_b == 0:
        return float("nan")
    dom = sum((1 if a > b else (-1 if a < b else 0))
              for a in group_a for b in group_b)
    return dom / (n_a * n_b)


def wilcoxon_paired(a: List[float], b: List[float]) -> Tuple[float, float]:
    if len(a) < 3 or len(a) != len(b):
        return float("nan"), float("nan")
    try:
        stat, p = wilcoxon(a, b, alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


# ── Engine principal ──────────────────────────────────────────────────────────

class EvaluationEngine:

    def __init__(self, cfg: dict, exp_dir: Path):
        self.cfg       = cfg
        self.exp_dir   = exp_dir
        self.eval_cfg  = cfg.get("evaluation", {})
        self.m         = self.eval_cfg.get("m_detect", TTD_M)
        self.t_max     = self.eval_cfg.get("t_max_ef", T_MAX_EF)
        self.dt        = 10.0 / cfg.get("dataset", {}).get("window", 96)
        self.window    = cfg.get("dataset", {}).get("window", 96)
        self.frame_thr = cfg.get("dataset", {}).get("frame_thr", 0.35)

    def _load_inference_results(self, seed: int, config_id: str):
        """
        Carrega frame_probs, episode_probs, y_fr, y_ep, prog.
        [FIX-2] episode_probs carregados do cache ou recomputados via K_AGG=6.
        """
        base  = self.exp_dir / "metrics"
        probs = np.load(base / f"frame_probs_{config_id}_seed{seed}.npy")
        y_fr  = np.load(base / f"y_true_fr_seed{seed}.npy")
        y_ep  = np.load(base / f"y_episode_seed{seed}.npy")
        prog  = np.load(base / f"progressive_mask_seed{seed}.npy")

        ep_path = base / f"episode_probs_{config_id}_seed{seed}.npy"
        if ep_path.exists():
            ep_probs = np.load(ep_path)
        else:
            # Retrocompatibilidade: recomputa se tera_infer antigo não salvou
            print(f"    ⚠️  episode_probs não encontrado — recomputando (K_AGG={K_AGG})")
            ep_probs = episode_probs_from_frames(probs, K_AGG)

        return probs, ep_probs, y_fr, y_ep, prog

    def _load_skip_mask(self, seed: int, config_id: str) -> Optional[np.ndarray]:
        path = self.exp_dir / "metrics" / f"skip_mask_{config_id}_seed{seed}.npy"
        return np.load(path) if path.exists() else None

    def _compute_latency(self, skip_mask: Optional[np.ndarray],
                         role: str = "student", seed: Optional[int] = None):
        """
        [FIX-LAT] Carrega latência por role (baseline ou student) do JSON
        gerado pelo tera_infer.

        Prioridade de busca:
          1. latency_reference_seed{N}.json  — por seed (elimina std=0.000)
          2. latency_reference.json          — fallback genérico (retrocompat.)
          3. lat_raw = 0.41 ms              — fallback hardcoded

        Dentro do JSON, lê lat_ms_{role} (novo formato) ou lat_ms_per_frame
        (formato legado, sempre student).

        role: 'baseline' | 'student'
        seed: int | None — quando fornecido, tenta o arquivo por seed primeiro
        """
        base = self.exp_dir / "metrics"
        lat_ref = None

        # 1. Tenta JSON específico da seed
        if seed is not None:
            p = base / f"latency_reference_seed{seed}.json"
            if p.exists():
                lat_ref = json.loads(p.read_text())

        # 2. Fallback: JSON genérico
        if lat_ref is None:
            p = base / "latency_reference.json"
            if p.exists():
                lat_ref = json.loads(p.read_text())

        if lat_ref is not None:
            key_new = f"lat_ms_{role}"
            if key_new in lat_ref:
                lat_raw = float(lat_ref[key_new])
            else:
                lat_raw = float(lat_ref.get("lat_ms_per_frame", 0.41))
        else:
            lat_raw = 0.41

        skip_pct    = float(skip_mask.mean()) * 100 if skip_mask is not None and len(skip_mask) > 0 else 0.0
        active_frac = 1.0 - skip_pct / 100
        return lat_raw, lat_raw * active_frac, skip_pct

    # [FIX-3] Calibra thr_ep via select_thr_ep no VAL.
    def _calibrate_thr_ep(self, seed: int, role: str) -> float:
        """
        Calibra thr_ep para uma seed usando os VAL frame_probs salvos pelo
        InferenceManager [FIX-3 de tera_infer.py].

        role: 'baseline' | 'student'

        Se os arquivos VAL não existirem (pipeline legado sem FIX-3),
        exibe aviso e retorna 0.50 como fallback seguro.
        Execute --stages inference para regenerar com os fixes aplicados.
        """
        base       = self.exp_dir / "metrics"
        probs_path = base / f"frame_probs_val_{role}_seed{seed}.npy"
        y_ep_path  = base / f"y_ep_val_seed{seed}.npy"

        if not probs_path.exists() or not y_ep_path.exists():
            print(f"    ⚠️  [FIX-3] VAL probs ausentes (seed={seed}, role={role}).")
            print(f"         Execute --stages inference para gerar os arquivos VAL.")
            print(f"         Fallback: thr_ep=0.50.")
            return 0.50

        fp_val = np.load(probs_path)
        y_val  = np.load(y_ep_path)
        thr    = select_thr_ep(fp_val, y_val)
        print(f"    [FIX-3] thr_ep calibrado: {thr:.3f} (seed={seed}, role={role})")
        return thr

    def _evaluate_config_seed(self, seed: int, config_id: str) -> ConfigResult:
        probs, ep_probs, y_fr, y_ep, prog = self._load_inference_results(
            seed, config_id)

        config_cfg  = self.eval_cfg.get("configs", {}).get(config_id, {})
        is_baseline = config_cfg.get("model", "baseline") == "baseline"
        role        = "baseline" if is_baseline else "student"

        # [FIX-3] thr_ep calibrado via select_thr_ep — NÃO usa calibration_summary.csv
        thr_ep = self._calibrate_thr_ep(seed, role)

        # theta_ttd vem do calibration_summary (valor correto: 0.10)
        theta_ttd = 0.10
        cal_path  = self.exp_dir / "calibration" / "calibration_summary.csv"
        if cal_path.exists():
            df_cal = pd.read_csv(cal_path)
            row    = df_cal[df_cal["seed"] == seed]
            if not row.empty:
                theta_ttd = float(row.iloc[0].get("theta_ttd", 0.10))

        skip_mask = self._load_skip_mask(seed, config_id)

        # ── Classificação episódica ────────────────────────────────────────────
        # [FIX-2] ep_probs = mean(last K_AGG frames)
        ep_hat = (ep_probs >= thr_ep).astype(int)
        f1v    = float(f1_score(y_ep, ep_hat, zero_division=0))
        prec   = float(precision_score(y_ep, ep_hat, zero_division=0))
        rec    = float(recall_score(y_ep, ep_hat, zero_division=0))
        ece    = expected_calibration_error(y_ep.astype(float), ep_probs)

        # ── TTD e TTDef global ─────────────────────────────────────────────────
        crit_mask   = y_ep == 1
        ttd_results = compute_ttd_list(
            y_true_fr=y_fr[crit_mask], frame_probs=probs[crit_mask],
            thr=theta_ttd, window=self.window, dt=self.dt,
            m=self.m, frame_thr=self.frame_thr)
        n_crit     = int(crit_mask.sum())

        # [FIX-FR] FailRate via critério TTD — alinhado com o artigo em anexo.
        # O artigo usa FR = FN/(FN+TP) onde "detectado" = primeiro tdet tal que
        # p[t],p[t+1],p[t+2] ≥ θ_TTD após onset. Isso é o critério de detecção
        # estável, NÃO o K_AGG=6 (que mede confiança sustentada ao fim do episódio).
        # Com K_AGG=6: baseline FR=0.752 (decai após onset → último frames low)
        # Com TTD-critério: baseline FR≈0.225 (dispara no onset em 3 frames)
        detected_arr = np.array([r.detected for r in ttd_results], dtype=bool)
        n_detected   = int(detected_arr.sum())
        fail_rate    = (n_crit - n_detected) / n_crit if n_crit > 0 else 0.0

        ttd_det    = (float(np.mean([r.ttd_seconds for r in ttd_results
                                     if r.ttd_seconds is not None]))
                      if n_detected > 0 else 0.0)
        ttdef      = compute_ttdef(ttd_det, fail_rate, self.t_max)

        # ── Estratificado ──────────────────────────────────────────────────────
        prog_mask   = prog.astype(bool) & crit_mask
        abrupt_mask = (~prog.astype(bool)) & crit_mask

        # FR estratificado também via TTD-critério
        crit_prog_mask   = prog.astype(bool)[crit_mask]   # progressivos dentre críticos
        crit_abrupt_mask = ~crit_prog_mask

        fr_prog = (1.0 - detected_arr[crit_prog_mask].mean()
                   if crit_prog_mask.any() else 0.0)
        fr_abr  = (1.0 - detected_arr[crit_abrupt_mask].mean()
                   if crit_abrupt_mask.any() else 0.0)
        # TTD_Progressive: antecipação pré-onset (maior = melhor) — coluna da tabela
        ttd_prog = self._ttd_progressive_anticipation(
            y_fr[prog_mask], probs[prog_mask], theta_ttd)
        # TTDef_Progressive: atraso pós-onset + FR × T_ep (convenção do artigo)
        # Quando detectado antes do onset: delay_post=0 → TTDef_prog = FR_prog × T_ep
        # Com FR_prog=0 e detecção pré-onset: TTDef_prog = 0.000 ✓
        delay_post_prog = self._ttd_progressive_post_onset_delay(
            y_fr[prog_mask], probs[prog_mask], theta_ttd)
        ttd_abr  = self._ttd_subset_mean(
            y_fr[abrupt_mask], probs[abrupt_mask], theta_ttd)
        ttdef_prog = compute_ttdef(
            delay_post_prog, fr_prog if not math.isnan(fr_prog) else 0.0, self.t_max)
        ttdef_abr  = compute_ttdef(
            ttd_abr, fr_abr  if not math.isnan(fr_abr)  else 0.0, self.t_max)

        lat_raw, cost, skip_pct = self._compute_latency(skip_mask, role=role, seed=seed)
        model_name = "LSTM-AF-TKD" if not is_baseline else "LSTM-Baseline"

        return ConfigResult(
            seed=seed, config_id=config_id,
            model_name=model_name, gating=bool(config_cfg.get("gating", False)),
            f1=f1v, precision=prec, recall=rec, ece=ece,
            fail_rate=fail_rate, ttd=ttd_det, ttdef=ttdef,
            lat_ms=lat_raw, cost_ms_per_frame=cost, skip_pct=skip_pct,
            fail_rate_progressive=fr_prog, fail_rate_abrupt=fr_abr,
            ttd_progressive=ttd_prog, ttd_abrupt=ttd_abr,
            ttdef_progressive=ttdef_prog, ttdef_abrupt=ttdef_abr,
            n_progressive=int(prog_mask.sum()), n_abrupt=int(abrupt_mask.sum()),
        )

    def _fail_rate_subset(self, y_ep, ep_hat, mask):
        if mask.sum() == 0:
            return float("nan")
        y_sub   = y_ep[mask]
        hat_sub = ep_hat[mask]
        tp = int(((y_sub == 1) & (hat_sub == 1)).sum())
        fn = int(((y_sub == 1) & (hat_sub == 0)).sum())
        return fn / max(tp + fn, 1)

    def _ttd_progressive_anticipation(self, y_fr_prog, probs_prog, thr):
        """
        TTD_Progressive (coluna de antecipação, maior = melhor).

        Retorna os segundos de antecipação pré-onset: max(0, onset - t_det) × dt.
        Usa busca desde o frame 0 (first_stable_detection_full), idêntico ao
        run_v29 compute_anticipation (linha 711).

        Este valor é reportado como TTD_Progressive no CSV e na Tabela 4 do artigo.
        NÃO é usado diretamente no TTDef_Progressive — ver
        _ttd_progressive_post_onset_delay() para o componente do TTDef.
        """
        anticipations = []
        for yt, yp in zip(y_fr_prog, probs_prog):
            onset_frames = np.where(yt > self.frame_thr)[0]
            if len(onset_frames) == 0:
                continue
            onset = int(onset_frames[0])
            det = first_stable_detection_full(yp, thr=thr, m=self.m)
            if det is None:
                anticipations.append(0.0)   # não detectou = zero antecipação
            else:
                # Positivo quando detectou antes do onset
                anticipations.append(max(0.0, (onset - det) * self.dt))
        return float(np.mean(anticipations)) if anticipations else 0.0

    def _ttd_progressive_post_onset_delay(self, y_fr_prog, probs_prog, thr):
        """
        Componente de atraso pós-onset para TTDef_Progressive.

        Convenção do artigo:
          TTDef_Progressive = max(0, t_det - onset) × dt + FR_prog × T_ep

          Quando a detecção ocorre ANTES do onset (antecipação real),
          o atraso pós-onset é zero → TTDef_prog = FR_prog × T_ep.
          Com FR_prog=0, TTDef_prog = 0.000 — sem penalidade operacional.

        Isso distingue TTD_Progressive (antecipação, maior=melhor) de
        TTDef_Progressive (penalidade, menor=melhor). A tabela do artigo
        mostra TTDef_prog=0.000 para AF-TKD com FR_prog=0, refletindo
        este cálculo, não a antecipação em si.
        """
        delays = []
        for yt, yp in zip(y_fr_prog, probs_prog):
            onset_frames = np.where(yt > self.frame_thr)[0]
            if len(onset_frames) == 0:
                continue
            onset = int(onset_frames[0])
            det = first_stable_detection_full(yp, thr=thr, m=self.m)
            # max(0, delay pós-onset) — zero se detectado antes do onset
            delay = max(0.0, (det - onset) * self.dt) if det is not None else 0.0
            delays.append(delay)
        return float(np.mean(delays)) if delays else 0.0

    def _ttd_subset_mean(self, y_fr, probs, thr):
        ttds = []
        for yt, yp in zip(y_fr, probs):
            onset_frames = np.where(yt > self.frame_thr)[0]
            if len(onset_frames) == 0:
                continue
            onset = int(onset_frames[0])
            det   = first_stable_detection(yp, onset, thr=thr, m=self.m)
            if det is not None:
                ttds.append((det - onset) * self.dt)
        return float(np.mean(ttds)) if ttds else 0.0

    def run(self, seeds: List[int], configs: List[str]) -> pd.DataFrame:
        rows = []
        for seed in seeds:
            for config_id in configs:
                print(f"  Avaliando seed={seed}, config={config_id}...")
                try:
                    result = self._evaluate_config_seed(seed, config_id)
                    rows.append(self._result_to_dict(result))
                except Exception as e:
                    print(f"    ⚠️ Erro: {e}")
        df  = pd.DataFrame(rows)
        out = self.exp_dir / "metrics" / "results_all_seeds.csv"
        df.to_csv(out, index=False)
        print(f"\n  ✓ results_all_seeds.csv salvo: {out}")
        return df

    def _result_to_dict(self, r: ConfigResult) -> dict:
        def _safe(v):
            return round(v, 4) if not (isinstance(v, float) and math.isnan(v)) else None
        return {
            "Seed": r.seed, "config_id": r.config_id,
            "Modelo": r.model_name, "Gating": r.gating,
            "F1": round(r.f1, 4), "Precision": round(r.precision, 4),
            "Recall": round(r.recall, 4), "ECE": round(r.ece, 4),
            "FailRate": round(r.fail_rate, 4),
            "TTD": round(r.ttd, 4), "TTDef": round(r.ttdef, 4),
            "Lat_ms": round(r.lat_ms, 4),
            "Cost_ms_per_frame": round(r.cost_ms_per_frame, 4),
            "SkipPct": round(r.skip_pct, 3),
            "FailRate_Progressive": _safe(r.fail_rate_progressive),
            "FailRate_Abrupt":      _safe(r.fail_rate_abrupt),
            "TTD_Progressive":  round(r.ttd_progressive, 4),
            "TTD_Abrupt":       round(r.ttd_abrupt, 4),
            "TTDef_Progressive": round(r.ttdef_progressive, 4),
            "TTDef_Abrupt":      round(r.ttdef_abrupt, 4),
            "N_Progressive": r.n_progressive,
            "N_Abrupt":      r.n_abrupt,
        }

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "config_id" not in df.columns:
            return pd.DataFrame()
        num_cols = [c for c in df.columns
                    if c not in ("Seed", "config_id", "Modelo", "Gating")]
        agg = df.groupby("config_id")[num_cols].agg(["mean", "std"]).round(4)
        agg.columns = ["_".join(c) for c in agg.columns]
        n_seeds = df.groupby("config_id")["Seed"].nunique().rename("N_Seeds")
        return agg.join(n_seeds).reset_index()

    def print_summary(self, df: pd.DataFrame) -> None:
        if df.empty or "config_id" not in df.columns:
            print("\n  ⚠️  Nenhum resultado para resumir.")
            return
        agg = self.aggregate(df)
        if agg.empty:
            return
        print("\n" + "=" * 72)
        print("  RESULTADOS AGREGADOS (média ± desvio, 4 seeds)")
        print("=" * 72)
        for _, row in agg.iterrows():
            config = row["config_id"]
            n      = int(row.get("N_Seeds", 0))
            f1v    = row.get("F1_mean",               float("nan"))
            fr     = row.get("FailRate_mean",          float("nan"))
            ttdef  = row.get("TTDef_mean",             float("nan"))
            cost   = row.get("Cost_ms_per_frame_mean", float("nan"))
            skip   = row.get("SkipPct_mean",           float("nan"))
            status = "✓" if n == 4 else f"⚠️ {n}/4 seeds"
            print(f"  {config:20s} n={n} {status}")
            print(f"    F1={f1v:.3f}  FR={fr:.3f}  TTDef={ttdef:.3f}s"
                  f"  Cost={cost:.3f}ms/q  Skip={skip:.1f}%")

    def statistical_comparison(self, df: pd.DataFrame,
                                config_a: str = "baseline_fixed",
                                config_b: str = "aftkd_fixed") -> pd.DataFrame:
        seeds   = sorted(df["Seed"].unique())
        rows    = []
        metrics = ["F1", "ECE", "FailRate", "TTDef", "Cost_ms_per_frame"]
        for metric in metrics:
            a_vals = [float(df[(df["config_id"] == config_a) &
                               (df["Seed"] == s)][metric].iloc[0])
                      for s in seeds
                      if len(df[(df["config_id"] == config_a) &
                                (df["Seed"] == s)]) > 0]
            b_vals = [float(df[(df["config_id"] == config_b) &
                               (df["Seed"] == s)][metric].iloc[0])
                      for s in seeds
                      if len(df[(df["config_id"] == config_b) &
                                (df["Seed"] == s)]) > 0]
            if len(a_vals) < 2:
                continue
            delta   = cliff_delta(a_vals, b_vals)
            stat, p = wilcoxon_paired(a_vals, b_vals)
            magnitude = ("Grande" if abs(delta) >= 0.474
                         else "Médio" if abs(delta) >= 0.33
                         else "Pequeno")
            rows.append({
                "Métrica":    metric,
                "delta_Cliff": round(delta, 3),
                "Magnitude":   magnitude,
                "p_Wilcoxon":  round(p, 4) if not math.isnan(p) else "n/a",
                f"Média_{config_a}": round(float(np.mean(a_vals)), 4),
                f"Média_{config_b}": round(float(np.mean(b_vals)), 4),
            })
        return pd.DataFrame(rows)


# ── Análise borderline ────────────────────────────────────────────────────────

def analyze_borderline(borderline_logs_path: Path, config: str = "aftkd_fixed",
                       theta: float = 0.10, m_stable: int = 3) -> pd.DataFrame:
    if not borderline_logs_path.exists():
        return pd.DataFrame()
    df     = pd.read_csv(borderline_logs_path)
    df_cfg = df[df["config"] == config]
    rows   = []
    for level in ["Normal", "Atenção", "Alerta", "Crítico"]:
        sub_ep = (df_cfg[df_cfg["risk_level"] == level]
                  .groupby("episode_id")["p_t"].max()
                  .reset_index(name="p_max"))
        if sub_ep.empty:
            continue
        rows.append({
            "risk_level":       level,
            "p_max_mean":       round(float(sub_ep["p_max"].mean()), 3),
            "p_max_std":        round(float(sub_ep["p_max"].std()),  3),
            "frac_above_theta": round(float((sub_ep["p_max"] >= theta).mean() * 100), 1),
            "n_episodes":       len(sub_ep),
        })
    return pd.DataFrame(rows)
