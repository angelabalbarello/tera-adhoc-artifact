# -*- coding: utf-8 -*-
"""
tera_pipeline/calibration/tera_calibrate.py
Calibração de limiares e parâmetros de gating do TERA Pipeline.

Calibra por seed (no conjunto de validação):
  · thr_ep       — threshold episódico (maximiza F1 no VAL)
  · theta_ttd    — threshold de detecção TTD (minimiza TTDef no VAL)
  · tau_delta    — limiar cinemático do MHEG
  · tau_H        — limiar entrópico do MHEG

Critério de seleção do gating:
  Maximiza SkipPct sob restrições:
    · FR_val ≤ fail_budget (5%)
    · TTDef_val ≤ 120% do TTDef da política fixa

Observação arquitetural importante:
  Os mesmos (tau_delta, tau_H) calibrados para o AF-TKD são usados
  para a Baseline Hybrid. Isso isola o efeito da supervisão temporal
  AF-TKD, não do gating em si. É uma escolha metodológica deliberada.

SAÍDAS em exp_dir/calibration/:
  calibration_summary.csv  — um row por seed com todos os parâmetros
  calibration_detail.csv   — grade completa para inspeção
"""

import math
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tera_pipeline.inference.tera_infer import (
    infer_stream_fixed,
    infer_stream_hybrid,
    StreamEval,
)
from tera_pipeline.evaluation.tera_eval import (
    first_stable_detection,
    binary_entropy,
    compute_ttdef,
    T_MAX_EF,
    TTD_M,
)


class CalibrationManager:
    """Gerencia calibração de todos os parâmetros por seed."""

    def __init__(self, cfg: dict, exp_dir: Path, seeds: List[int],
                 device: torch.device):
        self.cfg     = cfg
        self.exp_dir = exp_dir
        self.seeds   = seeds
        self.device  = device
        self.cal_cfg = cfg.get("calibration", {})
        self.model_cfg = cfg.get("model", {})
        self.out_dir = exp_dir / "calibration"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.window    = cfg["dataset"]["window"]
        self.frame_thr = cfg["dataset"]["frame_thr"]
        self.dt        = 10.0 / self.window
        self.m         = cfg.get("evaluation", {}).get("m_detect", TTD_M)
        self.t_max     = cfg.get("evaluation", {}).get("t_max_ef", T_MAX_EF)
        self.kin       = self.model_cfg.get("kinematic_indices", [12, 13, 14])
        self.fail_budget = self.cal_cfg.get("fail_rate_budget", 0.05)
        self.ttdef_degrad = self.cal_cfg.get("ttdef_degradation", 0.20)

    # Carregamento

    def _load_model(self, role: str, seed: int) -> nn.Module:
        from tera_pipeline.training.tera_train import MultiTaskLSTM
        D = self.cfg["dataset"]["input_dim"]
        H = self.model_cfg["hidden_dim"]
        # Busca em exp_dir/models/ ou legacy
        candidates = [
            self.exp_dir / "models" / f"{role}_seed{seed}.pt",
            Path("modelos_salvos") / "abl_A" / f"{role}_seed{seed}.pt",
        ]
        for ckpt in candidates:
            if ckpt.exists():
                m = MultiTaskLSTM(D, H, bi=False).to(self.device)
                m.load_state_dict(torch.load(ckpt, map_location=self.device))
                m.eval()
                return m
        raise FileNotFoundError(f"Modelo '{role}_seed{seed}.pt' não encontrado.")

    def _load_val(self, seed: int) -> Tuple[np.ndarray, ...]:
        d = self.exp_dir / "data"
        X   = np.load(d / f"X_va_seed{seed}.npy")
        y_fr = np.load(d / f"y_fr_va_seed{seed}.npy")
        y_ep = np.load(d / f"y_ep_va_seed{seed}.npy")
        prog = np.load(d / f"progressive_va_seed{seed}.npy")
        return X, y_fr, y_ep, prog

    # Calibração de thr_ep

    def calibrate_thr_ep(
        self, frame_probs: np.ndarray, y_ep: np.ndarray
    ) -> float:
        """Maximiza F1 episódico no VAL."""
        from sklearn.metrics import f1_score
        ep_probs = frame_probs.max(axis=1)
        best_thr, best_f1 = 0.5, 0.0
        grid_low  = np.linspace(*self.cal_cfg.get("thr_ep_grid_low",  [0.05, 0.40, 15]))
        grid_high = np.linspace(*self.cal_cfg.get("thr_ep_grid_high", [0.40, 0.95, 12]))
        for thr in np.concatenate([grid_low, grid_high]):
            f1 = f1_score(y_ep, (ep_probs >= thr).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        return best_thr

    # Calibração de theta_ttd

    def calibrate_theta_ttd(
        self, y_fr: np.ndarray, frame_probs: np.ndarray
    ) -> float:
        """Seleciona theta que minimiza TTDef no VAL sobre críticos."""
        theta_grid = self.cal_cfg.get("theta_ttd_grid",
                                       [0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                                        0.40, 0.45, 0.50])
        best_theta, best_ttdef = 0.10, math.inf
        for theta in theta_grid:
            ttdef = self._compute_ttdef_val(y_fr, frame_probs, theta)
            if ttdef < best_ttdef:
                best_ttdef, best_theta = ttdef, float(theta)
        return best_theta

    def _compute_ttdef_val(
        self, y_fr: np.ndarray, frame_probs: np.ndarray, theta: float
    ) -> float:
        """Calcula TTDef médio no VAL para um dado theta."""
        ttds, n_fail = [], 0
        for yt, yp in zip(y_fr, frame_probs):
            onset_idx = np.where(yt > self.frame_thr)[0]
            if len(onset_idx) == 0:
                continue
            onset = int(onset_idx[0])
            det = first_stable_detection(yp, onset, thr=theta, m=self.m)
            if det is not None:
                ttds.append((det - onset) * self.dt)
            else:
                n_fail += 1
        total = len(ttds) + n_fail
        if total == 0:
            return math.inf
        fr = n_fail / total
        ttd_mean = float(np.mean(ttds)) if ttds else 0.0
        return compute_ttdef(ttd_mean, fr, self.t_max)

    # Calibração de gating (tau_delta, tau_H)

    def calibrate_gating(
        self,
        student:     nn.Module,
        X_va:        np.ndarray,
        y_fr_va:     np.ndarray,
        y_ep_va:     np.ndarray,
        theta_ttd:   float,
        thr_ep:      float,
        ref_ttdef:   float,
    ) -> Tuple[float, float, pd.DataFrame]:
        """
        Busca em grade (tau_delta, tau_H) usando simulação analítica do gating.

        ESTRATÉGIA RÁPIDA:
          1. Roda inferência fixa UMA ÚNICA VEZ -> cacheia frame_probs
          2. Pré-computa deltas cinemáticos e entropy por frame
          3. Para cada ponto da grade, simula o gating em numpy puro
             (sem LSTM) — propagação de prob anterior nos frames suprimidos
          4. Computa FR e TTDef sobre as probs simuladas

        Reduz de ~9M forward passes para 1 inferência + operações numpy.
        Retorna (tau_delta_best, tau_H_best, grade_df).
        """
        # Passo 1: Inferência fixa UMA VEZ
        print(f"      Inferência fixa (cache)...", end=" ", flush=True)
        result_fixed = infer_stream_fixed(student, X_va, self.device)
        probs_fixed  = result_fixed.frame_probs       # (N, T)
        print("OK")

        # Passo 2: Deltas cinemáticos (N, T)
        print(f"      Pré-computando deltas...", end=" ", flush=True)
        N, T = probs_fixed.shape
        deltas = np.zeros((N, T), dtype=np.float32)
        for ep in range(N):
            for t in range(1, T):
                deltas[ep, t] = float(
                    np.abs(X_va[ep, t, self.kin] -
                           X_va[ep, t-1, self.kin]).max()
                )
        print("OK")

        # Passo 3: Entropy auxiliar (vectorizada)
        def _h(p: float) -> float:
            p = max(1e-7, min(1 - 1e-7, p))
            return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

        # Passo 4: Grade
        delta_flat = deltas[:, 1:].ravel()
        pctls = self.cal_cfg.get("tau_delta_percentiles",
                                  list(range(50, 95, 2)))
        tau_delta_grid = sorted(set(
            float(np.percentile(delta_flat, p)) for p in pctls
        ))
        tau_h_grid = self.cal_cfg.get("tau_h_grid",
                                       [0.10, 0.20, 0.30, 0.50, 0.70, 0.85, 0.95])

        ttdef_budget = ref_ttdef * (1 + self.ttdef_degrad)
        best_skip, best_td, best_th = -1.0, tau_delta_grid[0], tau_h_grid[-1]
        total = len(tau_delta_grid) * len(tau_h_grid)
        print(f"      Grade {len(tau_delta_grid)}×{len(tau_h_grid)}"
              f"={total} pts (numpy)...", end=" ", flush=True)

        rows = []
        for tau_d in tau_delta_grid:
            for tau_h in tau_h_grid:
                # Simula gating frame-a-frame sem LSTM
                hybrid = np.zeros_like(probs_fixed)
                skip_total = 0
                for ep in range(N):
                    y_prev = 0.0
                    for t in range(T):
                        activate = (deltas[ep, t] >= tau_d) or \
                                   (_h(y_prev) >= tau_h)
                        y_curr = float(probs_fixed[ep, t]) if activate \
                                 else y_prev
                        if not activate:
                            skip_total += 1
                        hybrid[ep, t] = y_curr
                        y_prev = y_curr

                skip_pct  = skip_total / (N * T) * 100
                ep_hat    = (hybrid.max(axis=1) >= thr_ep).astype(int)
                fr_val    = self._compute_fr(y_ep_va, ep_hat)
                ttdef_val = self._compute_ttdef_val(y_fr_va, hybrid, theta_ttd)
                feasible  = (fr_val <= self.fail_budget) and \
                            (ttdef_val <= ttdef_budget)

                rows.append({
                    "tau_delta": round(tau_d, 6),
                    "tau_h":     round(tau_h, 3),
                    "fr_val":    round(fr_val,    4),
                    "ttdef_val": round(ttdef_val, 4),
                    "skip_pct":  round(skip_pct,  2),
                    "feasible":  feasible,
                })
                if feasible and skip_pct > best_skip:
                    best_skip, best_td, best_th = skip_pct, tau_d, tau_h

        print("OK")
        return best_td, best_th, pd.DataFrame(rows)

    def _compute_kinematic_deltas(self, X: np.ndarray) -> np.ndarray:
        """Calcula todas as variações cinemáticas |Δx_kin| no conjunto."""
        deltas = []
        for ep in X:
            for t in range(1, ep.shape[0]):
                d = float(np.abs(ep[t, self.kin] - ep[t-1, self.kin]).max())
                deltas.append(d)
        return np.array(deltas, dtype=np.float32)

    def _compute_fr(self, y_ep: np.ndarray, ep_hat: np.ndarray) -> float:
        crit = y_ep == 1
        if crit.sum() == 0:
            return 0.0
        fn = ((y_ep == 1) & (ep_hat == 0)).sum()
        tp = ((y_ep == 1) & (ep_hat == 1)).sum()
        return float(fn / max(tp + fn, 1))

    # Orquestrador

    def run(self) -> None:
        """Calibra todos os parâmetros para todas as seeds."""
        all_rows  = []
        all_grade = []

        for seed in self.seeds:
            print(f"\n  [Calibration] seed={seed}")
            from tera_pipeline.utils.tera_utils import set_global_seed
            set_global_seed(seed)

            student  = self._load_model("student",  seed)
            baseline = self._load_model("baseline", seed)
            X_va, y_fr_va, y_ep_va, _ = self._load_val(seed)

            # 1. Calibra thr_ep (student)
            probs_fixed = infer_stream_fixed(student, X_va, self.device).frame_probs
            thr_ep = self.calibrate_thr_ep(probs_fixed, y_ep_va)

            # 2. Calibra theta_ttd (student)
            theta_ttd = self.calibrate_theta_ttd(y_fr_va, probs_fixed)

            # 3. TTDef de referência (política fixa) para budget do gating
            ref_ttdef = self._compute_ttdef_val(y_fr_va, probs_fixed, theta_ttd)

            # 4. Calibra (tau_delta, tau_H) no VAL do student
            print(f"    Calibrando gating (grade)...")
            tau_d, tau_h, grade_df = self.calibrate_gating(
                student=student, X_va=X_va, y_fr_va=y_fr_va,
                y_ep_va=y_ep_va, theta_ttd=theta_ttd,
                thr_ep=thr_ep, ref_ttdef=ref_ttdef,
            )

            # 5. thr_base (para Baseline)
            probs_base = infer_stream_fixed(baseline, X_va, self.device).frame_probs
            thr_base   = self.calibrate_thr_ep(probs_base, y_ep_va)

            feasible_n = int(grade_df["feasible"].sum())
            print(f"    thr_ep={thr_ep:.3f}  theta_ttd={theta_ttd:.3f}  "
                  f"τΔ={tau_d:.4f}  τH={tau_h:.2f}  "
                  f"feasible={feasible_n}/{len(grade_df)}")

            all_rows.append({
                "seed":      seed,
                "thr_ep":    round(thr_ep, 4),
                "thr_base":  round(thr_base, 4),
                "theta_ttd": round(theta_ttd, 4),
                "tau_delta": round(tau_d, 6),
                "tau_h":     round(tau_h, 3),
                "ref_ttdef": round(ref_ttdef, 4),
                "n_feasible": feasible_n,
            })
            grade_df["seed"] = seed
            all_grade.append(grade_df)

        # Salva summary
        summary_df = pd.DataFrame(all_rows)
        out_s = self.out_dir / "calibration_summary.csv"
        summary_df.to_csv(out_s, index=False)
        print(f"\n  calibration_summary.csv -> {out_s}")

        # Salva grade completa
        if all_grade:
            grade_all = pd.concat(all_grade, ignore_index=True)
            out_g = self.out_dir / "calibration_detail.csv"
            grade_all.to_csv(out_g, index=False)
            print(f"  calibration_detail.csv -> {out_g}")

        # Também salva no exp_dir/metrics/ para o inference manager
        metrics_dir = self.exp_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        summary_df.to_csv(metrics_dir / "calibration_summary.csv", index=False)
