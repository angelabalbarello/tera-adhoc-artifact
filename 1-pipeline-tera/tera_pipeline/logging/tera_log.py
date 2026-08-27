# -*- coding: utf-8 -*-
"""
tera_pipeline/logging/tera_log.py
═══════════════════════════════════════════════════════════════════════════════
Geração de frame logs para as figuras do artigo.

REGRA CANÔNICA:
  Frame logs são usados APENAS para figuras ilustrativas.
  Nunca para métricas quantitativas do paper.

  SEED_VIZ = 42 (configurável) — única seed para frame logs.
  Métricas do paper vêm de results_all_seeds.csv (4 seeds).

SAÍDAS em exp_dir/logs/:
  episode_frame_logs.csv    — p(t), H(t), supressões, por frame
  borderline_frame_logs.csv — distribuição de risco em episódios borderline
═══════════════════════════════════════════════════════════════════════════════
"""

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PRE_ONSET_WINDOW = 10   # frames antes de t0 para fase "pre_onset"


def binary_entropy(p: float) -> float:
    p = float(np.clip(p, 1e-7, 1 - 1e-7))
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


class FrameLogger:
    """
    Gera CSVs de frame logs para figuras do artigo.
    Usa apenas seed_viz (default 42).
    """

    CONFIGS = ["baseline_fixed", "baseline_hybrid", "aftkd_fixed", "aftkd_hybrid"]

    def __init__(self, cfg: dict, exp_dir: Path, device: torch.device):
        self.cfg      = cfg
        self.exp_dir  = exp_dir
        self.device   = device
        self.log_cfg  = cfg.get("logging", {})
        self.model_cfg = cfg.get("model", {})
        self.window   = cfg["dataset"]["window"]
        self.frame_thr = cfg["dataset"]["frame_thr"]
        self.dt       = 10.0 / self.window
        self.kin      = self.model_cfg.get("kinematic_indices", [12, 13, 14])
        self.out_dir  = exp_dir / "logs"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self, role: str, seed: int, bi: bool = False) -> nn.Module:
        from tera_pipeline.training.tera_train import MultiTaskLSTM
        D = self.cfg["dataset"]["input_dim"]
        H = self.model_cfg["hidden_dim"]
        candidates = [
            self.exp_dir / "models" / f"{role}_seed{seed}.pt",
            Path("modelos_salvos") / "abl_A" / f"{role}_seed{seed}.pt",
        ]
        for ckpt in candidates:
            if ckpt.exists():
                m = MultiTaskLSTM(D, H, bi=bi).to(self.device)
                m.load_state_dict(torch.load(ckpt, map_location=self.device))
                m.eval()
                return m
        raise FileNotFoundError(f"Modelo '{role}_seed{seed}.pt' não encontrado.")

    def _load_data(self, seed: int, split: str = "te"):
        d = self.exp_dir / "data"
        X    = np.load(d / f"X_{split}_seed{seed}.npy")
        y_fr = np.load(d / f"y_fr_{split}_seed{seed}.npy")
        y_ep = np.load(d / f"y_ep_{split}_seed{seed}.npy")
        prog = np.load(d / f"progressive_{split}_seed{seed}.npy")
        return X, y_fr, y_ep, prog

    def _load_calibration(self, seed: int) -> dict:
        for p in [self.exp_dir / "calibration" / "calibration_summary.csv",
                  self.exp_dir / "metrics" / "calibration_summary.csv"]:
            if p.exists():
                df = pd.read_csv(p)
                row = df[df["seed"] == seed]
                return row.iloc[0].to_dict() if not row.empty else {}
        return {}

    def _get_tau(self, seed: int) -> tuple:
        """Retorna (tau_delta, tau_h) da calibração."""
        cal = self._load_calibration(seed)
        return float(cal.get("tau_delta", 0.02)), float(cal.get("tau_h", 0.95))

    def _frame_phase(self, t: int, t0: int) -> str:
        if t0 < 0:
            return "normal"
        if t >= t0:
            return "post_onset"
        if t >= t0 - PRE_ONSET_WINDOW:
            return "pre_onset"
        return "stable"

    # ── Episode frame logs ────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_episode_logs(self, seeds: List[int]) -> None:
        """
        Gera episode_frame_logs.csv com p(t), H(t), supressão, fase temporal.
        Uma linha por (seed, config, episode_id, frame).
        """
        out_path = self.out_dir / "episode_frame_logs.csv"

        all_rows = []
        for seed in seeds:
            print(f"    [FrameLog] seed={seed}")
            X, y_fr, y_ep, prog = self._load_data(seed)
            tau_d, tau_h = self._get_tau(seed)

            baseline = self._load_model("baseline", seed)
            student  = self._load_model("student",  seed)

            models = {
                "baseline_fixed":   (baseline, False),
                "baseline_hybrid":  (baseline, True),
                "aftkd_fixed":      (student,  False),
                "aftkd_hybrid":     (student,  True),
            }
            for config_id, (model, use_gating) in models.items():
                rows = self._log_config(
                    seed=seed, config_id=config_id,
                    model=model, use_gating=use_gating,
                    X=X, y_fr=y_fr, y_ep=y_ep, prog=prog,
                    tau_d=tau_d, tau_h=tau_h,
                )
                all_rows.extend(rows)

        df = pd.DataFrame(all_rows)
        df.to_csv(out_path, index=False)
        print(f"    ✓ episode_frame_logs.csv → {out_path} ({len(df):,} linhas)")

    @torch.no_grad()
    def _log_config(
        self, seed, config_id, model, use_gating,
        X, y_fr, y_ep, prog, tau_d, tau_h,
    ) -> List[dict]:
        rows = []
        T = self.window
        for ep_id, (x_ep, yt, crit, is_prog) in enumerate(
                zip(X, y_fr, y_ep, prog)):
            # Frame-a-frame com ou sem gating
            probs  = np.zeros(T, dtype=np.float32)
            suppressed = np.zeros(T, dtype=bool)
            y_prev = 0.0

            for t in range(T):
                if use_gating:
                    delta = float(np.abs(
                        x_ep[t, self.kin] - x_ep[t-1, self.kin]
                    ).max()) if t > 0 else 0.0
                    H_prev = binary_entropy(y_prev)
                    activate = (delta >= tau_d) or (H_prev >= tau_h)
                else:
                    activate = True

                if activate:
                    xb = torch.tensor(
                        x_ep[t:t+1].reshape(1, 1, -1),
                        dtype=torch.float32, device=self.device
                    )
                    out, _ = model(xb)
                    y_curr = float(torch.sigmoid(out[0, -1]).cpu())
                else:
                    y_curr = y_prev
                    suppressed[t] = True

                probs[t] = y_curr
                y_prev   = y_curr

            # Onset frame
            onset_idx = np.where(yt > self.frame_thr)[0]
            t0_frame  = int(onset_idx[0]) if (int(crit) == 1 and len(onset_idx) > 0) else -1
            t0_time   = t0_frame * self.dt if t0_frame >= 0 else -1.0

            # Uma linha por frame
            for t in range(T):
                rows.append({
                    "seed":         seed,
                    "config":       config_id,
                    "episode_id":   ep_id,
                    "regime":       "progressive" if bool(is_prog) else
                                   ("abrupt" if int(crit) == 1 else "normal"),
                    "is_critical":  int(crit),
                    "t0_frame":     t0_frame,
                    "t0_time_s":    round(t0_time, 4),
                    "t":            t,
                    "time_s":       round(t * self.dt, 4),
                    "p_t":          round(float(probs[t]), 5),
                    "H_t":          round(binary_entropy(float(probs[t])), 5),
                    "is_suppressed": int(suppressed[t]),
                    "y_frame_clean": round(float(yt[t]), 4),
                    "phase":        self._frame_phase(t, t0_frame),
                })
        return rows

    # ── Borderline frame logs ─────────────────────────────────────────────────

    @torch.no_grad()
    def generate_borderline_logs(self, seeds: List[int]) -> None:
        """
        Gera borderline_frame_logs.csv com distribuição de p_t por nível de risco.
        Usado para fig:prob_dist (Normal / Atenção / Alerta / Crítico).
        """
        out_path = self.out_dir / "borderline_frame_logs.csv"

        # Tenta importar o gerador de borderline do módulo legado
        try:
            from generate_borderline_frame_logs import BORDERLINE_RECIPES
        except ImportError:
            BORDERLINE_RECIPES = None

        all_rows = []
        for seed in seeds:
            student  = self._load_model("student",  seed)
            baseline = self._load_model("baseline", seed)

            rows = self._log_borderline_seed(seed, student, baseline,
                                              BORDERLINE_RECIPES)
            all_rows.extend(rows)

        if all_rows:
            df = pd.DataFrame(all_rows)
            df.to_csv(out_path, index=False)
            print(f"    ✓ borderline_frame_logs.csv → {out_path} ({len(df):,} linhas)")
        else:
            print("    ⚠️  Nenhum episódio borderline gerado.")

    def _log_borderline_seed(
        self, seed: int, student: nn.Module, baseline: nn.Module,
        recipes
    ) -> List[dict]:
        """
        Gera episódios borderline e loga p(t) para baseline e student.
        Se o módulo borderline não estiver disponível, usa o test set.
        """
        rows = []
        X, y_fr, y_ep, _ = self._load_data(seed, split="te")

        risk_levels = {
            0: "Normal",
            1: "Crítico",
        }
        for ep_id, (x_ep, yt, crit) in enumerate(zip(X, y_fr, y_ep)):
            level = risk_levels.get(int(crit), "Crítico")
            for config_id, model in [("baseline_fixed", baseline),
                                       ("aftkd_fixed",   student)]:
                probs = self._infer_episode(model, x_ep)
                p_max = float(probs.max())
                for t in range(len(probs)):
                    rows.append({
                        "seed":        seed,
                        "config":      config_id,
                        "episode_id":  ep_id,
                        "risk_level":  level,
                        "t":           t,
                        "time_s":      round(t * self.dt, 4),
                        "p_t":         round(float(probs[t]), 5),
                        "y_frame_clean": round(float(yt[t]), 4),
                        "p_max_episode": round(p_max, 5),
                    })
        return rows

    @torch.no_grad()
    def _infer_episode(self, model: nn.Module, x_ep: np.ndarray) -> np.ndarray:
        """Inferência causal frame-a-frame para um episódio."""
        T = x_ep.shape[0]
        probs = np.zeros(T, dtype=np.float32)
        for t in range(T):
            xb = torch.tensor(
                x_ep[t:t+1].reshape(1, 1, -1),
                dtype=torch.float32, device=self.device
            )
            out, _ = model(xb)
            probs[t] = float(torch.sigmoid(out[0, -1]).cpu())
        return probs
