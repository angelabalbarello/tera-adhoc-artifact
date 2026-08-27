# -*- coding: utf-8 -*-
"""
tera_pipeline/figures/tera_figures.py
═══════════════════════════════════════════════════════════════════════════════
Geração automática de todas as figuras do artigo.

Figuras geradas:
  fig_temporal_trace.pdf    — p(t), H(t), supressões, onset (seed_viz)
  fig_entropy_dist.pdf      — distribuição de H(t) por fase (seed_viz)
  fig_gating_heatmap.pdf    — mapa de ativação MHEG (seed_viz)
  fig_tradeoff.pdf          — FR vs Cost por config (4 seeds, CI 95%)
  fig_multiseed.pdf         — métricas por seed com IC
  fig_prob_dist.pdf         — distribuição p_max por nível de risco

FONTE:
  · fig_tradeoff e fig_multiseed: results_all_seeds.csv (4 seeds)
  · Demais figuras: episode_frame_logs.csv (seed_viz=42 apenas)

REGRA: Figuras de trade-off usam SEMPRE os resultados de 4 seeds.
       Figuras ilustrativas (trace, entropy, heatmap) usam seed_viz.
═══════════════════════════════════════════════════════════════════════════════
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ── Cores canônicas (alinhadas com o artigo) ──────────────────────────────────
COLORS = {
    "baseline_fixed":   "#555555",
    "baseline_hybrid":  "#E07B39",
    "aftkd_fixed":      "#3B82F6",
    "aftkd_hybrid":     "#16A34A",
}
LABELS = {
    "baseline_fixed":   "Baseline Fixo",
    "baseline_hybrid":  "Baseline+Gating",
    "aftkd_fixed":      "AF-TKD Fixo",
    "aftkd_hybrid":     "AF-TKD+Gating",
}


class FigureGenerator:
    """Gera todas as figuras do artigo a partir dos logs e CSVs."""

    def __init__(self, cfg: dict, exp_dir: Path):
        self.cfg      = cfg
        self.exp_dir  = exp_dir
        self.fig_cfg  = cfg.get("figures", {})
        self.out_dir  = exp_dir / self.fig_cfg.get("output_dir", "figures")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.dpi      = self.fig_cfg.get("dpi",    300)
        self.fmt      = self.fig_cfg.get("format", "pdf")
        self.seed_viz = self.fig_cfg.get("seed_viz", 42)

    def _out(self, name: str) -> Path:
        return self.out_dir / f"{name}.{self.fmt}"

    def run(self) -> None:
        """Gera todas as figuras configuradas."""
        gen = self.fig_cfg.get("generate", [])
        fns = {
            "temporal_trace":  self.temporal_trace,
            "entropy_dist":    self.entropy_dist,
            "gating_heatmap":  self.gating_heatmap,
            "tradeoff":        self.tradeoff,
            "multiseed":       self.multiseed,
            "prob_dist":       self.prob_dist,
        }
        for name in gen:
            if name in fns:
                print(f"    → {name}")
                try:
                    fns[name]()
                except Exception as e:
                    print(f"      ⚠️  Erro em {name}: {e}")

    # ── Carregamento de dados ─────────────────────────────────────────────────

    def _load_frame_logs(self) -> Optional[pd.DataFrame]:
        p = self.exp_dir / "logs" / "episode_frame_logs.csv"
        return pd.read_csv(p) if p.exists() else None

    def _load_borderline_logs(self) -> Optional[pd.DataFrame]:
        p = self.exp_dir / "logs" / "borderline_frame_logs.csv"
        return pd.read_csv(p) if p.exists() else None

    def _load_results(self) -> Optional[pd.DataFrame]:
        p = self.exp_dir / "metrics" / "results_all_seeds.csv"
        return pd.read_csv(p) if p.exists() else None

    # ── fig_temporal_trace ────────────────────────────────────────────────────

    def temporal_trace(self) -> None:
        """
        Figura 2×2: (Baseline | AF-TKD) × (progressivo | abrupto)
        Mostra p(t), H(t), regiões de supressão, marcador t0.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("      matplotlib não disponível — pulando")
            return

        df = self._load_frame_logs()
        if df is None:
            print("      episode_frame_logs.csv não encontrado — pulando")
            return

        df_viz = df[df["seed"] == self.seed_viz]
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=False)

        config_pairs = [
            ("baseline_fixed", "Baseline"),
            ("aftkd_hybrid",   "AF-TKD+Gating"),
        ]
        regime_pairs = [("progressive", "Progressivo"), ("abrupt", "Abrupto")]

        for col, (config_id, config_label) in enumerate(config_pairs):
            for row, (regime, regime_label) in enumerate(regime_pairs):
                ax = axes[row][col]
                sub = df_viz[
                    (df_viz["config"]  == config_id) &
                    (df_viz["regime"]  == regime) &
                    (df_viz["is_critical"] == 1)
                ]
                if sub.empty:
                    ax.set_title(f"{config_label} / {regime_label}")
                    continue

                # Pega primeiro episódio
                ep_id = sub["episode_id"].iloc[0]
                ep = sub[sub["episode_id"] == ep_id].sort_values("t")

                t_axis = ep["time_s"].values
                p_vals = ep["p_t"].values
                H_vals = ep["H_t"].values
                t0     = float(ep["t0_time_s"].iloc[0])
                supp   = ep["is_suppressed"].values.astype(bool)

                # Regiões suprimidas
                for t_idx in np.where(supp)[0]:
                    ax.axvspan(t_axis[t_idx] - 0.05,
                               t_axis[t_idx] + 0.05,
                               alpha=0.15, color="gray")

                ax.plot(t_axis, p_vals, "b-", lw=1.5, label="$p(t)$")
                ax.plot(t_axis, H_vals, "r--", lw=1.0, label="$H(t)$")
                if t0 >= 0:
                    ax.axvline(t0, color="k", lw=1.5, ls="--", label="$t_0$")

                ax.set_title(f"{config_label} / {regime_label}", fontsize=9)
                ax.set_ylim(-0.05, 1.1)
                ax.set_xlabel("Tempo (s)", fontsize=8)
                ax.grid(alpha=0.3)
                if row == 0 and col == 0:
                    ax.legend(fontsize=7, loc="upper left")

        plt.tight_layout()
        plt.savefig(self._out("fig_temporal_trace"), dpi=self.dpi,
                    bbox_inches="tight")
        plt.close()
        print(f"      ✓ fig_temporal_trace.{self.fmt}")

    # ── fig_entropy_dist ──────────────────────────────────────────────────────

    def entropy_dist(self) -> None:
        """
        Distribuição de H(t) por fase temporal (stable / pre_onset / post_onset).
        Painel (a) Baseline, (b) AF-TKD, (c) diagrama de confiabilidade.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        df = self._load_frame_logs()
        if df is None:
            return

        df_viz = df[(df["seed"] == self.seed_viz) &
                    (df["config"].isin(["baseline_fixed", "aftkd_fixed"]))]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        phases = ["stable", "pre_onset", "post_onset"]
        phase_colors = {"stable": "#3B82F6", "pre_onset": "#F59E0B",
                        "post_onset": "#EF4444"}

        for ax_idx, (config_id, label) in enumerate(
                [("baseline_fixed", "(a) Baseline"),
                 ("aftkd_fixed",    "(b) AF-TKD")]):
            ax = axes[ax_idx]
            sub = df_viz[df_viz["config"] == config_id]
            for phase in phases:
                vals = sub[sub["phase"] == phase]["H_t"].dropna().values
                if len(vals) > 0:
                    ax.hist(vals, bins=30, alpha=0.6, density=True,
                            color=phase_colors[phase], label=phase)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("$H_t$", fontsize=9)
            ax.set_ylabel("Densidade", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        # Painel (c): diagrama de confiabilidade
        ax = axes[2]
        for config_id, color, label in [
                ("baseline_fixed", "#555555", "Baseline"),
                ("aftkd_fixed",    "#16A34A", "AF-TKD")]:
            sub = df_viz[df_viz["config"] == config_id]
            p_bins = np.linspace(0, 1, 11)
            centers, accs = [], []
            for i in range(len(p_bins) - 1):
                mask = (sub["p_t"] >= p_bins[i]) & (sub["p_t"] < p_bins[i+1])
                if mask.sum() > 10:
                    centers.append((p_bins[i] + p_bins[i+1]) / 2)
                    accs.append(float(sub[mask]["y_frame_clean"].mean()))
            ax.plot(centers, accs, "o-", color=color, label=label, lw=1.5)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfeito")
        ax.set_title("(c) Confiabilidade frame", fontsize=10)
        ax.set_xlabel("$p_t$ previsto", fontsize=9)
        ax.set_ylabel("Alvo real", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self._out("fig_entropy_dist"), dpi=self.dpi,
                    bbox_inches="tight")
        plt.close()
        print(f"      ✓ fig_entropy_dist.{self.fmt}")

    # ── fig_gating_heatmap ────────────────────────────────────────────────────

    def gating_heatmap(self) -> None:
        """
        Heatmap de H(t) sobre episódios críticos para Baseline vs AF-TKD.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        df = self._load_frame_logs()
        if df is None:
            return

        df_viz = df[(df["seed"] == self.seed_viz) & (df["is_critical"] == 1)]
        T = self.cfg["dataset"]["window"]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax_idx, config_id in enumerate(["baseline_hybrid", "aftkd_hybrid"]):
            sub = df_viz[df_viz["config"] == config_id]
            ep_ids = sub["episode_id"].unique()[:120]
            mat = np.zeros((len(ep_ids), T))
            for i, ep_id in enumerate(ep_ids):
                ep = sub[sub["episode_id"] == ep_id].sort_values("t")
                mat[i, :len(ep)] = ep["H_t"].values[:T]

            ax = axes[ax_idx]
            im = ax.imshow(mat, aspect="auto", cmap="viridis",
                           interpolation="none", vmin=0, vmax=1)
            ax.set_title(f"$H_t$ — {LABELS.get(config_id, config_id)}",
                         fontsize=9)
            ax.set_xlabel("Frame", fontsize=8)
            ax.set_ylabel("Episódio", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Painel (c): média de H(t) relativo ao onset
        ax = axes[2]
        dt = 10.0 / T
        for config_id, color in [("baseline_hybrid", "#E07B39"),
                                   ("aftkd_hybrid",   "#16A34A")]:
            sub = df_viz[df_viz["config"] == config_id]
            # Agrupa por (t - t0)
            sub = sub.copy()
            sub["dt_onset"] = sub["t"] - sub["t0_frame"]
            grp = sub[sub["dt_onset"].between(-15, 20)].groupby("dt_onset")["H_t"]
            x_vals = sorted(grp.groups.keys())
            means  = [grp.get_group(x).mean() for x in x_vals]
            stds   = [grp.get_group(x).std()  for x in x_vals]
            t_axis = [x * dt for x in x_vals]
            ax.plot(t_axis, means, color=color,
                    label=LABELS.get(config_id, config_id), lw=1.5)
            ax.fill_between(t_axis,
                            [m - s/2 for m,s in zip(means, stds)],
                            [m + s/2 for m,s in zip(means, stds)],
                            alpha=0.2, color=color)
        ax.axvline(0, color="k", lw=1.5, ls="--", label="$t_0$")
        ax.set_xlabel("$\\Delta t = t - t_0$ (s)", fontsize=9)
        ax.set_ylabel("$\\bar{H}_t$", fontsize=9)
        ax.set_title("(c) $\\bar{H}_t$ relativo ao onset", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self._out("fig_gating_heatmap"), dpi=self.dpi,
                    bbox_inches="tight")
        plt.close()
        print(f"      ✓ fig_gating_heatmap.{self.fmt}")

    # ── fig_tradeoff ──────────────────────────────────────────────────────────

    def tradeoff(self) -> None:
        """
        Mapa de desempenho conjunto: FR vs Cost (ms/q) com IC 95%.
        FONTE: results_all_seeds.csv (4 seeds, métricas quantitativas).
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from scipy import stats as sp_stats
        except ImportError:
            return

        df = self._load_results()
        if df is None:
            return

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.axhspan(0.05, 1.0, alpha=0.08, color="red",
                   label="FR > 0.05 (cobertura insatisfatória)")

        for config_id, color, label in [
            ("baseline_fixed",  "#555555", "Baseline Fixo"),
            ("baseline_hybrid", "#E07B39", "Baseline+Gating"),
            ("aftkd_fixed",     "#3B82F6", "AF-TKD Fixo"),
            ("aftkd_hybrid",    "#16A34A", "AF-TKD+Gating"),
        ]:
            sub = df[df["config_id"] == config_id]
            if sub.empty:
                continue
            cost_vals = sub["Cost_ms_per_frame"].values
            fr_vals   = sub["FailRate"].values
            n = len(cost_vals)
            cost_mean, cost_sem = cost_vals.mean(), cost_vals.std() / np.sqrt(n)
            fr_mean,   fr_sem   = fr_vals.mean(),   fr_vals.std()   / np.sqrt(n)
            ci = sp_stats.t.ppf(0.975, df=max(n-1,1))
            ax.errorbar(
                cost_mean, fr_mean,
                xerr=ci * cost_sem, yerr=ci * fr_sem,
                fmt="o", color=color, label=label,
                capsize=4, markersize=8, lw=1.5,
            )

        ax.set_xlabel("Custo efetivo (ms/q)", fontsize=10)
        ax.set_ylabel("Taxa de falha (FR)", fontsize=10)
        ax.set_title("Trade-off: cobertura vs eficiência", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(-0.01, 0.35)

        plt.tight_layout()
        plt.savefig(self._out("fig_tradeoff"), dpi=self.dpi,
                    bbox_inches="tight")
        plt.close()
        print(f"      ✓ fig_tradeoff.{self.fmt}")

    # ── fig_multiseed ─────────────────────────────────────────────────────────

    def multiseed(self) -> None:
        """
        Desempenho por seed com IC 95%.
        FONTE: results_all_seeds.csv (4 seeds).
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        df = self._load_results()
        if df is None:
            return

        seeds = sorted(df["Seed"].unique()) if "Seed" in df.columns else []
        if not seeds:
            return

        metrics   = ["F1", "FailRate", "Cost_ms_per_frame"]
        ylabels   = ["F1", "FR", "ms/q"]
        thresholds = [None, 0.05, None]

        fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4), sharey=False)
        for ax, metric, ylabel, thr in zip(axes, metrics, ylabels, thresholds):
            for config_id, color, label in [
                ("baseline_fixed",  "#555555", "BL Fixo"),
                ("baseline_hybrid", "#E07B39", "BL+Gating"),
                ("aftkd_fixed",     "#3B82F6", "AFTKD Fixo"),
                ("aftkd_hybrid",    "#16A34A", "AFTKD+Gating"),
            ]:
                sub = df[df["config_id"] == config_id]
                if sub.empty or metric not in sub.columns:
                    continue
                y_vals = [float(sub[sub["Seed"] == s][metric].iloc[0])
                          for s in seeds
                          if len(sub[sub["Seed"] == s]) > 0]
                ax.plot(seeds[:len(y_vals)], y_vals, "o-",
                        color=color, label=label, lw=1.5, markersize=6)
            if thr is not None:
                ax.axhline(thr, color="red", ls="--", lw=1, alpha=0.7,
                           label=f"Limite={thr}")
            ax.set_title(ylabel, fontsize=10)
            ax.set_xlabel("Seed", fontsize=8)
            ax.set_xticks(seeds)
            ax.grid(alpha=0.3)
            if metric == "F1":
                ax.legend(fontsize=6, loc="lower right")

        plt.tight_layout()
        plt.savefig(self._out("fig_multiseed"), dpi=self.dpi,
                    bbox_inches="tight")
        plt.close()
        print(f"      ✓ fig_multiseed.{self.fmt}")

    # ── fig_prob_dist ─────────────────────────────────────────────────────────

    def prob_dist(self) -> None:
        """
        Distribuição de p_max por nível de risco.
        FONTE: borderline_frame_logs.csv (seed_viz).
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        df = self._load_borderline_logs()
        if df is None:
            return

        df_viz = df[df["seed"] == self.seed_viz]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        levels = ["Normal", "Atenção", "Alerta", "Crítico"]
        colors = ["#3B82F6", "#F59E0B", "#F97316", "#EF4444"]

        for ax_idx, (config_id, label) in enumerate([
                ("baseline_fixed", "(a) Baseline"),
                ("aftkd_fixed",    "(b) AF-TKD")]):
            ax = axes[ax_idx]
            sub = df_viz[df_viz["config"] == config_id]
            ep = sub.groupby(["risk_level", "episode_id"])["p_t"].max().reset_index(
                name="p_max")
            for level, color in zip(levels, colors):
                vals = ep[ep["risk_level"] == level]["p_max"].values
                if len(vals) > 0:
                    ax.hist(vals, bins=20, alpha=0.6, density=True,
                            color=color, label=level)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("$\\bar{p}_{\\max}$", fontsize=9)
            if ax_idx == 0:
                ax.set_ylabel("Densidade", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self._out("fig_prob_dist"), dpi=self.dpi,
                    bbox_inches="tight")
        plt.close()
        print(f"      ✓ fig_prob_dist.{self.fmt}")
