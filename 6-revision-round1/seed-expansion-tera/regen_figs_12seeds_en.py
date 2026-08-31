#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenera fig_multiseed (EN) e fig_tradeoff (EN) para a campanha de 12 seeds
(exp_seed12_round1), com fontes maiores (atende R2.6 nestas figuras).
Fonte de dados: results_all_seeds.csv canonico da campanha (nenhum
pos-processamento manual; apenas plot).
Saida: --outdir (image/ do artigo revisado).
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE.parent.parent / "1-pipeline-tera" / "results" / "exp_seed12_round1" / "metrics" / "results_all_seeds.csv"

CONFIGS = [
    ("baseline_fixed",  "Baseline\nFixed",   "Baseline Fixed",             "#4C72B0", "o"),
    ("baseline_hybrid", "Baseline\n+MHEG",   "Baseline+MHEG\n(neg. control)", "#55A868", "s"),
    ("aftkd_fixed",     "AF-TOI\nFixed",     "AF-TOI Fixed",               "#C44E52", "^"),
    ("aftkd_hybrid",    "AF-TOI\n+MHEG",     "AF-TOI+MHEG",                "#8172B3", "D"),
]

def ci95(v):
    v = np.asarray(v, float)
    return stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV)
    seeds = sorted(df["Seed"].unique())
    n = len(seeds)
    assert n == 12, f"esperava 12 seeds, achei {n}"

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                         "axes.labelsize": 12, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "legend.fontsize": 10.5})

    # ---------- fig_multiseed_en (12 seeds) ----------
    panels = [("F1", r"F1 $\uparrow$", "(a) Detection"),
              ("FailRate", r"Failure Rate (FR) $\downarrow$", "(b) Coverage"),
              ("TTDef", r"TTD$_{\mathrm{ef}}$ (s) $\downarrow$", "(c) Effective latency")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    rng = np.random.default_rng(0)  # jitter fixo, reprodutivel
    for ax, (metric, ylabel, title) in zip(axes, panels):
        for i, (key, xlab, _leg, color, marker) in enumerate(CONFIGS):
            vals = df[df.config_id == key].sort_values("Seed")[metric].values
            jit = rng.uniform(-0.16, 0.16, size=n)
            ax.scatter(np.full(n, i) + jit, vals, s=42, marker=marker,
                       color=color, alpha=0.35, edgecolors="none", zorder=2)
            m, h = vals.mean(), ci95(vals)
            ax.errorbar([i], [m], yerr=[h], fmt=marker, color=color, ms=13,
                        mec="white", mew=0.8, capsize=6, lw=2.2, zorder=3)
        ax.set_xticks(range(4))
        ax.set_xticklabels([c[1] for c in CONFIGS])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.grid(axis="y", ls="--", alpha=0.4)
        ax.set_xlim(-0.5, 3.5)
        if metric == "FailRate":
            ax.axhline(0.05, color="red", ls="--", lw=1.6)
            ymax = max(0.2, df.FailRate.max() * 1.15)
            ax.axhspan(0.05, ymax, color="red", alpha=0.10)
            ax.set_ylim(-0.008, ymax)
            ax.text(3.42, 0.054, "FR budget", color="red", fontsize=10.5,
                    ha="right", va="bottom")
    h_pts = plt.Line2D([], [], marker="o", ls="", color="gray", alpha=0.4,
                       ms=7, label=f"individual seeds (n={n})")
    h_ci = plt.Line2D([], [], marker="o", ls="-", color="gray", ms=11,
                      label=r"mean $\pm$ 95% CI")
    axes[2].legend(handles=[h_pts, h_ci], loc="upper right", framealpha=0.9)
    fig.suptitle("Per-seed performance across twelve independent partitions "
                 "(95% CI, seeds 42–53)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "10-fig_multiseed_en.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---------- fig_tradeoff (EN, 12 seeds) ----------
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    for key, _xlab, leg, color, marker in CONFIGS:
        sub = df[df.config_id == key]
        x, y = sub.Cost_ms_per_frame.values, sub.FailRate.values
        ax.errorbar(x.mean(), y.mean(), xerr=ci95(x), yerr=ci95(y),
                    fmt=marker, color=color, ms=14, mec="white", mew=0.8,
                    capsize=5, lw=2.0, label=leg, zorder=3)
    ax.axhline(0.05, color="red", ls="--", lw=2)
    xmax = df.Cost_ms_per_frame.max() * 1.25
    ymax = 0.42
    ax.axhspan(0.05, ymax, color="red", alpha=0.10, zorder=1)
    ax.text(xmax * 0.985, 0.056, "FR = 0.05", color="red", fontsize=12, ha="right")
    ax.text(xmax * 0.5, 0.28, "FR > 0.05\n(unsafe scheduling zone)",
            color="#c44", fontsize=13, ha="center", style="italic")
    ax.annotate("lower cost", xy=(0.02, -0.028), xytext=(0.32, -0.028),
                arrowprops=dict(arrowstyle="->", color="gray"),
                color="gray", fontsize=11, va="center",
                annotation_clip=False)
    ax.set_xlabel("Effective inference cost (ms/frame)")
    ax.set_ylabel("Failure Rate (FR)")
    ax.set_title("Coverage vs. efficiency trade-off (95% CI, 12 seeds)", pad=12)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.035, ymax)
    ax.grid(ls="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out / "7-fig_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"figuras regeneradas (n={n} seeds) em {out}")

if __name__ == "__main__":
    main()
