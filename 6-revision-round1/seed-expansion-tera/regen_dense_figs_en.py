#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2.6 — Regenera as cinco figuras densas do artigo com fontes maiores (EN).
Mesmos dados canonicos (seed 42 reproduzida bit a bit pela campanha nova):
  fig 4  4-fig_entropy_dist_en.pdf   <- episode_frame_logs.csv (exp_seed12_round1)
  fig 5  5-fig_temporal_trace_en.pdf <- idem
  fig 6  6-fig_gating_heatmap_en.pdf <- idem
  fig 8  8-fig_ttdef_dist.pdf        <- df_episodios_cdf.csv (campanha 4 seeds publicada)
  fig 11 11-fig_prob_dist_en.pdf     <- borderline_frame_logs.csv (replay, A/L recipes)
Apenas replot (composicao identica, rotulos EN, fontes >= 11pt); nenhuma
metrica recalculada alem das ja exibidas nas figuras originais.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
# Figs 4/5/6 usam os logs de frames do protocolo replay (mesma fonte das
# figuras publicadas — ECE 0.129/0.044 reproduzidos exatamente; H_t ja em nats)
LOGS = ROOT / "5-robustness-replay-runv29" / "Inferencia"
EPCSV = ROOT / "2-campaign-exp_20260519_055950" / "df_episodios_cdf.csv"
BLCSV = ROOT / "5-robustness-replay-runv29" / "Inferencia" / "borderline_frame_logs.csv"
LN2 = np.log(2.0)
THETA = 0.1

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13.5,
                     "axes.labelsize": 12.5, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 11})


def nats_axis(ax, axis="x"):
    ticks = [0, LN2 / 2, LN2]
    labels = ["0", r"$\frac{\ln 2}{2}$", r"$\ln 2$"]
    if axis == "x":
        ax.set_xticks(ticks); ax.set_xticklabels(labels)
        for v in ticks[1:]:
            ax.axvline(v, color="gray", ls=":", lw=1.0, alpha=0.8)
    else:
        ax.set_yticks(ticks); ax.set_yticklabels(labels)


def fig_entropy_dist(df, out):
    dv = df[(df.seed == 42) & df.config.isin(["baseline_fixed", "afkd_fixed"])].copy()
    dv["H_nats"] = dv.H_t  # ja em nats
    groups = [("normal", "Normal", "#4C72B0"), ("stable", "Stable", "#55A868"),
              ("pre_onset", "Pre-onset", "#D4A017"), ("post_onset", "Post-onset", "#C44E52")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (cfg, title) in zip(axes[:2], [("baseline_fixed", "(a) Baseline — phases blur"),
                                           ("afkd_fixed", "(b) AF-TOI — phases stratify")]):
        sub = dv[dv.config == cfg]
        xs = np.linspace(0, LN2 * 1.02, 400)
        for phase, label, color in groups:
            vals = sub[sub.phase == phase]["H_nats"].dropna().values
            if len(vals) < 20:
                continue
            kde = gaussian_kde(vals, bw_method=0.12)
            ys = kde(xs)
            ax.plot(xs, ys, color=color, lw=2.2, label=label)
            ax.fill_between(xs, ys, color=color, alpha=0.18)
        ax.set_title(title)
        ax.set_xlabel(r"Entropy $H_t$ (nats)")
        ax.set_xlim(-0.01, LN2 * 1.05)
        nats_axis(ax, "x")
        ax.grid(alpha=0.25, ls="--")
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper right", framealpha=0.95)

    # (c) reliability diagram
    ax = axes[2]
    styles = {"baseline_fixed": ("#C44E52", "--", "Baseline"),
              "afkd_fixed": ("#4C72B0", "-", "AF-TOI")}
    edges = np.linspace(0, 1, 11)
    for cfg, (color, ls, name) in styles.items():
        sub = dv[dv.config == cfg]
        centers, accs, ece, N = [], [], 0.0, len(sub)
        for i in range(10):
            m = (sub.p_t >= edges[i]) & (sub.p_t < edges[i + 1])
            if m.sum() > 10:
                conf = float(sub[m].p_t.mean())
                acc = float(sub[m].y_frame_clean.mean())
                centers.append((edges[i] + edges[i + 1]) / 2)
                accs.append(acc)
                ece += (m.sum() / N) * abs(conf - acc)
        ax.plot(centers, accs, marker="o", ls=ls, color=color, lw=2.2, ms=7,
                label=f"{name} (ECE={ece:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1.4, label="Perfect cal.")
    ax.annotate("over-\nconfident", xy=(0.80, 0.47), color="#C44E52",
                fontsize=11, ha="center")
    ax.set_title("(c) Reliability diagram")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel(r"Mean frame risk $y_{\mathrm{frame}}$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for l in [labels[-1]] + labels[:-1]]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25, ls="--")
    fig.tight_layout()
    fig.savefig(out / "4-fig_entropy_dist_en.pdf", bbox_inches="tight")
    plt.close(fig)


def _detect(ep):
    """Primeiro t >= t0 com p >= THETA por 3 quadros consecutivos."""
    t0 = float(ep.t0_time_s.iloc[0])
    e = ep[ep.time_s >= t0].reset_index(drop=True)
    run = 0
    for i in range(len(e)):
        run = run + 1 if e.p_t.iloc[i] >= THETA else 0
        if run >= 3:
            j = i - 2
            return float(e.time_s.iloc[j]), float(e.p_t.iloc[j])
    return None, None


def fig_temporal_trace(df, out):
    dv = df[(df.seed == 42) & (df.is_critical == 1)]
    # escolhe episodios com t0 mais proximos dos da figura publicada
    ep_ids = {}
    base = dv[dv.config == "afkd_hybrid"]
    # abrupto: t0 proximo ao da figura publicada (~2.9 s)
    cand = base[base.regime == "abrupt"].groupby("episode_id")["t0_time_s"].first()
    ep_ids["abrupt"] = (cand - 2.92).abs().idxmin()
    # progressivo: episodio que ilustra a estratificacao (H inicial baixo,
    # subindo ate perto de ln 2 apos o onset), com t0 em posicao central
    best, best_score = None, None
    for ep_id, g in base[base.regime == "progressive"].groupby("episode_id"):
        g = g.sort_values("t")
        t0 = float(g.t0_time_s.iloc[0])
        if not (4.0 <= t0 <= 7.0):
            continue
        h_pre = float(g[g.time_s < t0 - 1.0].H_t.mean())
        h_post = float(g[g.time_s >= t0].H_t.mean())
        score = h_post - h_pre  # maior estratificacao pre->pos onset
        if best_score is None or score > best_score:
            best, best_score = ep_id, score
    ep_ids["progressive"] = best

    cfgs = [("baseline_hybrid", "Baseline", "#4C72B0"),
            ("afkd_hybrid", "AF-TOI+MHEG", "#C44E52")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6))
    for row, (regime, rlabel) in enumerate([("progressive", "Progressive episode"),
                                            ("abrupt", "Abrupt episode")]):
        for col, (cfg, clabel, color) in enumerate(cfgs):
            ax = axes[row][col]
            ep = dv[(dv.config == cfg) & (dv.episode_id == ep_ids[regime])].sort_values("t")
            t = ep.time_s.values
            supp = ep.is_suppressed.values.astype(bool)
            # spans contiguos suprimidos
            i = 0
            while i < len(supp):
                if supp[i]:
                    j = i
                    while j + 1 < len(supp) and supp[j + 1]:
                        j += 1
                    ax.axvspan(t[i] - 0.05, t[j] + 0.05, color="gray", alpha=0.22, lw=0)
                    i = j + 1
                else:
                    i += 1
            ax.plot(t, ep.p_t.values, color=color, lw=2.6, zorder=3)
            ax2 = ax.twinx()
            ax2.plot(t, ep.H_t.values, color=color, lw=1.8, ls="--",
                     alpha=0.55, zorder=2)
            ax2.set_ylim(0, 0.72)
            if col == 1:
                ax2.set_ylabel(r"$H(t)$ (nats)")
            else:
                ax2.set_yticklabels([])
            t0 = float(ep.t0_time_s.iloc[0])
            ax.axvline(t0, color="k", lw=2.0, zorder=4)
            ax.axhline(THETA, color="k", ls=":", lw=1.3, alpha=0.7)
            td, pd_ = _detect(ep)
            if td is not None:
                ax.plot([td], [pd_], marker="v", color=color, ms=13,
                        mec="k", mew=0.6, zorder=5)
            ax.set_ylim(-0.04, 1.06)
            ax.set_xlim(t[0], t[-1])
            if row == 0:
                ax.set_title(clabel, fontweight="bold", fontsize=14, pad=10)
            if col == 0:
                ax.set_ylabel(f"{rlabel}\n$p(t)$", fontsize=12.5)
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=0.2, ls="--")
    handles = [Line2D([], [], color="#4C72B0", lw=2.6, label="$p(t)$ Baseline"),
               Line2D([], [], color="#C44E52", lw=2.6, label="$p(t)$ AF-TOI+MHEG"),
               Line2D([], [], color="gray", lw=1.8, ls="--", label="$H(t)$ (dashed)"),
               Patch(facecolor="gray", alpha=0.22, label="Suppressed (MHEG)"),
               Line2D([], [], color="k", lw=2.0, label=r"$t_{\mathrm{onset}}$"),
               Line2D([], [], color="k", lw=1.3, ls=":", label=r"$\theta_{\mathrm{det}}$"),
               Line2D([], [], marker="v", color="gray", ls="", ms=11, mec="k",
                      label=r"$t_{\mathrm{det}}$ marker")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "5-fig_temporal_trace_en.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_gating_heatmap(df, out):
    dv = df[(df.seed == 42) & (df.is_critical == 1)].copy()
    dv["H_nats"] = dv.H_t  # ja em nats
    T = int(dv.t.max()) + 1
    fig = plt.figure(figsize=(17.5, 5.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.05, 1.05, 0.05, 0.22, 1.35],
                          wspace=0.25)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    spacer = fig.add_subplot(gs[0, 3]); spacer.axis("off")
    axc = fig.add_subplot(gs[0, 4])

    for ax, (cfg, title) in zip(axs, [("baseline_hybrid", "(a) Baseline+MHEG"),
                                      ("afkd_hybrid", "(b) AF-TOI+MHEG")]):
        sub = dv[dv.config == cfg]
        rows, onsets, nprog = [], [], 0
        for regime in ["progressive", "abrupt"]:
            eps = (sub[sub.regime == regime].groupby("episode_id")["t0_frame"]
                   .first().sort_values())
            if regime == "progressive":
                nprog = len(eps)
            for ep_id, t0f in eps.items():
                e = sub[sub.episode_id == ep_id].sort_values("t")
                v = np.zeros(T)
                v[: len(e)] = e.H_nats.values[:T]
                rows.append(v)
                onsets.append(t0f)
        mat = np.vstack(rows)
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                       interpolation="none", vmin=0, vmax=LN2)
        ax.scatter(onsets, range(len(onsets)), marker="|", color="white",
                   s=48, lw=1.6)
        ax.axhline(nprog - 0.5, color="teal", lw=1.6)
        ax.text(1.5, nprog * 0.5, "Prog.", color="white", fontsize=12,
                fontweight="bold", rotation=90, va="center")
        ax.text(1.5, nprog + (len(rows) - nprog) * 0.5, "Abrupt", color="white",
                fontsize=12, fontweight="bold", rotation=90, va="center")
        ax.set_title(title)
        ax.set_xlabel("Time step $t$")
        ax.set_yticks([])
    axs[0].set_ylabel("Critical episodes")
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"Entropy $H_t$ (nats)", fontsize=12)

    for cfg, color, ls, name in [("baseline_hybrid", "#C44E52", "--", "Baseline"),
                                 ("afkd_hybrid", "#4C72B0", "-", "AF-TOI")]:
        sub = dv[dv.config == cfg].copy()
        sub["dt"] = sub.t - sub.t0_frame
        g = sub[sub.dt.between(-20, 29)].groupby("dt")["H_nats"]
        xs = np.array(sorted(g.groups))
        m = np.array([g.get_group(x).mean() for x in xs])
        s = np.array([g.get_group(x).std() for x in xs])
        axc.plot(xs, m, color=color, ls=ls, lw=2.6, label=name)
        axc.fill_between(xs, m - s / 2, m + s / 2, color=color, alpha=0.16)
    axc.axvline(0, color="k", lw=1.8)
    axc.text(0.8, LN2 * 0.86, r"$t_{\mathrm{onset}}$", fontsize=12)
    axc.set_title(r"(c) $\bar{H}_t \pm \frac{1}{2}\sigma$ around $t_{\mathrm{onset}}$")
    axc.set_xlabel(r"$\Delta t = t - t_{\mathrm{onset}}$ (frames)")
    axc.set_ylabel(r"$\bar{H}_t$ (nats)")
    axc.set_ylim(0, LN2 * 1.05)
    nats_axis(axc, "y")
    axc.legend(loc="upper left", framealpha=0.95)
    axc.grid(alpha=0.25, ls="--")
    fig.savefig(out / "6-fig_gating_heatmap_en.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_ttdef_dist(out):
    ep = pd.read_csv(EPCSV)
    configs = [("baseline_fixed", "Baseline Fixed", "#4C72B0"),
               ("baseline_hybrid", "Baseline+MHEG\n(neg. control)", "#55A868"),
               ("aftkd_fixed", "AF-TOI Fixed", "#C44E52"),
               ("aftkd_hybrid", "AF-TOI+MHEG", "#8172B3")]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.6))
    bins = np.linspace(0, 2.5, 21)
    for ax, (cfg, title, color) in zip(axes, configs):
        vals = ep[ep.config_id == cfg]["TTDef"].values
        det = vals[vals < 10]
        n_miss = int((vals >= 10).sum())
        ax.hist(det, bins=bins, color=color, alpha=0.88, edgecolor="white", lw=0.5)
        ax.bar([2.35], [n_miss], width=0.14, color=color, alpha=0.88,
               hatch="//", edgecolor="k", lw=0.6)
        ax.annotate(f"{n_miss}\nmisses\n(10 s)", xy=(2.35, n_miss),
                    xytext=(2.28, n_miss + max(8, n_miss * 0.25)),
                    fontsize=10.5, ha="center")
        ax.text(0.97, 0.94, f"mean={vals.mean():.3f}s\nFR={n_miss/len(vals):.3f}",
                transform=ax.transAxes, fontsize=11, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
        ax.set_title(title, color=color, fontweight="bold", fontsize=12.5)
        ax.set_xlabel(r"TTD$_{\mathrm{ef}}$ (s)")
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.35])
        ax.set_xticklabels(["0.0", "0.5", "1.0", "1.5", "2.0", "10 s\n(miss)"],
                           fontsize=10)
        ax.grid(alpha=0.2, ls="--", axis="y")
    axes[0].set_ylabel("Critical episodes")
    fig.suptitle(r"Distribution of TTD$_{\mathrm{ef}}$ per critical episode "
                 "(seeds 42–45)", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(out / "8-fig_ttdef_dist.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_prob_dist(out):
    bl = pd.read_csv(BLCSV)
    bl = bl[bl.seed == 42]
    pmax = (bl.groupby(["config", "risk_level", "episode_id"])["p_t"]
            .max().reset_index(name="p_max"))
    levels = [("Normal", "Normal", "#4C72B0"), ("Atenção", "Attention", "#55A868"),
              ("Alerta", "Alert", "#E07B39"), ("Crítico", "Critical", "#8172B3")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    for ax, (cfg, title) in zip(axes, [("baseline_fixed", "(a) Baseline Fixed"),
                                       ("afkd_fixed", "(b) AF-TOI Fixed")]):
        sub = pmax[pmax.config == cfg]
        for i, (pt_lv, en_lv, color) in enumerate(levels):
            vals = sub[sub.risk_level == pt_lv]["p_max"].values
            if len(vals) < 3:
                ax.text(i, 0.5, "N/A\n(no data)", ha="center", va="center",
                        fontsize=11, color="gray", style="italic")
                continue
            parts = ax.violinplot([vals], positions=[i], widths=0.75,
                                  showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(color); b.set_alpha(0.55)
                b.set_edgecolor(color); b.set_lw(1.4)
            ax.hlines(np.median(vals), i - 0.22, i + 0.22, color="k", lw=1.8)
            ax.plot([i], [vals.mean()], marker="D", color="k", ms=6)
            pct = 100.0 * float((vals > THETA).mean())
            ax.text(i, -0.085, f"{pct:.1f}% > $\\theta$", ha="center",
                    fontsize=10, color=color)
        ax.axhline(THETA, color="red", ls="--", lw=1.6)
        ax.text(-0.42, THETA + 0.015, r"$\theta = 0.1$", color="red", fontsize=11)
        ax.set_xticks(range(4))
        ax.set_xticklabels([l[1] for l in levels])
        ax.set_xlabel("Risk level")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(-0.13, 1.05)
        ax.grid(alpha=0.2, ls="--", axis="y")
    axes[0].set_ylabel(r"Max. episode probability ($\bar{p}_{\max}$)")
    handles = [Line2D([], [], color="k", lw=1.8, label="Median"),
               Line2D([], [], marker="D", color="k", ls="", ms=6, label="Mean")]
    axes[1].legend(handles=handles, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out / "11-fig_prob_dist_en.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    df = pd.read_csv(LOGS / "episode_frame_logs.csv")
    fig_entropy_dist(df, out);   print("ok 4-fig_entropy_dist_en")
    fig_temporal_trace(df, out); print("ok 5-fig_temporal_trace_en")
    fig_gating_heatmap(df, out); print("ok 6-fig_gating_heatmap_en")
    fig_ttdef_dist(out);         print("ok 8-fig_ttdef_dist")
    # fig 11 (prob_dist / violinos borderline) NAO e regenerada: o recorte de
    # dados da figura publicada (n=37 Attention, p_max 0.048+-0.195) nao foi
    # localizado em disco (nem o replay 240-episodios nem os logs TERA o
    # reproduzem), e regenerar com outro recorte contradiria os numeros do
    # texto do artigo. Mantida a figura original (fontes ja legiveis).
    # fig_prob_dist(out)


if __name__ == "__main__":
    main()
