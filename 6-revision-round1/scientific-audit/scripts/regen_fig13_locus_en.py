#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenera a Figura 13 do artigo (locus ECE x TTDef, EN) a partir de artefatos
versionados: paper_calibration_macros.tex (variantes TS/PS/IC, 4 seeds) e
expanded_per_seed.csv (pontos de referencia Baseline/AF-TOI, seeds 42-45).
Corrige o titulo ("95% CI" sem escape LaTeX vazado) e usa fontes maiores (R2.6).
CI 95% com t(3)=3.182 sobre 4 seeds: ci = 3.182*sd/sqrt(4).
"""
import re, sys, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
ART  = HERE.parent.parent.parent   # tera-adhoc-artifact
MACROS = ART / "3-calibration-posthoc" / "paper_calibration_macros.tex"
PERSEED = ART / "6-revision-round1" / "seed-expansion-tera" / "expanded_per_seed.csv"
T3 = 3.1824  # t two-sided 95%, df=3

def load_macros():
    m = {}
    for name, val in re.findall(r"\\newcommand{\\(\w+)}{([\d.]+)}", MACROS.read_text()):
        m[name] = float(val)
    return m

def ci4(sd): return T3 * sd / 2.0

def main(out_path):
    m = load_macros()
    df = pd.read_csv(PERSEED)
    df = df[df.Seed.isin([42, 43, 44, 45])]
    ref = {}
    for cid, lab in [("baseline_fixed", "Baseline"), ("baseline_hybrid", "Baseline+MHEG"),
                     ("aftkd_fixed", "AF-TOI Fixed"), ("aftkd_hybrid", "AF-TOI+MHEG")]:
        g = df[df.config_id == cid]
        ref[lab] = (g.ECE.mean(), g.ECE.std(ddof=1), g.TTDef.mean(), g.TTDef.std(ddof=1))
    variants = [
        ("Baseline+TS",      m["CalibTSFixedECE"], m["CalibTSFixedECEStd"], m["CalibTSFixedTTDef"], m["CalibTSFixedTTDefStd"], "#4C72B0", "o"),
        ("Baseline+PS",      m["CalibPSFixedECE"], m["CalibPSFixedECEStd"], m["CalibPSFixedTTDef"], m["CalibPSFixedTTDefStd"], "#DD8452", "s"),
        ("Baseline+IC",      m["CalibICFixedECE"], m["CalibICFixedECEStd"], m["CalibICFixedTTDef"], m["CalibICFixedTTDefStd"], "#8172B3", "D"),
        ("Baseline+TS+MHEG", m["CalibTSHybrECE"],  m["CalibTSHybrECEStd"],  m["CalibTSHybrTTDef"],  m["CalibTSHybrTTDefStd"],  "#4C72B0", "^"),
        ("Baseline+PS+MHEG", m["CalibPSHybrECE"],  m["CalibPSHybrECEStd"],  m["CalibPSHybrTTDef"],  m["CalibPSHybrTTDefStd"],  "#DD8452", "v"),
        ("Baseline+IC+MHEG", m["CalibICHybrECE"],  m["CalibICHybrECEStd"],  m["CalibICHybrTTDef"],  m["CalibICHybrTTDefStd"],  "#8172B3", "P"),
    ]
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5,
                         "axes.labelsize": 12, "legend.fontsize": 10,
                         "xtick.labelsize": 11, "ytick.labelsize": 11})
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for lab, e, esd, t, tsd, c, mk in variants:
        ax.errorbar(e, t, xerr=ci4(esd), yerr=ci4(tsd), color=c, marker=mk,
                    ms=9, capsize=3, lw=1.4, ls="none", label=lab, alpha=0.95)
    for lab, c, mk in [("Baseline", "#555555", "o"), ("Baseline+MHEG", "#2E8B57", "s")]:
        e, esd, t, tsd = ref[lab]
        ax.errorbar(e, t, xerr=ci4(esd), yerr=ci4(tsd), color=c, marker=mk,
                    ms=10, capsize=3, lw=1.4, ls="none", label=lab)
    for lab, c, mk in [("AF-TOI Fixed", "#C44E52", "*"), ("AF-TOI+MHEG", "#C44E52", "X")]:
        e, esd, t, tsd = ref[lab]
        ax.errorbar(e, t, xerr=ci4(esd), yerr=ci4(tsd), color=c, marker=mk,
                    ms=14 if mk == "*" else 10, capsize=3, lw=1.6, ls="none",
                    label=lab, zorder=5)
    e_af = ref["AF-TOI Fixed"][0]
    ax.annotate("AF-TOI\nfrontier", xy=(e_af, ref["AF-TOI Fixed"][2]),
                xytext=(e_af + 0.05, 0.28), color="#C44E52", fontsize=11,
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2))
    ax.set_xlabel("ECE (post-calibration)  $\\downarrow$ better")
    ax.set_ylabel("TTD$_{\\mathrm{ef}}$ (s)  $\\downarrow$ better")
    ax.set_title("ECE vs. TTD$_{\\mathrm{ef}}$: calibration improves ECE\n"
                 "without reaching the AF-TOI operational frontier "
                 "(95% CI, seeds 42–45)")
    ax.grid(alpha=0.25, ls="--")
    ax.legend(loc="upper left", framealpha=0.95, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print("saved:", out_path)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fig_neg_control_operating_curve.pdf")
