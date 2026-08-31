#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consolida as 12 seeds e compara a estatistica com a campanha de 4 seeds.

Gera, nesta pasta: expanded_summary.csv (media +- sd por configuracao),
expanded_per_seed.csv (observacoes individuais), expanded_paper_table2_rows.tex
(linhas no formato da Table 8, para conferencia cruzada com as geradas pelo
tera_latex do pipeline) e expanded_stats_report.txt (Wilcoxon pareado exato e
Cliff's delta, n=4 vs n=12). Todas as seeds entram no relatorio,
independentemente do desfecho; o artigo nao e alterado por este script.
"""
import csv
import math
import statistics as st
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
NEW_CSV = ROOT / "1-pipeline-tera" / "results" / "exp_seed12_round1" / "metrics" / "results_all_seeds.csv"
REF_CSV = ROOT / "2-campaign-exp_20260519_055950" / "metrics" / "results_all_seeds.csv"

CONFIGS = ["baseline_fixed", "baseline_hybrid", "aftkd_fixed", "aftkd_hybrid"]
METRICS = ["F1", "ECE", "FailRate", "TTD_Progressive", "TTD", "TTDef",
           "Cost_ms_per_frame", "SkipPct"]
STAT_METRICS = ["F1", "ECE", "FailRate", "TTDef", "Cost_ms_per_frame"]
PAIRS = [("baseline_fixed", "aftkd_fixed"),
         ("baseline_hybrid", "aftkd_hybrid"),
         ("aftkd_fixed", "aftkd_hybrid"),
         ("baseline_fixed", "baseline_hybrid")]


def load(p):
    data = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            data.setdefault(r["config_id"], {})[int(r["Seed"])] = r
    return data


def vals(data, cfg, m, seeds=None):
    ss = sorted(data[cfg]) if seeds is None else seeds
    return [float(data[cfg][s][m]) for s in ss if s in data[cfg]]


def cliffs(a, b):
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def wilcoxon_exact(a, b):
    # implementacao propria usada apenas quando o scipy nao esta disponivel
    d = [x - y for x, y in zip(a, b) if x != y]
    n = len(d)
    if n == 0:
        return 1.0
    from collections import Counter
    sabs = sorted(abs(x) for x in d)
    cnt = Counter(sabs)
    pos, start = {}, 1
    for v in sorted(cnt):
        k = cnt[v]
        pos[v] = (2 * start + k - 1) / 2
        start += k
    ranks = [pos[abs(x)] for x in d]
    wp = sum(r for r, x in zip(ranks, d) if x > 0)
    wmin = min(wp, sum(ranks) - wp)
    tot = le = 0
    for signs in product([0, 1], repeat=n):
        w = sum(r for r, s in zip(ranks, signs) if s)
        tot += 1
        if w <= wmin:
            le += 1
    return min(1.0, 2 * le / tot)


def wilcoxon(a, b):
    try:
        from scipy.stats import wilcoxon as sw
        return float(sw(a, b, zero_method="wilcox", alternative="two-sided",
                        method="exact").pvalue)
    except Exception:
        return wilcoxon_exact(a, b)


def main():
    if not NEW_CSV.exists():
        raise SystemExit(f"{NEW_CSV} nao existe; rode a campanha e a validacao antes")
    new = load(NEW_CSV)
    ref = load(REF_CSV)
    seeds12 = sorted({s for c in new for s in new[c]})
    print(f"seeds na campanha nova: {seeds12}")
    if len(seeds12) != 12:
        print(f"aviso: esperadas 12 seeds, encontradas {len(seeds12)}")

    with open(HERE / "expanded_per_seed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config_id", "Seed"] + METRICS)
        for c in CONFIGS:
            for s in sorted(new.get(c, {})):
                w.writerow([c, s] + [new[c][s][m] for m in METRICS])

    with open(HERE / "expanded_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config_id", "n_seeds"] +
                   [x for m in METRICS for x in (m + "_mean", m + "_sd")])
        for c in CONFIGS:
            line = [c, len(new.get(c, {}))]
            for m in METRICS:
                v = vals(new, c, m)
                line += [f"{st.mean(v):.4f}", f"{st.stdev(v):.4f}" if len(v) > 1 else "0"]
            w.writerow(line)

    def cell(c, m, bold=False):
        v = vals(new, c, m)
        t = f"{st.mean(v):.3f}\\pm{st.stdev(v):.3f}"
        return f"$\\mathbf{{{t}}}$" if bold else f"${t}$"

    order = [("baseline_fixed", "LSTM-Baseline", "No", False),
             ("baseline_hybrid", "LSTM-Baseline", "Yes", False),
             ("aftkd_fixed", "LSTM-AF-TOI", "No", False),
             ("aftkd_hybrid", "LSTM-AF-TOI", "\\textbf{Yes}", True)]
    with open(HERE / "expanded_paper_table2_rows.tex", "w", encoding="utf-8") as f:
        f.write("% conferencia cruzada das linhas da Table 8 (n=12); a versao\n"
                "% aplicada no artigo vem do tera_latex do pipeline\n")
        for c, name, mheg, bold in order:
            cells = [cell(c, m, bold) for m in METRICS]
            if c == "baseline_hybrid":
                cells[4] = "{---}"
            f.write(f"{name} & {mheg} & " + " & ".join(cells) + " \\\\\n")

    lines = ["Estatistica pareada por seed: campanha publicada (n=4) vs expandida (n=12)",
             "Wilcoxon signed-rank bicaudal exato; Cliff's delta.",
             "=" * 76]
    for c1, c2 in PAIRS:
        lines.append(f"\n{c1}  vs  {c2}")
        lines.append(f"  {'metrica':18s} {'n=4 p':>8s} {'n=4 delta':>10s} "
                     f"{'n=12 p':>8s} {'n=12 delta':>10s}")
        for m in STAT_METRICS:
            a4, b4 = vals(ref, c1, m), vals(ref, c2, m)
            a12, b12 = vals(new, c1, m), vals(new, c2, m)
            p4, d4 = wilcoxon(a4, b4), cliffs(a4, b4)
            p12, d12 = wilcoxon(a12, b12), cliffs(a12, b12)
            sig = " *" if p12 < 0.05 else ""
            lines.append(f"  {m:18s} {p4:8.4f} {d4:+10.3f} {p12:8.4f}{sig} {d12:+10.3f}")
    lines.append("\n* p < 0.05 na campanha n=12.")
    report = "\n".join(lines)
    (HERE / "expanded_stats_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
