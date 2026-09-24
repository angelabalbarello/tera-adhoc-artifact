#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agrega os bracos da ablacao de mecanismo (formato REAL do run_v29:
arquivos planos exp_<label>_results_seedNN.csv na pasta FGCS/Inferencia)
e gera a nova Tabela 11.

Rodar DENTRO de FGCS/Inferencia:
    python ..\\..\\Experimentos-Artigo1-AdHoc\\ablacao_mecanismo_temporal\\analyze_mechanism_ablation.py

Le, por seed e braco, as linhas Modelo=LSTM-AF-KD:
  - Politica "Fixa"    -> F1, ECE, FR (FailRate), TTDef
  - Politica "Hibrida" -> SkipPct (e FR/TTDef hibridos, colunas extras)
Estatistica pareada (Wilcoxon exato n<=12 + Cliff's delta) vs braco a015
(alpha=0.15 treinado sob o MESMO protocolo run_v29).
Saidas (nesta pasta de analise): mechanism_ablation_results.csv,
mechanism_ablation_per_seed.csv, stats.json, table11_new.tex
"""
import os, sys, json, glob, itertools, math, csv

SEEDS = list(range(42, 54))
ARMS = {  # label -> (nome na tabela, ordem)
    "mechabl_notemp": ("No temporal organization", 0),
    "mechabl_a005":   ("$\\alpha=0.05$", 2),
    "mechabl_a015":   ("Full \\mbox{AF-TOI} ($\\alpha=0.15$, canonical)", 3),
    "mechabl_a030":   ("$\\alpha=0.30$", 4),
    "mechabl_a050":   ("$\\alpha=0.50$", 5),
    "mechabl_a000":   ("$\\alpha=0$ (uniform window)", 1),  # opcional
}
CANON = "mechabl_a015"
OUTDIR = os.path.dirname(os.path.abspath(__file__))

def read_seed(label, seed):
    p = f"exp_{label}_results_seed{seed}.csv"
    if not os.path.exists(p):
        return None
    fixa, hib = None, None
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Modelo"].strip() != "LSTM-AF-KD":
                continue
            if r["Politica"].strip().startswith("Fixa"):
                fixa = r
            elif r["Politica"].strip().startswith("Hibrida"):
                hib = r
    if fixa is None:
        return None
    g = lambda row, k: float(row[k]) if row and row.get(k) not in (None, "",) else float("nan")
    return dict(seed=seed,
                F1=g(fixa, "F1"), ECE=g(fixa, "ECE"), FR=g(fixa, "FailRate"),
                TTDef=g(fixa, "TTDef"),
                SupPct=g(hib, "SkipPct"), FR_hyb=g(hib, "FailRate"),
                TTDef_hyb=g(hib, "TTDef"))

def wilcoxon_exact(a, b):
    d = [x - y for x, y in zip(a, b) if x != y]
    n = len(d)
    if n == 0:
        return 1.0
    from collections import Counter
    cnt = Counter(sorted(abs(x) for x in d)); pos = {}; start = 1
    for v in sorted(cnt):
        k = cnt[v]; pos[v] = (2 * start + k - 1) / 2; start += k
    W = sum(pos[abs(x)] for x in d if x > 0); vals = [pos[abs(x)] for x in d]
    tot = le = 0
    for signs in itertools.product([0, 1], repeat=n):
        w = sum(v for v, sg in zip(vals, signs) if sg); tot += 1
        if w <= min(W, sum(vals) - W):
            le += 1
    return min(1.0, 2 * le / tot)

def cliffs(a, b):
    return sum((x > y) - (x < y) for x in a for y in b) / (len(a) * len(b))

def mean(v): return sum(v) / len(v)
def sd(v):
    m = mean(v); return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0

METRICS = ["F1", "ECE", "FR", "TTDef", "SupPct", "FR_hyb", "TTDef_hyb"]

per_seed, agg = {}, []
for label, (name, order) in ARMS.items():
    rows = [read_seed(label, s) for s in SEEDS]
    rows = [r for r in rows if r]
    if not rows:
        print(f"[skip] {label}: nenhum exp_{label}_results_seedNN.csv encontrado")
        continue
    per_seed[label] = rows
    a = dict(label=label, arm=name, order=order, n_seeds=len(rows))
    for m in METRICS:
        vals = [r[m] for r in rows if not math.isnan(r[m])]
        if vals:
            a[m + "_mean"], a[m + "_sd"] = mean(vals), sd(vals)
    agg.append(a)
    print(f"[ok] {name}: {len(rows)} seeds")

# CSVs
with open(os.path.join(OUTDIR, "mechanism_ablation_per_seed.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["arm"] + ["seed"] + METRICS)
    for label, rows in per_seed.items():
        for r in rows:
            w.writerow([label, r["seed"]] + [r[m] for m in METRICS])
with open(os.path.join(OUTDIR, "mechanism_ablation_results.csv"), "w", newline="") as f:
    keys = sorted({k for a in agg for k in a})
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(a) for a in agg]

# estatistica pareada vs canonico
stats = {}
if CANON in per_seed:
    canon = {r["seed"]: r for r in per_seed[CANON]}
    for label, rows in per_seed.items():
        if label == CANON:
            continue
        stats[label] = {}
        for m in METRICS:
            pairs = [(canon[r["seed"]][m], r[m]) for r in rows
                     if r["seed"] in canon and not math.isnan(r[m]) and not math.isnan(canon[r["seed"]][m])]
            if len(pairs) < 6:
                continue
            a, b = [p[0] for p in pairs], [p[1] for p in pairs]
            stats[label][m] = {"n": len(pairs),
                               "p_wilcoxon": round(wilcoxon_exact(a, b), 4),
                               "cliffs_delta": round(cliffs(a, b), 3),
                               "mean_canonical": round(mean(a), 4),
                               "mean_arm": round(mean(b), 4)}
    json.dump(stats, open(os.path.join(OUTDIR, "stats.json"), "w"), indent=1)
    print("[ok] stats.json (pareado vs a015 canonico)")
else:
    print("[AVISO] braco mechabl_a015 ausente — rode: python run_mechanism_ablation.py a015")
    print("        (necessario para a comparacao pareada sob o MESMO protocolo)")

# LaTeX
def fmt(a, m, nd=3):
    if m + "_mean" not in a:
        return "---"
    return f"{a[m+'_mean']:.{nd}f}$\\pm${a[m+'_sd']:.{nd}f}"
agg.sort(key=lambda x: x["order"])
lines = [f"{a['arm']} & {fmt(a,'F1')} & {fmt(a,'FR')} & {fmt(a,'ECE')} & {fmt(a,'TTDef')} & {fmt(a,'SupPct',1)} \\\\"
         for a in agg]
open(os.path.join(OUTDIR, "table11_new.tex"), "w").write(
"""% Nova Tabela 11 — mechanism-removal ablation + alpha sensitivity
% 12 seeds (42--53), mean±sd, protocolo run_v29 identico nos 5 bracos.
% F1/FR/ECE/TTDef: politica Fixa; %Sup: politica Hibrida (MHEG).
% Estatistica pareada vs braco canonico em stats.json.
\\begin{tabular}{l c c c c c}
\\toprule
Condition & F1$\\uparrow$ & FR$\\downarrow$ & ECE$\\downarrow$
  & TTD$_{\\mathrm{ef}}$ (s)$\\downarrow$ & \\%Sup (MHEG)$\\uparrow$ \\\\
\\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
print("[done] mechanism_ablation_results.csv, mechanism_ablation_per_seed.csv, table11_new.tex em", OUTDIR)
