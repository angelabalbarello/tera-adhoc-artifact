#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidacao 12 seeds + estatistica (pos run_seed_expansion.py).
Le exp_abl_A_results_seed{42..53}.csv (na pasta corrente = FGCS/Inferencia),
agrega as 4 configuracoes do estudo fatorial e emite:
  - expanded_summary.csv                (media +- sd por configuracao)
  - expanded_paper_table2_rows.tex      (linhas prontas para a Tabela 8)
  - expanded_stats_report.txt           (Wilcoxon pareado n=12 + Cliff's delta)
"""
import csv, math, statistics as st
from pathlib import Path
from itertools import combinations

SEEDS = list(range(42, 54))
rows = []
missing = []
for s in SEEDS:
    p = Path(f"exp_abl_A_results_seed{s}.csv")
    if not p.exists():
        missing.append(s); continue
    rows += list(csv.DictReader(open(p)))
if missing:
    print(f"AVISO: seeds sem CSV: {missing} (prossegue com as demais)")

def cfg(r):
    m = 'AF-TOI' if 'AF' in r['Modelo'] else 'Baseline'
    pol = 'MHEG' if r['Politica'].startswith('Hibrida') else 'Fixed'
    return f"{m} {pol}"

def fnum(x):
    try: return float(x)
    except: return float('nan')

configs = {}
for r in rows:
    configs.setdefault(cfg(r), {}).setdefault(int(r['Seed']), r)

METRICS = ['F1','ECE','FailRate','TTD_Progressive','TTD','TTDef','Cost_ms_per_frame','SkipPct']

def agg(c, m):
    vals = [fnum(configs[c][s][m]) for s in sorted(configs[c]) if not math.isnan(fnum(configs[c][s][m]))]
    return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0, len(vals))

# ---------- summary csv ----------
with open('expanded_summary.csv','w',newline='') as f:
    w = csv.writer(f); w.writerow(['Config','n_seeds'] + [x for m in METRICS for x in (m+'_mean', m+'_sd')])
    for c in sorted(configs):
        line = [c, len(configs[c])]
        for m in METRICS:
            mu, sd, _ = agg(c, m); line += [f"{mu:.4f}", f"{sd:.4f}"]
        w.writerow(line)

# ---------- LaTeX rows (formato paper_table2_rows.tex) ----------
def cell(c, m, bold=False):
    mu, sd, _ = agg(c, m)
    t = f"{mu:.3f}\\pm{sd:.3f}"
    return f"$\\mathbf{{{t}}}$" if bold else f"${t}$"

order = [('Baseline Fixed','LSTM-Baseline','No',False),
         ('Baseline MHEG','LSTM-Baseline','Yes',False),
         ('AF-TOI Fixed','LSTM-AF-TOI','No',False),
         ('AF-TOI MHEG','LSTM-AF-TOI','\\textbf{Yes}',True)]
with open('expanded_paper_table2_rows.tex','w') as f:
    for c, name, mheg, bold in order:
        if c not in configs: continue
        cells = [cell(c,m,bold) for m in ['F1','ECE','FailRate','TTD_Progressive','TTD','TTDef','Cost_ms_per_frame','SkipPct']]
        if 'MHEG' in c: cells[4] = '{---}' if not bold else cells[4]
        f.write(f"{name} & {mheg} & " + " & ".join(cells) + " \\\\\n")

# ---------- Wilcoxon pareado + Cliff ----------
def wilcoxon_exact(a, b):
    d = [x-y for x,y in zip(a,b) if x != y]
    n = len(d)
    if n == 0: return 1.0, 0
    ranks = sorted(range(n), key=lambda i: abs(d[i]))
    rk = [0.0]*n
    i = 0
    sabs = sorted(abs(x) for x in d)
    # average ranks for ties
    from collections import Counter
    cnt = Counter(sabs); pos = {}
    start = 1
    for v in sorted(cnt):
        k = cnt[v]; pos[v] = (2*start + k - 1)/2; start += k
    Wp = sum(pos[abs(x)] for x in d if x > 0)
    # exact two-sided p by enumeration (n<=20 ok para n=12)
    tot = 0; le = 0
    vals = [pos[abs(x)] for x in d]
    from itertools import product
    for signs in product([0,1], repeat=n):
        w = sum(v for v,s in zip(vals,signs) if s)
        tot += 1
        if w <= min(Wp, sum(vals)-Wp): le += 1
    p = min(1.0, 2*le/tot)
    return p, Wp

def cliffs(a, b):
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a)*len(b))

def paired(c1, c2, m):
    common = sorted(set(configs[c1]) & set(configs[c2]))
    a = [fnum(configs[c1][s][m]) for s in common]
    b = [fnum(configs[c2][s][m]) for s in common]
    try:
        from scipy.stats import wilcoxon as sw
        p = sw(a, b, zero_method='wilcox', alternative='two-sided', method='exact').pvalue
    except Exception:
        p, _ = wilcoxon_exact(a, b)
    return p, cliffs(a, b), len(common)

with open('expanded_stats_report.txt','w') as f:
    f.write("Estatistica pareada por seed (n = seeds em comum)\n")
    f.write("="*60 + "\n")
    pairs = [('Baseline Fixed','AF-TOI Fixed'), ('Baseline MHEG','AF-TOI MHEG'),
             ('AF-TOI Fixed','AF-TOI MHEG'), ('Baseline Fixed','Baseline MHEG')]
    for c1, c2 in pairs:
        if c1 not in configs or c2 not in configs: continue
        f.write(f"\n{c1}  vs  {c2}\n")
        for m in ['F1','ECE','FailRate','TTDef','Cost_ms_per_frame']:
            p, d, n = paired(c1, c2, m)
            sig = ' *' if p < 0.05 else ''
            f.write(f"  {m:18s} p={p:.4f}{sig}   Cliff's delta={d:+.3f}   (n={n})\n")
    f.write("\n* p < 0.05 (Wilcoxon signed-rank, bicaudal, exato)\n")

print(open('expanded_stats_report.txt').read())
print("Gerados: expanded_summary.csv, expanded_paper_table2_rows.tex, expanded_stats_report.txt")
