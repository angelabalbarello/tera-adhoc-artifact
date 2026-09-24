#!/usr/bin/env python3
"""Recompute the FR Wilcoxon contrast (Table 9) and the mechanism-ablation
alpha-grid contrasts (Table 11) directly from the per-seed CSVs.

Outputs: results/audit_fr_wilcoxon.csv, results/audit_fr_wilcoxon_summary.txt,
results/ablation_vs_alpha_grid.txt
"""
import pandas as pd, numpy as np, pathlib
from scipy import stats
HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent / 'results'; RES.mkdir(exist_ok=True)
SEEDEXP = HERE.parent.parent / 'seed-expansion-tera' / 'expanded_per_seed.csv'
MECH = HERE.parent.parent.parent.parent / 'Experimentos-Artigo1-AdHoc' / 'ablacao_mecanismo_temporal' / 'mechanism_ablation_per_seed.csv'

def cliff(x, y):
    gt = sum((a > b) for a in x for b in y); lt = sum((a < b) for a in x for b in y)
    return (gt - lt) / (len(x) * len(y))

df = pd.read_csv(SEEDEXP)
b = df[df.config_id == 'baseline_fixed'].set_index('Seed').FailRate
a = df[df.config_id == 'aftkd_fixed'].set_index('Seed').FailRate
pd.DataFrame({'seed': b.index, 'FR_baseline_fixed': b.values,
              'FR_aftoi_fixed': a.values, 'difference': (b - a).values}
             ).to_csv(RES / 'audit_fr_wilcoxon.csv', index=False)
d = (b - a).values
lines = [f'zeros={int((d==0).sum())} pos={int((d>0).sum())} neg={int((d<0).sum())} n_eff={int((d!=0).sum())}']
for mode in ['wilcox', 'pratt', 'zsplit']:
    r = stats.wilcoxon(b.values, a.values, zero_method=mode, alternative='two-sided',
                       mode='exact' if mode != 'pratt' else 'auto')
    lines.append(f'{mode}: W={r.statistic}, p={r.pvalue:.6f}')
lines += [f'Baseline seeds FR=0: {int((b.values==0).sum())}/12 (FR>0: {sorted(b.index[b.values>0].tolist())})',
          f'AF-TOI seeds FR=0: {int((a.values==0).sum())}/12',
          f'Cliff delta = {cliff(b.values, a.values):+.4f}']
(RES / 'audit_fr_wilcoxon_summary.txt').write_text('\n'.join(lines))
print('\n'.join(lines), '\n')

m = pd.read_csv(MECH)
alphas = ['mechabl_a005', 'mechabl_a015', 'mechabl_a030', 'mechabl_a050']
out = []
for col, tag in [('TTDef', 'fixed'), ('TTDef_hyb', 'hybrid'), ('ECE', 'ECE'), ('SupPct', 'SupPct')]:
    piv = m.pivot(index='seed', columns='arm', values=col)
    nt = piv['mechabl_notemp']
    worst = piv[alphas].mean().idxmax() if col != 'ECE' else piv[alphas].mean().idxmax()
    for label, ref in [('canonical a015', piv['mechabl_a015']),
                       (f'worst-alpha {worst}', piv[worst]),
                       ('grid mean', piv[alphas].mean(axis=1))]:
        r = stats.wilcoxon(nt.values, ref.values, zero_method='wilcox',
                           alternative='two-sided', mode='exact')
        out.append(f'{col} NoTemp={nt.mean():.3f} vs {label}={ref.mean():.3f} '
                   f'ratio={nt.mean()/max(ref.mean(),1e-9):.2f}x p={r.pvalue:.4f} '
                   f'delta={cliff(nt.values, ref.values):+.3f}')
(RES / 'ablation_vs_alpha_grid.txt').write_text('\n'.join(out))
print('\n'.join(out))
