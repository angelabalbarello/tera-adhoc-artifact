#!/usr/bin/env python3
"""Matched-suppression operating point for conf/smooth gates:
tune tau_h so %Sup ~= the safe %Sup achieved by the entropy+kinematic gate
on the same trajectories, then report FR / TTDef at that point."""
import numpy as np, json, csv, statistics as st
exec(open('baselines_novos/run_gating_baselines.py').read().split("rows_out = []")[0])  # reuse defs

targets = {}  # (seed,traj) -> mheg sup
prev = list(csv.DictReader(open('baselines_novos/gating_baselines_results.csv')))
for r in prev:
    if 'MHEG' in r['method']:
        targets[(int(r['seed']), r['traj'])] = float(r['sup'])

rows = []
for seed in SEEDS:
    pb, pk, is_crit, t0, dX = load_seed(seed)
    for tag, probs in [('Baseline', pb), ('AF-TOI', pk)]:
        tgt = targets[(seed, tag)]
        for method in ['conf', 'smooth']:
            best = None
            for th_h in np.round(np.arange(0.05, 1.0, 0.02), 3):
                out, sup = replay(probs, dX, method, th_h)
                s = sup.mean()
                # best theta for detection at this gate setting
                mbest = None
                for theta in np.round(np.arange(0.10, 0.55, 0.05), 3):
                    fr, ttd, ttdef, _ = metrics(out, sup, is_crit, t0, theta)
                    if mbest is None or (fr, ttdef) < (mbest[0], mbest[1]):
                        mbest = (fr, ttdef)
                key = abs(s - tgt)
                if best is None or key < best[0]:
                    best = (key, s, mbest[0], mbest[1], th_h)
            rows.append(dict(seed=seed, traj=tag, method=method,
                             sup=best[1], fr=best[2], ttdef=best[3], tau=best[4]))
            print(f"seed{seed} {tag:9s} {method:7s} matched-sup={best[1]*100:5.1f}% (target {tgt*100:.1f}%) FR={best[2]:.3f} TTDef={best[3]:.3f}")

json.dump(rows, open('baselines_novos/matched_sup.json','w'), indent=1, default=float)
print('\n== mean over seeds ==')
for tag in ['Baseline','AF-TOI']:
    for m in ['conf','smooth']:
        sel = [r for r in rows if r['traj']==tag and r['method']==m]
        print(f"{tag:9s} {m:7s} Sup={st.mean([r['sup'] for r in sel])*100:.1f}+-{st.stdev([r['sup'] for r in sel])*100:.1f}%  FR={st.mean([r['fr'] for r in sel]):.3f}+-{st.stdev([r['fr'] for r in sel]):.3f}  TTDef={st.mean([r['ttdef'] for r in sel]):.3f}+-{st.stdev([r['ttdef'] for r in sel]):.3f}")
