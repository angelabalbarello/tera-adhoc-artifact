#!/usr/bin/env python3
"""A8-bis: reproduzir os numeros publicados da campanha exp_seed12_round1
nos test sets ORIGINAIS, com a maquina numpy + limiares congelados. Resumivel."""
import sys, csv, json, time, os
sys.path.insert(0,'.')
import numpy as np
from common import *

pub = {}
for r in csv.DictReader(open('/tmp/tera-repo/6-revision-round1/seed-expansion-tera/expanded_per_seed.csv')):
    pub[(int(r['Seed']), r['config_id'])] = r

t0=time.time(); budget = float(sys.argv[1]) if len(sys.argv)>1 else 150
done = json.load(open('audit/anchor_done_v2.json')) if os.path.exists('audit/anchor_done_v2.json') else {}
outp = 'audit/anchor_validation_v2.csv'
if not os.path.exists(outp):
    open(outp,'w').write('seed,config,FR_ours,FR_pub,TTDef_ours,TTDef_pub,Skip_ours,Skip_pub\n')
for seed in SEEDS:
    if str(seed) in done: continue
    if time.time()-t0 > budget: print(f'[budget] {len(done)}/12'); sys.exit(0)
    te, X, yfr, yep, ycont, prog = campaign_test_split(seed)
    theta, tau_d, tau_h = CAL[seed]
    lines=[]
    for tag, cfix, chyb in [('base','baseline_fixed','baseline_hybrid'), ('kd','aftkd_fixed','aftkd_hybrid')]:
        m = load_model(seed, tag)
        P = fixed_policy_probs(m, X)
        ttd, fr, ttdef, _ = ttd_metrics_binary(yfr, P, theta)
        pr = pub[(seed,cfix)]
        lines.append(f"{seed},{cfix},{fr:.4f},{pr['FailRate']},{ttdef:.4f},{pr['TTDef']},0.0,{pr['SkipPct']}")
        Pg, sup = gate_replay(P, X, tau_d, tau_h)
        ttd, fr, ttdef, _ = ttd_metrics_binary(yfr, Pg, theta)
        ph = pub[(seed,chyb)]
        lines.append(f"{seed},{chyb},{fr:.4f},{ph['FailRate']},{ttdef:.4f},{ph['TTDef']},{sup.mean()*100:.2f},{ph['SkipPct']}")
    open(outp,'a').write('\n'.join(lines)+'\n')
    done[str(seed)]=True; json.dump(done, open('audit/anchor_done_v2.json','w'))
    print(f'[ok] seed{seed} ({len(done)}/12) t={time.time()-t0:.0f}s')
print('ANCHOR VALIDATION COMPLETA')
