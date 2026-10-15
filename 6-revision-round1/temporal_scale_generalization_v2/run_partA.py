#!/usr/bin/env python3
"""PART A: frozen-threshold temporal-scale generalization (shape-preserving warp).
12 seeds canonicas, 9 duracoes, 4 configs. Resumivel: python3 -B run_partA.py [budget_s]
"""
import sys, os, json, time, csv as csvmod
import numpy as np
sys.path.insert(0,'.')
from common import (load_model, CAL, KIN, DT, FRAME_THR, binary_entropy,
                    ttd_metrics_binary, gate_replay, fs_onset_original, ece_frame, f1_episode)
sys.path.insert(0,'../temporal_scale_generalization')
from np_model import fixed_policy_probs
from gen_warped import build_test_set

OUT='duration_only'
SEEDS = list(range(42,54))
DURS  = [2,6,10,14,18,22,31,42,50]
SUPPORT = {2:'extrapolation-fast OOD',6:'ID anchor (abrupt)',10:'ID anchor (abrupt edge)',
           14:'interpolation OOD',18:'interpolation OOD',22:'interpolation OOD',
           31:'ID anchor (progressive)',42:'extrapolation-slow OOD',50:'extrapolation-slow OOD'}

def entropy_phases(P, metas):
    Hs, Hp, Ho = [], [], []
    for i,m in enumerate(metas):
        if m['categoria']!='Critico': continue
        s = m['event_start']; H = binary_entropy(P[i])
        if s-12 > 4: Hs.append(H[4:s-12].mean())
        Hp.append(H[max(0,s-12):s].mean()); Ho.append(H[s:s+12].mean())
    return float(np.mean(Hs)), float(np.mean(Hp)), float(np.mean(Ho))

def main():
    budget = float(sys.argv[1]) if len(sys.argv)>1 else 150.0
    t0=time.time()
    dp = f'{OUT}/done.json'
    done = json.load(open(dp)) if os.path.exists(dp) else {}
    rp = f'{OUT}/frozen_results.csv'
    fields = ['seed','dur_frames','dur_s','support','config','FR','TTD','TTDef','SupPct','ComputeRate',
              'FSonset','FSonsetSimple','F1','ECE','H_stable','H_pre','H_post','n_crit']
    if not os.path.exists(rp):
        with open(rp,'w',newline='') as f: csvmod.DictWriter(f, fieldnames=fields).writeheader()
    models = {(s,t): load_model(s,t) for s in SEEDS for t in ('base','kd')}
    for seed in SEEDS:
        theta, tau_d, tau_h = CAL[seed]
        for d in DURS:
            key=f'{seed}_{d}'
            if key in done: continue
            if time.time()-t0 > budget: print(f'[budget] {len(done)}/108'); return
            X, Y, metas = build_test_set(seed, d)
            yep = np.array([1 if m['categoria']=='Critico' else 0 for m in metas])
            yfr = ((Y > FRAME_THR).astype(np.int64) * yep[:,None])
            onsets = []
            for i,m in enumerate(metas):
                idx = np.where(yfr[i] > 0.1)[0]
                onsets.append(int(idx[0]) if len(idx) else -1)
            rows=[]
            for tag, nfix, nhyb in [('base','Baseline Fixed','Baseline+MHEG'), ('kd','AF-TOI Fixed','AF-TOI+MHEG')]:
                P = fixed_policy_probs(models[(seed,tag)], X)
                Hs,Hp,Ho = entropy_phases(P, metas)
                ttd, fr, ttdef, nc = ttd_metrics_binary(yfr, P, theta)
                rows.append(dict(seed=seed,dur_frames=d,dur_s=round(d*DT,3),support=SUPPORT[d],config=nfix,
                    FR=round(fr,4),TTD=round(ttd,4),TTDef=round(ttdef,4),SupPct=0.0,ComputeRate=1.0,
                    FSonset=0.0,FSonsetSimple=0.0,F1=round(f1_episode(P,yep),4),ECE=round(ece_frame(P,yfr),4),
                    H_stable=round(Hs,4),H_pre=round(Hp,4),H_post=round(Ho,4),n_crit=nc))
                Pg, sup = gate_replay(P, X, tau_d, tau_h)
                ttd, fr, ttdef, _ = ttd_metrics_binary(yfr, Pg, theta)
                fsn, fss = fs_onset_original(sup, Pg, yfr, onsets, theta)
                np.savez_compressed(f'{OUT}/cell_{seed}_{d}_{tag}.npz',
                                    P=P.astype(np.float16), sup=sup, onsets=np.array(onsets))
                rows.append(dict(seed=seed,dur_frames=d,dur_s=round(d*DT,3),support=SUPPORT[d],config=nhyb,
                    FR=round(fr,4),TTD=round(ttd,4),TTDef=round(ttdef,4),SupPct=round(sup.mean()*100,2),
                    ComputeRate=round(1-sup.mean(),4),FSonset=round(fsn,4),FSonsetSimple=round(fss,4),
                    F1=round(f1_episode(Pg,yep),4),ECE=round(ece_frame(Pg,yfr),4),
                    H_stable=round(Hs,4),H_pre=round(Hp,4),H_post=round(Ho,4),n_crit=nc))
            with open(rp,'a',newline='') as f:
                w=csvmod.DictWriter(f,fieldnames=fields); [w.writerow(r) for r in rows]
            done[key]=True; json.dump(done, open(dp,'w'))
            print(f'[ok] seed{seed} d={d} ({len(done)}/108) t={time.time()-t0:.0f}s')
    print('PART A COMPLETA')

if __name__=='__main__':
    main()
