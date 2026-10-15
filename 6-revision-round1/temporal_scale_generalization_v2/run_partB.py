#!/usr/bin/env python3
"""PART B: iso-cost (budget-matched) — protocolo da Tabela 17 / solve_threshold_for_cost.
Target por celula (seed,dur) = compute rate observado do AF-TOI+MHEG (thresholds
congelados) naquela celula. Ajusta tau_h do Baseline por bissecao (tol 2%);
fallback: escala tau_d (com tau_h=1) se o piso cinematico impedir o casamento.
Resumivel: python3 -B run_partB.py [budget_s]
"""
import sys, os, json, time, csv as csvmod
import numpy as np
sys.path.insert(0,'.')
from common import CAL, KIN, DT, FRAME_THR, binary_entropy, ttd_metrics_binary, gate_replay, fs_onset_original, f1_episode
from gen_warped import build_test_set

OUT='iso_cost'; SRC='duration_only'
SEEDS=list(range(42,54)); DURS=[2,6,10,14,18,22,31,42,50]; TOL=0.02

def replay_rate(P, X, tau_d, tau_h):
    out, sup = gate_replay(P, X, tau_d, tau_h)
    return out, sup, 1.0 - sup.mean()

def match_budget(P, X, tau_d0, target):
    # 1) bissecao em tau_h (custo decresce com tau_h)
    lo, hi = 0.0, 1.0; best=None
    for _ in range(14):
        mid = 0.5*(lo+hi)
        out, sup, c = replay_rate(P, X, tau_d0, mid)
        if best is None or abs(c-target) < abs(best[3]-target): best=(tau_d0, mid, (out,sup), c)
        if c > target: lo = mid
        else: hi = mid
        if abs(c-target) <= TOL*target: return best[0], best[1], best[2], best[3], 'tau_h'
    # 2) fallback: tau_h=1 (so cinematica) e escala tau_d
    lo, hi = 0.0, 8.0
    for _ in range(14):
        mid = 0.5*(lo+hi)
        out, sup, c = replay_rate(P, X, tau_d0*mid if mid>0 else 1e-9, 1.0)
        if abs(c-target) < abs(best[3]-target): best=(tau_d0*mid, 1.0, (out,sup), c)
        if c > target: lo = mid
        else: hi = mid
        if abs(c-target) <= TOL*target: return best[0], best[1], best[2], best[3], 'tau_d'
    return best[0], best[1], best[2], best[3], 'best-effort'

def main():
    budget = float(sys.argv[1]) if len(sys.argv)>1 else 150.0
    t0=time.time()
    dp=f'{OUT}/done.json'; done = json.load(open(dp)) if os.path.exists(dp) else {}
    rp=f'{OUT}/iso_cost_results.csv'
    fields=['seed','dur_frames','dur_s','method','compute_rate','target_rate','budget_diff','match_mode',
            'SupPct','FSonset','FSonsetSimple','FR','TTDef','F1','valid']
    if not os.path.exists(rp):
        with open(rp,'w',newline='') as f: csvmod.DictWriter(f,fieldnames=fields).writeheader()
    for seed in SEEDS:
        theta, tau_d, tau_h = CAL[seed]
        for d in DURS:
            key=f'{seed}_{d}'
            if key in done: continue
            if time.time()-t0>budget: print(f'[budget] {len(done)}/108'); return
            X, Y, metas = build_test_set(seed, d)
            yep = np.array([1 if m['categoria']=='Critico' else 0 for m in metas])
            yfr = ((Y > FRAME_THR).astype(np.int64) * yep[:,None])
            onsets=[]
            for i in range(len(metas)):
                idx=np.where(yfr[i]>0.1)[0]; onsets.append(int(idx[0]) if len(idx) else -1)
            Pk = np.load(f'{SRC}/cell_{seed}_{d}_kd.npz')['P'].astype(np.float32)
            Pb = np.load(f'{SRC}/cell_{seed}_{d}_base.npz')['P'].astype(np.float32)
            # AF-TOI+MHEG no ponto congelado (referencia do orcamento)
            outK, supK, rateK = replay_rate(Pk, X, tau_d, tau_h)
            ttd,fr,ttdef,_ = ttd_metrics_binary(yfr, outK, theta)
            fsn,fss = fs_onset_original(supK, outK, yfr, onsets, theta)
            rows=[dict(seed=seed,dur_frames=d,dur_s=round(d*DT,3),method='AF-TOI+MHEG (frozen)',
                 compute_rate=round(rateK,4),target_rate=round(rateK,4),budget_diff=0.0,match_mode='frozen',
                 SupPct=round(supK.mean()*100,2),FSonset=round(fsn,4),FSonsetSimple=round(fss,4),
                 FR=round(fr,4),TTDef=round(ttdef,4),F1=round(f1_episode(outK,yep),4),valid=True)]
            # Baseline+MHEG casado ao orcamento
            td2, th2, (outB,supB), rateB, mode = match_budget(Pb, X, tau_d, rateK)
            ttd,fr,ttdef,_ = ttd_metrics_binary(yfr, outB, theta)
            fsn,fss = fs_onset_original(supB, outB, yfr, onsets, theta)
            rows.append(dict(seed=seed,dur_frames=d,dur_s=round(d*DT,3),method='Baseline+MHEG (matched)',
                 compute_rate=round(rateB,4),target_rate=round(rateK,4),
                 budget_diff=round(abs(rateB-rateK)/max(rateK,1e-9),4),match_mode=mode,
                 SupPct=round(supB.mean()*100,2),FSonset=round(fsn,4),FSonsetSimple=round(fss,4),
                 FR=round(fr,4),TTDef=round(ttdef,4),F1=round(f1_episode(outB,yep),4),
                 valid=bool(abs(rateB-rateK)<=TOL*rateK)))
            with open(rp,'a',newline='') as f:
                w=csvmod.DictWriter(f,fieldnames=fields); [w.writerow(r) for r in rows]
            done[key]=True; json.dump(done, open(dp,'w'))
            print(f'[ok] seed{seed} d={d} rateK={rateK:.3f} rateB={rateB:.3f} ({mode}) ({len(done)}/108)')
    print('PART B COMPLETA')

if __name__=='__main__':
    main()
