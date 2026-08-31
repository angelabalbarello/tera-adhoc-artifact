#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gating baselines on stored full-inference trajectories (replay protocol).
Baselines requested by Reviewer 3 (item 5):
  A) confidence-threshold gate (no kinematic channel)
  B) smoothed-entropy gate (moving average, w=5; no kinematic channel)
  C) entropy+kinematic gate (MHEG-style reference under the same replay)
Each simulated on Baseline (unorganized) and AF-TOI (organized) trajectories,
seeds 42-45. Thresholds oracle-calibrated on the test partition (max %Sup
s.t. FR<=0.05) -- an upper bound that FAVORS the baselines.
Replay approximation: on update, the buffered probability is refreshed from
the stored full-inference trajectory (ignores hidden-state drift; favorable
to all gated methods equally).
"""
import numpy as np, json, csv, io, sys

SEEDS = [42, 43, 44, 45]
DT    = 1.0/9.6
T_EP  = 10.0
M_CONSEC = 3
FAIL_BUDGET = 0.05
W_SMOOTH = 5

def H(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return -(p*np.log2(p) + (1-p)*np.log2(1-p))

def load_seed(seed):
    probs_b = np.load(f'Inferencia/exp_abl_A_frame_probs_base_seed{seed}.npy')
    probs_k = np.load(f'Inferencia/exp_abl_A_frame_probs_kd_seed{seed}.npy')
    sp = json.load(open(f'dados_sinteticos/splits_seed{seed}.json'))
    te = np.array(sp['test_idx'])
    rows = list(csv.DictReader(open(f'dados_sinteticos/episodios_seed{seed}.csv'), delimiter=';'))
    is_crit = np.array([r['class_name'].strip()=='Critico' for r in rows])[te]
    t0 = np.array([int(r['event_start']) for r in rows])[te]
    X = np.load(f'dados_sinteticos/dataset_sintetico_seed{seed}.npz')['X'][te]  # (N,96,31)
    dX = np.abs(np.diff(X, axis=1)).max(axis=2)          # (N,95) max over channels
    dX = np.concatenate([np.zeros((len(te),1),dtype=np.float32), dX], axis=1)
    return probs_b, probs_k, is_crit, t0, dX

def replay(probs, dX, method, tau_h, tau_d=None):
    """Simulate gate; returns emitted trajectory and suppression mask."""
    N, T = probs.shape
    out = np.empty_like(probs); sup = np.zeros((N,T), dtype=bool)
    for i in range(N):
        buf = probs[i,0]; out[i,0] = buf
        hist = [H(buf)]
        for t in range(1, T):
            if method == 'conf':
                fire = H(buf) >= tau_h
            elif method == 'smooth':
                w = hist[-W_SMOOTH:]
                fire = (sum(w)/len(w)) >= tau_h
            elif method == 'mheg':
                fire = (H(buf) >= tau_h) or (dX[i,t] >= tau_d)
            if fire:
                buf = probs[i,t]
            else:
                sup[i,t] = True
            out[i,t] = buf
            hist.append(H(buf))
    return out, sup

def metrics(out, sup, is_crit, t0, theta):
    N, T = out.shape
    det_lat, miss = [], 0
    crit = np.where(is_crit)[0]
    for i in crit:
        s = t0[i]
        tr = out[i]
        found = None
        for t in range(max(s,0), T - M_CONSEC + 1):
            if np.all(tr[t:t+M_CONSEC] >= theta):
                found = t; break
        if found is None: miss += 1
        else: det_lat.append((found - s)*DT)
    fr = miss/len(crit)
    ttd = float(np.mean(det_lat)) if det_lat else float('nan')
    ttdef = (ttd if det_lat else 0.0) + fr*T_EP
    return fr, ttd, ttdef, sup.mean()

def calibrate(probs, dX, method, is_crit, t0):
    grid_h = np.round(np.arange(0.05, 1.0, 0.05), 3)
    grid_th = np.round(np.arange(0.10, 0.55, 0.05), 3)
    grid_d = [0.5, 1.0, 1.5, 2.0] if method=='mheg' else [None]
    best = None
    for th_h in grid_h:
        for td in grid_d:
            out, sup = replay(probs, dX, method, th_h, td)
            for theta in grid_th:
                fr, ttd, ttdef, s = metrics(out, sup, is_crit, t0, theta)
                ok = fr <= FAIL_BUDGET
                key = (ok, s, -ttdef)  # prefer feasible, then max sup, then min ttdef
                if best is None or key > best[0]:
                    best = (key, dict(fr=fr, ttd=ttd, ttdef=ttdef, sup=s,
                                      tau_h=th_h, tau_d=td, theta=theta, feasible=ok))
    return best[1]

rows_out = []
for seed in SEEDS:
    pb, pk, is_crit, t0, dX = load_seed(seed)
    for tag, probs in [('Baseline', pb), ('AF-TOI', pk)]:
        for method, name in [('conf','Confidence-threshold gate'),
                             ('smooth', f'Smoothed-entropy gate (w={W_SMOOTH})'),
                             ('mheg','Entropy+kinematic gate (MHEG-style)')]:
            r = calibrate(probs, dX, method, is_crit, t0)
            r.update(seed=seed, traj=tag, method=name)
            rows_out.append(r)
            print(f"seed{seed} {tag:9s} {name:38s} FR={r['fr']:.3f} TTDef={r['ttdef']:.3f}s Sup={r['sup']*100:5.1f}% feas={r['feasible']}")

import statistics as st
with open('baselines_novos/gating_baselines_results.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)

print('\n===== MEAN +- SD over seeds =====')
summary = {}
for tag in ['Baseline','AF-TOI']:
    for name in sorted(set(r['method'] for r in rows_out)):
        sel = [r for r in rows_out if r['traj']==tag and r['method']==name]
        fr  = [r['fr'] for r in sel]; te = [r['ttdef'] for r in sel]; s=[r['sup']*100 for r in sel]
        summary[(tag,name)] = (st.mean(fr), st.stdev(fr), st.mean(te), st.stdev(te), st.mean(s), st.stdev(s))
        print(f"{tag:9s} {name:38s} FR={st.mean(fr):.3f}+-{st.stdev(fr):.3f}  TTDef={st.mean(te):.3f}+-{st.stdev(te):.3f}  Sup={st.mean(s):.1f}+-{st.stdev(s):.1f}%")
json.dump({f"{k[0]}|{k[1]}": v for k,v in summary.items()}, open('baselines_novos/summary.json','w'), indent=1)
