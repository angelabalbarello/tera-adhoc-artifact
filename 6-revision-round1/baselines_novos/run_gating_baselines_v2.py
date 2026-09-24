#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gating baselines v2 — DEFINICOES HARMONIZADAS (correcoes da auditoria 12/09).
=============================================================================
Diferencas vs run_gating_baselines.py (originais preservados como estimando
alternativo):
  [A1] marco temporal = ONSET DO ROTULO (1o frame com y_frame_clean > 0.35),
       nao event_start; episodios criticos SEM frame positivo sao EXCLUIDOS
       das metricas de deteccao (mesma regra do protocolo canonico) e
       contados/reportados;
  [A2] gatilho "cinematico" usa SOMENTE os canais KIN=[12,13,14] (regra do
       MHEG), com grid de tau_delta na escala canonica (0.010-0.050;
       tau_delta por seed calibrado fica em 0.015-0.034);
  [A3] safe point e matched-suppression point calculados numa unica passada.
Mantidos deliberadamente: replay sobre trajetorias armazenadas, calibracao
oracle na particao de teste (favorece os baselines), m=3, deadlock dos gates
puramente entropicos (e o achado, nao um bug).
Rodar na pasta Experimentos-Artigo1-AdHoc:  python baselines_novos/run_gating_baselines_v2.py
Saidas: baselines_novos/gating_baselines_v2_results.csv, summary_v2.json
"""
import numpy as np, json, csv
import statistics as st

SEEDS = [42, 43, 44, 45]
DT    = 1.0/9.6
T_EP  = 10.0
M_CONSEC = 3
FAIL_BUDGET = 0.05
W_SMOOTH = 5
KIN = [12, 13, 14]
FRAME_THR = 0.35

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
    d = np.load(f'dados_sinteticos/dataset_sintetico_seed{seed}.npz')
    X = d['X'][te]
    yc = (d['y_frame_clean'] if 'y_frame_clean' in d.files else d['y_frame'])[te]
    # [A1] onset do ROTULO; -1 = critico sem frame positivo (excluido e reportado)
    t0 = np.full(len(te), -1, dtype=int)
    for i in range(len(te)):
        if is_crit[i]:
            pos = np.where(yc[i] > FRAME_THR)[0]
            if len(pos): t0[i] = int(pos[0])
    n_no_onset = int(np.sum(is_crit & (t0 < 0)))
    # [A2] delta cinematico APENAS nos canais KIN
    dX = np.abs(np.diff(X[:, :, KIN], axis=1)).max(axis=2)
    dX = np.concatenate([np.zeros((len(te),1), dtype=np.float32), dX], axis=1)
    return probs_b, probs_k, is_crit, t0, dX, n_no_onset

def replay(probs, dX, method, tau_h, tau_d=None):
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
            if fire: buf = probs[i,t]
            else:    sup[i,t] = True
            out[i,t] = buf
            hist.append(H(buf))
    return out, sup

def metrics(out, sup, is_crit, t0, theta):
    N, T = out.shape
    det_lat, miss, n_used = [], 0, 0
    for i in np.where(is_crit)[0]:
        s = t0[i]
        if s < 0:            # [A1] critico sem onset observavel: excluido
            continue
        n_used += 1
        tr = out[i]; found = None
        for t in range(s, T - M_CONSEC + 1):
            if np.all(tr[t:t+M_CONSEC] >= theta):
                found = t; break
        if found is None: miss += 1
        else: det_lat.append((found - s)*DT)
    fr = miss/max(n_used,1)
    ttd = float(np.mean(det_lat)) if det_lat else float('nan')
    ttdef = (ttd if det_lat else 0.0) + fr*T_EP
    return fr, ttd, ttdef, sup.mean(), n_used

GRID_H  = np.round(np.arange(0.05, 1.0, 0.05), 3)
GRID_TH = np.round(np.arange(0.10, 0.55, 0.05), 3)
GRID_D  = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]   # [A2] escala KIN

def calibrate_safe(probs, dX, method, is_crit, t0):
    grid_d = GRID_D if method == 'mheg' else [None]
    best = None
    for th_h in GRID_H:
        for td in grid_d:
            out, sup = replay(probs, dX, method, th_h, td)
            for theta in GRID_TH:
                fr, ttd, ttdef, s, n = metrics(out, sup, is_crit, t0, theta)
                key = (fr <= FAIL_BUDGET, s, -ttdef)
                if best is None or key > best[0]:
                    best = (key, dict(fr=fr, ttd=ttd, ttdef=ttdef, sup=s, n_crit=n,
                                      tau_h=th_h, tau_d=td, theta=theta,
                                      feasible=fr <= FAIL_BUDGET))
    return best[1]

def matched_point(probs, dX, method, is_crit, t0, target_sup):
    best = None
    for th_h in np.round(np.arange(0.05, 1.0, 0.02), 3):
        out, sup = replay(probs, dX, method, th_h)
        s = sup.mean()
        mbest = None
        for theta in GRID_TH:
            fr, ttd, ttdef, _, n = metrics(out, sup, is_crit, t0, theta)
            if mbest is None or (fr, ttdef) < (mbest[0], mbest[1]):
                mbest = (fr, ttdef, theta, n)
        if best is None or abs(s - target_sup) < best[0]:
            best = (abs(s - target_sup), s, mbest, th_h)
    return dict(sup=best[1], fr=best[2][0], ttdef=best[2][1],
                theta=best[2][2], n_crit=best[2][3], tau_h=best[3])

rows_out = []
for seed in SEEDS:
    pb, pk, is_crit, t0, dX, n_no = load_seed(seed)
    print(f"[seed{seed}] criticos sem onset de rotulo (excluidos): {n_no}")
    mheg_sup = {}
    for tag, probs in [('Baseline', pb), ('AF-TOI', pk)]:
        for method, name in [('conf','Confidence-threshold gate'),
                             ('smooth', f'Smoothed-entropy gate (w={W_SMOOTH})'),
                             ('mheg','Entropy+kinematic gate (MHEG-style)')]:
            r = calibrate_safe(probs, dX, method, is_crit, t0)
            r.update(seed=seed, traj=tag, method=name, point='safe')
            rows_out.append(dict(r))
            if method == 'mheg': mheg_sup[tag] = r['sup']
            print(f"seed{seed} {tag:9s} {name:38s} SAFE FR={r['fr']:.3f} TTDef={r['ttdef']:.3f}s "
                  f"Sup={r['sup']*100:5.1f}% n={r['n_crit']} feas={r['feasible']}")
    for tag, probs in [('Baseline', pb), ('AF-TOI', pk)]:
        for method, name in [('conf','Confidence-threshold gate'),
                             ('smooth', f'Smoothed-entropy gate (w={W_SMOOTH})')]:
            m = matched_point(probs, dX, method, is_crit, t0, mheg_sup[tag])
            m.update(seed=seed, traj=tag, method=name, point='matched', feasible=m['fr']<=FAIL_BUDGET)
            rows_out.append(dict(m))
            print(f"seed{seed} {tag:9s} {name:38s} MATCH FR={m['fr']:.3f} TTDef={m['ttdef']:.3f}s "
                  f"Sup={m['sup']*100:5.1f}% (alvo {mheg_sup[tag]*100:.1f}%)")

keys = sorted({k for r in rows_out for k in r})
with open('baselines_novos/gating_baselines_v2_results.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows_out)

print('\n===== MEAN +- SD (4 seeds) =====')
summary = {}
for point in ['safe','matched']:
    for tag in ['Baseline','AF-TOI']:
        for name in sorted({r['method'] for r in rows_out if r['point']==point}):
            sel = [r for r in rows_out if r['traj']==tag and r['method']==name and r['point']==point]
            if not sel: continue
            fr=[r['fr'] for r in sel]; te=[r['ttdef'] for r in sel]; s=[r['sup']*100 for r in sel]
            summary[f"{point}|{tag}|{name}"] = dict(
                fr=st.mean(fr), fr_sd=st.stdev(fr), ttdef=st.mean(te), ttdef_sd=st.stdev(te),
                sup=st.mean(s), sup_sd=st.stdev(s))
            print(f"{point:7s} {tag:9s} {name:38s} FR={st.mean(fr):.3f}+-{st.stdev(fr):.3f} "
                  f"TTDef={st.mean(te):.3f}+-{st.stdev(te):.3f} Sup={st.mean(s):.1f}+-{st.stdev(s):.1f}%")
json.dump(summary, open('baselines_novos/summary_v2.json','w'), indent=1)
print('[done] gating_baselines_v2_results.csv, summary_v2.json')
