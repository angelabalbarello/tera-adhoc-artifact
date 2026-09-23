#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Periodic-refresh + confidence gate (auditoria 23/09, resposta ao ponto
"o gate de confianca puro trava por construcao / espantalho").
Mesmo protocolo harmonizado do run_gating_baselines_v2.py (marco = onset do
rotulo, m=3, theta grid, calibracao oracle no teste, replay sobre trajetorias
armazenadas). Novo metodo: update se (t % R == 0) OU H(buffer) >= tau_h,
com R em {2,4,8}. O refresh quebra o deadlock por construcao.
Saida: baselines_novos/refresh_gate_v2_results.csv (+ resumo no stdout)
"""
import numpy as np, json, csv
import statistics as st

SEEDS=[42,43,44,45]; DT=1.0/9.6; T_EP=10.0; M_CONSEC=3; FAIL_BUDGET=0.05
KIN=[12,13,14]; FRAME_THR=0.35
GRID_H=np.round(np.arange(0.05,1.0,0.05),3)
GRID_TH=np.round(np.arange(0.10,0.55,0.05),3)

def H(p):
    p=np.clip(p,1e-6,1-1e-6); return -(p*np.log2(p)+(1-p)*np.log2(1-p))

def load_seed(seed):
    pb=np.load(f'Inferencia/exp_abl_A_frame_probs_base_seed{seed}.npy')
    pk=np.load(f'Inferencia/exp_abl_A_frame_probs_kd_seed{seed}.npy')
    sp=json.load(open(f'dados_sinteticos/splits_seed{seed}.json')); te=np.array(sp['test_idx'])
    rows=list(csv.DictReader(open(f'dados_sinteticos/episodios_seed{seed}.csv'),delimiter=';'))
    is_crit=np.array([r['class_name'].strip()=='Critico' for r in rows])[te]
    d=np.load(f'dados_sinteticos/dataset_sintetico_seed{seed}.npz')
    yc=(d['y_frame_clean'] if 'y_frame_clean' in d.files else d['y_frame'])[te]
    t0=np.full(len(te),-1,dtype=int)
    for i in range(len(te)):
        if is_crit[i]:
            pos=np.where(yc[i]>FRAME_THR)[0]
            if len(pos): t0[i]=int(pos[0])
    return pb,pk,is_crit,t0

def replay_refresh(probs,R,tau_h):
    N,T=probs.shape; out=np.empty_like(probs); sup=np.zeros((N,T),bool)
    for i in range(N):
        buf=probs[i,0]; out[i,0]=buf
        for t in range(1,T):
            if (t % R == 0) or (H(buf)>=tau_h): buf=probs[i,t]
            else: sup[i,t]=True
            out[i,t]=buf
    return out,sup

def metrics(out,sup,is_crit,t0,theta):
    N,T=out.shape; det,miss,n=[ ],0,0
    for i in np.where(is_crit)[0]:
        s=t0[i]
        if s<0: continue
        n+=1; tr=out[i]; found=None
        for t in range(s,T-M_CONSEC+1):
            if np.all(tr[t:t+M_CONSEC]>=theta): found=t; break
        if found is None: miss+=1
        else: det.append((found-s)*DT)
    fr=miss/max(n,1); ttd=float(np.mean(det)) if det else float('nan')
    return fr, (ttd if det else 0.0)+fr*T_EP, sup.mean(), n

rows_out=[]
for seed in SEEDS:
    pb,pk,is_crit,t0=load_seed(seed)
    for tag,probs in [('Baseline',pb),('AF-TOI',pk)]:
        for R in [2,4,8]:
            best=None
            for th_h in GRID_H:
                out,sup=replay_refresh(probs,R,th_h)
                for theta in GRID_TH:
                    fr,ttdef,s,n=metrics(out,sup,is_crit,t0,theta)
                    key=(fr<=FAIL_BUDGET,s,-ttdef)
                    if best is None or key>best[0]:
                        best=(key,dict(seed=seed,traj=tag,R=R,fr=fr,ttdef=ttdef,sup=s,
                                       n_crit=n,tau_h=th_h,theta=theta,
                                       feasible=fr<=FAIL_BUDGET))
            r=best[1]; rows_out.append(r)
            print(f"seed{seed} {tag:9s} refresh R={R}: SAFE FR={r['fr']:.3f} "
                  f"TTDef={r['ttdef']:.3f}s Sup={r['sup']*100:5.1f}% feas={r['feasible']}")
keys=sorted({k for r in rows_out for k in r})
with open('baselines_novos/refresh_gate_v2_results.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows_out)
print('\n===== MEAN +- SD (4 seeds) =====')
for tag in ['Baseline','AF-TOI']:
    for R in [2,4,8]:
        sel=[r for r in rows_out if r['traj']==tag and r['R']==R]
        fr=[r['fr'] for r in sel]; te=[r['ttdef'] for r in sel]; s=[r['sup']*100 for r in sel]
        print(f"{tag:9s} R={R}: FR={st.mean(fr):.3f}+-{st.stdev(fr):.3f} "
              f"TTDef={st.mean(te):.3f}+-{st.stdev(te):.3f} Sup={st.mean(s):.1f}+-{st.stdev(s):.1f}%")
