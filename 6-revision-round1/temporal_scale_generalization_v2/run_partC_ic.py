#!/usr/bin/env python3
"""PART C: Baseline+IC+MHEG no protocolo de warp temporal (reavaliacao, sem retreino).

Atende a recomendacao 6 do documento Angela / E.3.3 do parecer externo: testar o
Baseline+IC+MHEG (Tab. 14) no protocolo de warp da §4.12, com tudo congelado.

Procedimento (identico ao canonico tera_posthoc_calibration_experiment.py):
  1. IsotonicCalibrator.fit nas frame_probs_val do Baseline (arquivadas) + y_fr_val;
  2. limiares (tau_d, tau_h) re-calibrados na validacao via
     calibrate_gating_for_calibrated_model (funcao canonica importada);
  3. ANCORA: avaliacao no teste NATIVO (P windowed numpy -> iso -> gate_replay)
     comparada aos macros publicados (TTDef 0.0527, FR 0, Skip 57.11%);
  4. warp: para cada duracao da grade (DURS de run_partA), P_raw do Baseline
     sobre o conjunto warpado (fixed_policy_probs congelado), iso frame a frame,
     gate_replay com os limiares IC, mesmas metricas de run_partA.
Seeds 42-45 (escopo da campanha de calibracao). Saida: ic_warp_results.csv.
Pre-declarado: resultado reportado seja favoravel ou nao; nenhum limiar ajustado
apos ver resultados de teste/warp.
"""
import os, sys, csv, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
ART = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ART, '3-calibration-posthoc'))
sys.path.insert(0, os.path.join(ART, '1-pipeline-tera'))

# indices EXATOS da campanha p/ seeds 42-45 (mesma convencao do run_partA)
if not os.path.exists('/tmp/tera-repo'):
    os.symlink(ART, '/tmp/tera-repo')

from common import (SEEDS, KIN, DT, FRAME_THR, CAL, load_model, gate_replay,
                    ttd_metrics_binary, fs_onset_original, f1_episode, ece_frame)
sys.path.insert(0, '/sessions/zealous-lucid-ride/mnt/TreinamentoNovo/Experimentos-Artigo1-AdHoc/temporal_scale_generalization')
from np_model import fixed_policy_probs
from gen_warped import build_test_set
from tera_posthoc_calibration_experiment import (
    IsotonicCalibrator, calibrate_gating_for_calibrated_model)

R = os.path.join(ART, '1-pipeline-tera', 'results', 'exp_seed12_round1')
DURS = [2, 6, 10, 14, 18, 22, 31, 42, 50]
IC_SEEDS = [42, 43, 44, 45]

def ttdef_val_fixed(P, y_fr, theta):
    ttd, fr, ttdef, _ = ttd_metrics_binary((y_fr > FRAME_THR).astype(np.int64)
                                           if y_fr.dtype != np.int64 else y_fr,
                                           P, theta)
    return ttdef

def main():
    out = os.path.join(HERE, 'ic_warp_results.csv')
    fields = ['seed','dur_frames','dur_s','config','FR','TTD','TTDef','SupPct',
              'FSonset','FSonsetSimple','F1','ECE','tau_d','tau_h','n_crit']
    if not os.path.exists(out):
        with open(out,'w',newline='') as f: csv.DictWriter(f, fieldnames=fields).writeheader()
    anchor_rows = []
    for seed in IC_SEEDS:
        theta, _, _ = CAL[seed]
        # 1-2) fit + limiares na VALIDACAO (insumos arquivados)
        pv  = np.load(f'{R}/metrics/frame_probs_val_baseline_seed{seed}.npy').astype(np.float32)
        Xv  = np.load(f'{R}/data/X_va_seed{seed}.npy').astype(np.float32)
        yfv = np.load(f'{R}/data/y_fr_va_seed{seed}.npy').astype(np.float32)
        yev = np.load(f'{R}/data/y_ep_va_seed{seed}.npy').astype(int)
        cache_f = os.path.join(HERE, f'ic_cache_seed{seed}.json')
        ic  = IsotonicCalibrator().fit(pv, yfv)
        pcv = ic.transform(pv)
        yfv_bin = (yfv > FRAME_THR).astype(np.int64) * yev[:, None]
        ref = ttdef_val_fixed(pcv, yfv_bin, theta)
        if os.path.exists(cache_f):
            cc = json.load(open(cache_f)); tau_d, tau_h = cc['tau_d'], cc['tau_h']
        else:
            tau_d, tau_h = calibrate_gating_for_calibrated_model(
                pcv, Xv, yfv_bin, yev, KIN, theta, ref, cal_cfg={})
            json.dump(dict(tau_d=tau_d, tau_h=tau_h), open(cache_f, 'w'))
        print(f'[seed {seed}] IC thresholds: tau_d={tau_d:.6f} tau_h={tau_h:.3f} refTTDef={ref:.4f}', flush=True)
        # 3) ANCORA nativa (cacheada)
        anc_f = os.path.join(HERE, f'ic_anchor_seed{seed}.json')
        Xt  = np.load(f'{R}/data/X_te_seed{seed}.npy').astype(np.float32)
        yft = np.load(f'{R}/data/y_fr_te_seed{seed}.npy').astype(np.float32)
        yet = np.load(f'{R}/data/y_ep_te_seed{seed}.npy').astype(int)
        yft_bin = (yft > FRAME_THR).astype(np.int64) * yet[:, None]
        m = load_model(seed, 'base')
        if os.path.exists(anc_f):
            ar = json.load(open(anc_f)); anchor_rows.append(ar)
            print(f'[seed {seed}] ancora cacheada: {ar}', flush=True)
        else:
            Praw = fixed_policy_probs(m, Xt)
            Pcal = ic.transform(Praw)
            Pg, sup = gate_replay(Pcal, Xt, tau_d, tau_h)
            ttd, fr, ttdef, nc = ttd_metrics_binary(yft_bin, Pg, theta)
            ar = dict(seed=seed, FR=fr, TTDef=round(ttdef,4), Sup=round(100*sup.mean(),2))
            json.dump(ar, open(anc_f, 'w')); anchor_rows.append(ar)
            print(f'[seed {seed}] ANCORA nativa IC+MHEG: FR={fr:.4f} TTDef={ttdef:.4f} '
                  f'Sup={100*sup.mean():.2f}% (pub: 0.000/0.0527+-0.05/57.11+-8.8)', flush=True)
        # 4) warp (com retomada)
        feitos=set()
        if os.path.exists(out):
            import csv as _c
            for r in _c.DictReader(open(out)):
                feitos.add((int(r['seed']),int(r['dur_frames'])))
        for d in DURS:
            if (seed,d) in feitos:
                print(f'[seed {seed}] d={d} ja feito, pulando', flush=True); continue
            X, Y, metas = build_test_set(seed, d)
            yep = np.array([1 if mm['categoria']=='Critico' else 0 for mm in metas])
            yfr = ((Y > FRAME_THR).astype(np.int64) * yep[:, None])
            onsets = [int(np.where(yfr[i] > 0.1)[0][0]) if (yfr[i] > 0.1).any() else -1
                      for i in range(len(yfr))]
            Praw = fixed_policy_probs(m, X.astype(np.float32))
            Pcal = ic.transform(Praw)
            Pg, sup = gate_replay(Pcal, X.astype(np.float32), tau_d, tau_h)
            ttd, fr, ttdef, nc = ttd_metrics_binary(yfr, Pg, theta)
            fsn, fss = fs_onset_original(sup, Pg, yfr, onsets, theta)
            row = dict(seed=seed, dur_frames=d, dur_s=round(d*DT,3),
                       config='Baseline+IC+MHEG', FR=round(fr,4), TTD=round(ttd,4),
                       TTDef=round(ttdef,4), SupPct=round(100*sup.mean(),2),
                       FSonset=round(fsn,4), FSonsetSimple=round(fss,4),
                       F1=round(f1_episode(Pg, yep),4), ECE=round(ece_frame(Pg, yfr),4),
                       tau_d=round(tau_d,6), tau_h=tau_h, n_crit=nc)
            with open(out,'a',newline='') as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            print(f'[seed {seed}] d={d} FR={fr:.4f} TTDef={ttdef:.4f} Sup={100*sup.mean():.1f}%', flush=True)
    json.dump(anchor_rows, open(os.path.join(HERE,'ic_warp_anchor.json'),'w'), indent=1)
    print('DONE')

if __name__ == '__main__':
    main()
