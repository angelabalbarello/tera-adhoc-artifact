#!/usr/bin/env python3
"""Infra comum v2: modelos congelados, splits da campanha, convencoes de metrica."""
import sys, csv, json
import numpy as np
sys.path.insert(0, '../temporal_scale_generalization')
from np_model import load_state_dict, NpLSTM, fixed_policy_probs, sha256
from sklearn.model_selection import train_test_split

MODELS_DIR = '../temporal_scale_generalization/frozen_models'
DATA_DIR   = 'data_campaign'
SEEDS = list(range(42,54))
KIN = [12,13,14]; DT = 10.0/96; FRAME_THR = 0.35; M_CONSEC = 3
CAL = {int(r['seed']): (float(r['theta_ttd']), float(r['tau_delta']), float(r['tau_h']))
       for r in csv.DictReader(open(f'{MODELS_DIR}/calibration_summary.csv'))}

def binary_entropy(p):
    p = np.clip(p, 1e-7, 1-1e-7)
    return -(p*np.log2(p) + (1-p)*np.log2(1-p))

def load_model(seed, tag):
    fn = f'{MODELS_DIR}/{"baseline" if tag=="base" else "student"}_seed{seed}.pt'
    return NpLSTM(load_state_dict(fn))

PUB_IDX = '/tmp/tera-repo/2-campaign-exp_20260519_055950/data'

def campaign_test_split(seed):
    """Reproduz _make_splits do tera_pipeline (70/15/15 estratificado, random_state=seed)."""
    npz = np.load(f'{DATA_DIR}/dataset_sintetico_seed{seed}.npz')
    rows = list(csv.DictReader(open(f'{DATA_DIR}/episodios_seed{seed}.csv'), delimiter=';'))
    X = npz['X'].astype(np.float32); ycont = npz['y_frame_clean'].astype(np.float32)
    cat = np.array([r['class_name'].strip() for r in rows])
    y_ep = (cat == 'Critico').astype(np.int64)
    y_fr = ((ycont > FRAME_THR).astype(np.int64) * y_ep[:,None]).astype(np.int64)
    strat = np.where(cat=='Critico', np.array([r['recipe_id'] for r in rows]), cat)
    import os
    pub = f'{PUB_IDX}/indices_te_seed{seed}.npy'
    if os.path.exists(pub):
        te = np.load(pub)          # indices EXATOS da campanha (seeds 42-45)
    else:
        idx = np.arange(len(X))    # reproducao deterministica (seeds 46-53; ver nota no REPORT)
        tr, tmp = train_test_split(idx, test_size=0.30, random_state=seed, stratify=strat)
        va, te  = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=strat[tmp])
    prog = np.array([int(r['progressive']) for r in rows])
    return te, X[te], y_fr[te], y_ep[te], ycont[te], prog[te]

def first_stable_detection(probs, onset, thr, m=3):
    count = 0
    for t in range(int(onset), len(probs)):
        if probs[t] >= thr:
            count += 1
            if count >= m: return t - m + 1
        else: count = 0
    return None

def ttd_metrics_binary(y_fr_bin, P, thr, m=3):
    """Convencao da campanha: onset = 1o frame com label binario >0.1 (i.e., ==1)."""
    ttds, misses, positives = [], 0, 0
    for yt, yp in zip(y_fr_bin, P):
        idx = np.where(yt > 0.1)[0]
        if len(idx)==0: continue
        positives += 1
        det = first_stable_detection(yp, int(idx[0]), thr, m)
        if det is None: misses += 1
        else: ttds.append(max(0, det-int(idx[0]))*DT)
    # convencao da campanha canonica (= Eq. do artigo): TTD sobre DETECTADOS; TTDef = TTD_det + FR*T_ep
    ttd = float(np.mean(ttds)) if ttds else 0.0
    fr = misses/positives if positives else 0.0
    return ttd, fr, ttd + fr*10.0, positives

def gate_replay(P, X, tau_d, tau_h):
    n, T = P.shape
    out = np.empty_like(P); sup = np.zeros((n,T), bool)
    for i in range(n):
        last = P[i,0]; out[i,0] = last
        for t in range(1,T):
            dk = np.abs(X[i,t,KIN]-X[i,t-1,KIN]).max()
            if (dk >= tau_d) or (binary_entropy(last) >= tau_h): last = P[i,t]
            else: sup[i,t] = True
            out[i,t] = last
    return out, sup

def fs_onset_original(sup, out, y_fr_bin, onsets, theta):
    """Stress campaign: janela [t0-5,t0+5]; pos = frame positivo; danosa = pos & sup & held<theta."""
    vals, simple = [], []
    for i, t0 in enumerate(onsets):
        if t0 < 0: continue
        L = sup.shape[1]
        w0, w1 = max(0, t0-5), min(L, t0+6)
        win = np.zeros(L, bool); win[w0:w1] = True
        pos = y_fr_bin[i] > 0.1
        wp = win & pos
        if wp.sum() > 0:
            harmful = wp & sup[i] & (out[i] < theta)
            vals.append(harmful.sum()/wp.sum())
        simple.append(sup[i, w0:w1].mean())
    return (float(np.mean(vals)) if vals else 0.0), (float(np.mean(simple)) if simple else 0.0)

def ece_frame(P, y_fr_bin, bins=10):
    p = P.ravel(); y = (y_fr_bin > 0.1).ravel().astype(float)
    edges = np.linspace(0,1,bins+1); e = 0.0
    for b in range(bins):
        m = (p >= edges[b]) & (p < edges[b+1] if b < bins-1 else p <= 1.0)
        if m.sum()==0: continue
        e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)

def f1_episode(P, y_ep, thr_ep=0.05):
    ep = P[:,-6:].mean(axis=1)
    pred = (ep >= 0.5).astype(int)   # F1 informativo (thr 0.5); campanha usa outra convencao p/ P/R
    tp=int(((pred==1)&(y_ep==1)).sum()); fp=int(((pred==1)&(y_ep==0)).sum()); fn=int(((pred==0)&(y_ep==1)).sum())
    prec=tp/(tp+fp) if tp+fp else 0.0; rec=tp/(tp+fn) if tp+fn else 0.0
    return 2*prec*rec/(prec+rec) if prec+rec else 0.0
