"""Artigo 1 — Reforço: controles negativos plausíveis, cost-matching e métricas
de janela de onset (false suppression near onset), sobre dados sintético-controlados.

Princípio metodológico: os controles negativos DEVEM realmente tentar economizar
custo (orçamento comparável ao MHEG hybrid). O objetivo é distinguir
'ausência de economia' de 'supressão insegura'. Nenhum resultado é forçado:
os mecanismos são fiéis e o que emergir é reportado como está.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import lab

THETA = 0.5
ONSET_WINDOW = (-5, 5)          # frames relativos ao onset t0
TAU_H, TAU_D = 0.35, 0.30       # limiares MHEG default
TAU_H_CAL, TAU_D_CAL = 0.40, 0.32   # MHEG calibrado (levemente mais econômico)

# ---------------- Episódio enriquecido (canal de entropia NÃO organizada) ------
def make_episode(seed, onset="abrupt", scenario="balanced_onsets", L=60):
    """Reusa o gerador fiel do lab e adiciona: canal H_uns (entropia SEM
    organização temporal, sem informação de onset) + modificadores de cenário."""
    ep = lab.gen_gating_episode(seed, onset=onset, organized=True, L=L)
    r = lab.rng(seed + 7)
    # entropia não-organizada: ruído plano, NÃO estratifica no onset
    ep["H_uns"] = np.clip(np.full(L, 0.5) + r.normal(0, 0.05, L), 0, 1)
    ep["scenario"] = scenario
    ep["seed"] = int(seed)
    ep["has_onset"] = True
    ep["false_onset_t"] = None
    if scenario == "high_noise_entropy":
        ep["H"] = np.clip(ep["H"] + r.normal(0, 0.16, L), 0, 1)   # entropia degradada
    elif scenario == "rare_abrupt_critical_events":
        # evento crítico TRANSITÓRIO: risco alto só por poucos frames -> perder a
        # janela de onset significa perder o evento (coverage vira métrica sensível)
        burst = int(r.integers(3, 6)); t0 = ep["t0"]; e1 = min(L, t0 + burst)
        risk = np.zeros(L); risk[t0:e1] = 1.0
        ep["risk_true"] = risk
        ep["prob"] = np.clip(risk * 0.9 + r.normal(0, 0.04, L), 0, 1)
        Hn = np.full(L, 0.12) + r.normal(0, 0.03, L)
        Hn[t0:e1] = 0.9 + r.normal(0, 0.03, e1 - t0)
        ep["H"] = np.clip(Hn, 0, 1)
        dxn = np.abs(r.normal(0, 0.04, L)); dxn[t0:t0 + 2] += r.uniform(0.55, 0.8)
        ep["dx"] = np.clip(dxn, 0, 1)
    elif scenario == "kinematic_signal_missing":
        ep["dx"] = np.zeros(L)                                    # cinemática indisponível
        ep["kinematic_available"] = False
    elif scenario == "false_onset":
        tf = int(r.integers(6, 18))                               # pico ANTES do risco, risco=0
        ep["dx"][tf:tf+2] = np.clip(ep["dx"][tf:tf+2] + r.uniform(0.55, 0.8), 0, 1)
        ep["H"][tf:tf+3] = np.clip(ep["H"][tf:tf+3] + 0.4, 0, 1)
        ep["false_onset_t"] = tf
    elif scenario == "no_onset_negative_control":
        ep["risk_true"] = np.zeros(L)                             # nunca há risco real
        ep["prob"] = np.clip(0.12 + r.normal(0, 0.04, L), 0, 1)
        ep["H"] = np.clip(np.full(L, 0.12) + r.normal(0, 0.03, L), 0, 1)
        ep["dx"] = np.abs(r.normal(0, 0.04, L))
        ep["has_onset"] = False
    ep.setdefault("kinematic_available", True)
    return ep

# ---------------- Gatilhos por método (vetor booleano compute[t]) --------------
def trigger(ep, method, params=None):
    p = params or {}
    L = ep["L"]; H = ep["H"]; dx = ep["dx"]; Hu = ep["H_uns"]
    th = p.get("tau_h", TAU_H); td = p.get("tau_d", TAU_D)
    if method == "always":
        return np.ones(L, bool)
    if method == "MHEG_hybrid":
        return (H >= th) | (dx >= td)
    if method == "MHEG_calibrated":
        return (H >= TAU_H_CAL) | (dx >= TAU_D_CAL)
    if method == "MHEG_entropy_only":
        return H >= th
    if method == "MHEG_kinematic_only":
        return dx >= td
    if method == "NC1_random_budget_matched":
        r = lab.rng(int(ep.get("seed", 0)) * 131 + p.get("salt", 0) + 101)
        return r.random(L) < p.get("p_compute", 0.3)
    if method == "NC2_entropy_unstructured_aggressive":
        return Hu >= p.get("tau_h", 0.62)          # agressivo -> força economia
    if method == "NC3_entropy_lagged":
        lag = int(p.get("lag", 3)); Hl = np.concatenate([np.full(lag, H[0]), H[:-lag]]) if lag > 0 else H
        return Hl >= th
    if method == "NC4_entropy_miscalibrated":
        return H >= p.get("tau_h", 0.55)           # limiar calibrado em outra distribuição
    if method == "NC5_periodic_skip":
        N = int(p.get("N", 3)); idx = np.arange(L)
        return (idx % N) == 0
    if method == "NC6_cost_matched_unstructured":
        return Hu >= p.get("tau_h", 0.5)
    raise ValueError("método desconhecido: " + method)

# ---------------- Simulação + métricas (inclui janela de onset e calibração) ---
def run_episode(ep, method, params=None):
    L = ep["L"]; t0 = ep["t0"]; risk = ep["risk_true"]; prob = ep["prob"]
    comp = trigger(ep, method, params)
    avail = prob.copy(); last = prob[0]; held = np.empty(L)
    for t in range(L):
        if comp[t]:
            last = prob[t]
        else:
            avail[t] = last
        held[t] = avail[t]
    pos = risk >= 0.5
    # false suppression global: frame de risco suprimido cujo valor retido < theta
    supp_pos = (~comp) & pos & (held < THETA)
    fs_rate = supp_pos.sum() / max(1, pos.sum()) if pos.any() else 0.0
    # detecção / TTD
    det = None
    if ep["has_onset"]:
        for t in range(t0, L):
            if held[t] >= THETA:
                det = t; break
    detected = det is not None
    ttd = (det - t0) if detected else None
    ttdef = ttd if detected else (L if ep["has_onset"] else np.nan)
    # janela de onset
    w0, w1 = max(0, t0 + ONSET_WINDOW[0]), min(L, t0 + ONSET_WINDOW[1] + 1)
    win = np.zeros(L, bool); win[w0:w1] = True
    win_pos = win & pos
    win_supp = win_pos & (~comp) & (held < THETA)
    n_win_pos = int(win_pos.sum())
    fs_near = (win_supp.sum() / n_win_pos) if n_win_pos > 0 else 0.0
    onset_fail = 1 if win_supp.any() else 0
    # calibração: predição = valor retido; alvo = risco real
    brier = float(np.mean((held - pos.astype(float)) ** 2))
    return dict(cost=float(comp.mean()), detected=detected, ttd=ttd, ttdef=ttdef,
                fs_rate=float(fs_rate), fs_near=float(fs_near), onset_fail=onset_fail,
                n_win_pos=n_win_pos, n_win_supp=int(win_supp.sum()),
                has_onset=ep["has_onset"], onset=ep["onset"], brier=brier,
                held=held, pos=pos.astype(float))

def _ece(held_all, pos_all, bins=10):
    held_all = np.asarray(held_all); pos_all = np.asarray(pos_all)
    edges = np.linspace(0, 1, bins + 1); ece = 0.0; n = len(held_all)
    for i in range(bins):
        m = (held_all >= edges[i]) & (held_all < edges[i + 1] + (1e-9 if i == bins - 1 else 0))
        if m.sum() == 0: continue
        conf = held_all[m].mean(); acc = pos_all[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)

def entropy_stratification_score(eps):
    """AUC simples: entropia discrimina frames de onset (t0..t0+5) vs frames calmos?"""
    onv, offv = [], []
    for ep in eps:
        if not ep["has_onset"]: continue
        t0 = ep["t0"]; on = np.zeros(ep["L"], bool); on[t0:min(ep["L"], t0 + 6)] = True
        onv += list(ep["H"][on]); offv += list(ep["H"][~on])
    onv = np.array(onv); offv = np.array(offv)
    if len(onv) == 0 or len(offv) == 0: return float("nan")
    # AUC via Mann-Whitney
    gt = sum((onv[:, None] > offv[None, :]).sum() for _ in [0]) if len(onv)*len(offv) < 4e6 else None
    if gt is None:
        s = 0
        for v in onv: s += (v > offv).sum() + 0.5 * (v == offv).sum()
        return float(s / (len(onv) * len(offv)))
    eq = (onv[:, None] == offv[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(onv) * len(offv)))

def aggregate(eps, method, params=None):
    rows = [run_episode(e, method, params) for e in eps]
    onset_rows = [r for r in rows if r["has_onset"]]
    def mean(key, subset=rows):
        vals = [r[key] for r in subset if r[key] is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")
    cov = np.mean([r["detected"] for r in onset_rows]) if onset_rows else float("nan")
    cov_ab = np.mean([r["detected"] for r in onset_rows if r["onset"] == "abrupt"]) if any(r["onset"]=="abrupt" for r in onset_rows) else float("nan")
    cov_pr = np.mean([r["detected"] for r in onset_rows if r["onset"] == "progressive"]) if any(r["onset"]=="progressive" for r in onset_rows) else float("nan")
    held_all = np.concatenate([r["held"] for r in rows]); pos_all = np.concatenate([r["pos"] for r in rows])
    return dict(method=method, cost=mean("cost"),
                coverage=cov, miss_rate=(1 - cov) if cov == cov else float("nan"),
                coverage_abrupt=cov_ab, coverage_progressive=cov_pr,
                ttd_det=mean("ttd", onset_rows), ttdef=mean("ttdef", onset_rows),
                fs_rate=mean("fs_rate"), fs_near=mean("fs_near", onset_rows),
                onset_window_failure=mean("onset_fail", onset_rows),
                n_onsets=len(onset_rows),
                n_win_supp=int(sum(r["n_win_supp"] for r in onset_rows)),
                brier=mean("brier"), ece=_ece(held_all, pos_all),
                strat=entropy_stratification_score(eps))

# ---------------- Cost-matching solver -----------------------------------------
def solve_threshold_for_cost(eps, method, target_cost, key="tau_h", lo=0.0, hi=1.0, tol=0.02, iters=40):
    """Ajusta um limiar por busca binária para bater custo alvo (±tol). Fiel:
    se não for possível casar, retorna o melhor e o erro observado."""
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        c = aggregate(eps, method, {key: mid})["cost"]
        if best is None or abs(c - target_cost) < abs(best[1] - target_cost):
            best = (mid, c)
        # custo é DECRESCENTE no limiar (limiar maior -> computa menos)
        if c > target_cost: lo = mid
        else: hi = mid
        if abs(c - target_cost) <= tol * target_cost:
            return mid, c
    return best[0], best[1]
