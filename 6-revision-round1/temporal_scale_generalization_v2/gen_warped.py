#!/usr/bin/env python3
"""Geracao contrafactual SHAPE-PRESERVING por time-warp da curva ORIGINAL do TERA-Gen.
Constroi o episodio exatamente como generate_episode() (mesma ordem de RNG),
computa a curva original r_orig (curve_for_name + persistent tail), mede d0
(t0 -> 90% do pico) e entao aplica warp temporal
    r_d(t) = r_orig(s + (t-s)*d0/d)   para t >= s   (interp. linear, fonte
    limitada ao ultimo indice; pre-onset inalterado)
Todo RNG e sorteado ANTES do warp e nao depende de d -> contrafactuais pareados.
Unica divergencia de pareamento (documentada): regime R2 pre-sorteia bl p/ todos
os frames (o original sorteia condicionalmente a r>0.55).
"""
import sys, numpy as np
sys.path.insert(0, '/sessions/relaxed-lucid-thompson/mnt/TreinamentoNovo/FGCS/Inferencia')
import synthetic_driver_risk_v7 as g

T = 96
SENSOR_LEVEL = 4

def _original_curve_and_skeleton(recipe_key, ep_seed):
    """Fase 1: skeleton + curva ORIGINAL (RNG identico ao generate_episode ate o fim da curva)."""
    np.random.seed(ep_seed)
    spec = g.RECIPES_TRAIN[recipe_key]
    frames, state_names = [], []
    blocks = spec["sequence"]
    scale = T / sum(d for _, d, _ in blocks)
    for name, dur, kw in blocks:
        fn = g.STATE_FUNCS[name]
        for _ in range(max(1, int(round(dur * scale)))):
            if len(frames) >= T: break
            frames.append(fn(SENSOR_LEVEL=SENSOR_LEVEL, **kw))
            state_names.append(name)
    while len(frames) < T:
        frames.append(frames[-1]); state_names.append(state_names[-1])
    X = np.stack(frames, axis=0)

    base_curve = np.full(T, g.LOW_RISK_BASE, dtype=np.float32)
    segs = g.find_event_segments(state_names)
    for (s_, e_) in segs:
        L = max(1, e_ - s_)
        cname = spec.get("curve", "")
        if spec.get("categoria") == "Critico" and spec.get("progressive", False):
            cname = "critical_progressive"
        base_curve[s_:e_] = np.maximum(base_curve[s_:e_],
                                       g.curve_for_name(cname, L, spec["risk_target"]))
    if spec.get("categoria") == "Critico":
        base_curve, _ = g.inject_critical_persistent_tail(base_curve, segs, spec["risk_target"])
    return spec, X, state_names, segs, base_curve.astype(np.float32)

def measure_d0(curve, s):
    """d0 = frames de t0 ate atingir 90% do pico pos-onset (curva original)."""
    peak = curve[s:].max()
    thr = 0.9 * peak
    for t in range(s, T):
        if curve[t] >= thr: return t - s, float(peak)
    return T - 1 - s, float(peak)

def time_warp(curve, s, d0, d):
    """Warp shape-preserving do trecho pos-onset. d0,d em frames (>0)."""
    out = curve.copy()
    if d0 <= 0 or d <= 0 or d == d0: return out
    f = d0 / float(d)
    src = s + (np.arange(s, T) - s) * f
    src = np.clip(src, 0, T - 1)
    i0 = np.floor(src).astype(int); i1 = np.minimum(i0 + 1, T - 1)
    w = (src - i0).astype(np.float32)
    out[s:] = (1 - w) * curve[i0] + w * curve[i1]
    return out.astype(np.float32)

def generate_episode_warped(recipe_key, d_target, ep_seed):
    """d_target em frames; para Normais, curva original (sem warp)."""
    spec, X, state_names, segs, curve0 = _original_curve_and_skeleton(recipe_key, ep_seed)
    is_crit = (spec.get("categoria") == "Critico") and len(segs) > 0
    s = int(segs[0][0]) if segs else -1
    if is_crit:
        d0, peak = measure_d0(curve0, s)
        base_curve = time_warp(curve0, s, d0, int(d_target))
    else:
        d0, peak = -1, -1.0
        base_curve = curve0

    # ---- regime cinematico (fiel; bl pre-sorteado p/ pareamento) ----
    u = np.random.rand()
    regime = ("R2_STABLE_RISK" if u < g.P_REGIME_STABLE_RISK
              else "R3_FALSE_KIN" if u < g.P_REGIME_STABLE_RISK + g.P_REGIME_FALSE_KIN
              else "R1_NORMAL")
    road_noise = np.random.normal(0, g.KIN_NOISE_BASE, size=T)
    if regime == "R1_NORMAL":
        for t in range(T):
            r = base_curve[t]
            X[t,12]=g.clip01(0.30*r+road_noise[t]); X[t,13]=g.clip01(0.22*r+road_noise[t]); X[t,14]=g.clip01(0.16*r+road_noise[t])
    elif regime == "R2_STABLE_RISK":
        bls = 0.18 + 0.05*np.random.rand(T)
        for t in range(T):
            r = base_curve[t]
            if r > 0.55:
                bl = bls[t]
                X[t,12]=g.clip01(bl+0.5*road_noise[t]); X[t,13]=g.clip01(bl*0.8+0.5*road_noise[t]); X[t,14]=g.clip01(bl*0.6+0.5*road_noise[t])
            else:
                X[t,12]=g.clip01(0.30*r+road_noise[t]); X[t,13]=g.clip01(0.22*r+road_noise[t]); X[t,14]=g.clip01(0.16*r+road_noise[t])
    else:
        spikes = np.zeros(T, dtype=np.float32)
        for _ in range(np.random.randint(2, 6)):
            c=np.random.randint(0,T); w=np.random.randint(2,8)
            a=np.random.uniform(0.15, g.KIN_FALSE_SPIKE_SCALE)
            spikes[max(0,c-w):min(T,c+w)] = np.maximum(spikes[max(0,c-w):min(T,c+w)], a)
        for t in range(T):
            r = base_curve[t]
            X[t,12]=g.clip01(0.18*r+spikes[t]+road_noise[t]); X[t,13]=g.clip01(0.14*r+0.9*spikes[t]+road_noise[t]); X[t,14]=g.clip01(0.10*r+0.7*spikes[t]+road_noise[t])

    # ---- precursor (fiel) ----
    if np.random.rand() < g.P_PRECURSOR_VISUAL and len(segs) > 0:
        first_start = segs[0][0]
        L = np.random.randint(g.PRECURSOR_LEN_MIN, g.PRECURSOR_LEN_MAX + 1)
        t0p = max(0, first_start - L)
        drop = np.random.uniform(g.PRECURSOR_CONF_DROP[0], g.PRECURSOR_CONF_DROP[1])
        for t in range(t0p, first_start):
            X[t,9]  = g.clip01(float(X[t,9])  - drop*np.random.uniform(0.5,0.9))
            X[t,10] = g.clip01(float(X[t,10]) - drop*np.random.uniform(0.5,0.9))
            if spec.get("progressive", False):
                if X[t,16] < 0.5 and np.random.rand() < 0.20:
                    X[t,16] = g.clip01(float(X[t,16]) + np.random.uniform(0.05,0.15))
                if SENSOR_LEVEL >= 3:
                    X[t,12] = g.clip01(float(X[t,12]) + np.random.uniform(0.01,0.05))
                    X[t,13] = g.clip01(float(X[t,13]) + np.random.uniform(0.01,0.04))
                    X[t,14] = g.clip01(float(X[t,14]) + np.random.uniform(0.00,0.03))

    # ---- gaze/lane + y_clean (fiel) ----
    X[:,29] = g.generate_eye_gaze_series(state_names, base_curve, SENSOR_LEVEL)
    X[:,30] = g.generate_lane_offset_series(base_curve, state_names)
    y_clean = np.clip(base_curve.astype(np.float32), 0.0, 1.0)
    kernel = np.ones(3, dtype=np.float32)/3.0
    ys = np.convolve(y_clean, kernel, mode='same').astype(np.float32)
    ys[0]=y_clean[0]; ys[-1]=y_clean[-1]
    y_clean = np.clip(ys, 0.0, 1.0)

    meta = dict(recipe=recipe_key, categoria=spec["categoria"],
                progressive=int(bool(spec.get("progressive", False))),
                event_start=s, d0=int(d0), d_target=(int(d_target) if is_crit else -1),
                regime=regime, peak=peak, ep_seed=ep_seed)
    return X.astype(np.float32), y_clean, base_curve, curve0, meta

CRIT = [f'C{i:02d}' for i in range(1,11)]
NORM = ['N01','N02','N03']

def build_test_set(seed, d_frames, per_recipe=12, return_orig=False):
    Xs, Ys, metas, origs = [], [], [], []
    k = 0
    for rk in CRIT + NORM:
        for j in range(per_recipe):
            ep_seed = seed*1_000_000 + k
            X, y, bc, c0, m = generate_episode_warped(rk, d_frames, ep_seed)
            Xs.append(X); Ys.append(y); metas.append(m); origs.append(c0); k += 1
    if return_orig:
        return np.stack(Xs), np.stack(Ys), metas, np.stack(origs)
    return np.stack(Xs), np.stack(Ys), metas
