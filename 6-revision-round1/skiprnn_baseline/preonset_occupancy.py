#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ocupacao de alerta PRE-ONSET — semantica temporal do sinal de risco.
====================================================================
Mede, no conjunto de teste publicado da campanha (seeds 42-45, theta=0.10):
  - PreOcc  : fracao media de frames PRE-onset com p>=theta em episodios
              criticos (alarme durante conducao ainda normal);
  - NormOcc : fracao media de frames com p>=theta em episodios NAO-criticos;
  - Ppre/Ppost: media de p antes/depois do onset (estratificacao temporal).
Sistemas: AF-TOI Fixed (frozen), AF-TOI+MHEG (frozen, probs held),
Skip-RNN matched-lambda e Skip-RNN max-suppression (congelados).

RODAR NESTA pasta (skiprnn_baseline):  python preonset_occupancy.py
Saida: preonset_occupancy.csv + medias no console.
"""
import os, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TSG  = os.path.join(HERE, "..", "temporal_scale_generalization")
sys.path.insert(0, TSG)
from np_model import load_state_dict, NpLSTM, fixed_policy_probs   # noqa: E402

import torch, torch.nn as nn                                       # noqa: E402
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = os.path.join(TSG, "frozen_models")
CAMP   = os.path.join(HERE, "..", "..", "tera-adhoc-artifact",
                      "2-campaign-exp_20260519_055950", "data")
SEEDS  = [42, 43, 44, 45]
KIN    = [12, 13, 14]
THETA  = 0.10
H, LAYERS = 64, 2
MATCHED = {42: 5e-05, 43: 5e-05, 44: 5e-05, 45: 0.0001}
SAFELAM = 0.005

CAL = {int(r["seed"]): (float(r["theta_ttd"]), float(r["tau_delta"]), float(r["tau_h"]))
       for r in csv.DictReader(open(os.path.join(MODELS, "calibration_summary.csv")))}

def binary_entropy(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def gate_replay(P, X, tau_d, tau_h):
    n, T = P.shape
    out = np.empty_like(P)
    for i in range(n):
        last = P[i, 0]; out[i, 0] = last
        for t in range(1, T):
            dk = np.abs(X[i, t, KIN] - X[i, t - 1, KIN]).max()
            if (dk >= tau_d) or (binary_entropy(last) >= tau_h):
                last = P[i, t]
            out[i, t] = last
    return out

class SkipLSTM(nn.Module):
    def __init__(self, D=31, H=H, L=LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(D, H, num_layers=L, batch_first=True)
        self.head = nn.Linear(H, 1)
        self.gate = nn.Linear(H, 1)
    def forward(self, x):
        B, T, D = x.shape
        hs = [torch.zeros(self.lstm.num_layers, B, H, device=x.device) for _ in range(2)]
        u_tilde = torch.ones(B, 1, device=x.device)
        probs = []
        h_out = torch.zeros(B, H, device=x.device)
        for t in range(T):
            u = (u_tilde >= 0.5).float()
            out, (h_new, c_new) = self.lstm(x[:, t:t+1, :], (hs[0], hs[1]))
            h_out = u * out[:, 0, :] + (1 - u) * h_out
            hs[0] = u.unsqueeze(0) * h_new + (1 - u).unsqueeze(0) * hs[0]
            hs[1] = u.unsqueeze(0) * c_new + (1 - u).unsqueeze(0) * hs[1]
            probs.append(torch.sigmoid(self.head(h_out)))
            delta_u = torch.sigmoid(self.gate(h_out))
            u_tilde = u * delta_u + (1 - u) * torch.clamp(u_tilde + torch.minimum(delta_u, 1 - u_tilde), 0, 1)
        return torch.cat(probs, 1)

def occupancy(P, yfr, yep, theta):
    crit = yep > 0
    onset = np.argmax(yfr > 0.1, axis=1)
    pre_occ, ppre, ppost = [], [], []
    for i in range(len(P)):
        if not crit[i] or onset[i] < 3:
            continue
        pre = P[i, :onset[i]]; post = P[i, onset[i]:]
        pre_occ.append(float((pre >= theta).mean()))
        ppre.append(float(pre.mean())); ppost.append(float(post.mean()))
    norm_occ = [float((P[i] >= theta).mean()) for i in range(len(P)) if not crit[i]]
    f = lambda v: float(np.mean(v)) if v else float("nan")
    return dict(PreOcc=f(pre_occ), NormOcc=f(norm_occ), Ppre=f(ppre), Ppost=f(ppost))

def main():
    rows = []
    for seed in SEEDS:
        theta_c, tau_d, tau_h = CAL[seed]
        X  = np.load(os.path.join(CAMP, f"X_te_seed{seed}.npy")).astype(np.float32)
        yf = np.load(os.path.join(CAMP, f"y_fr_te_seed{seed}.npy")).astype(np.float32)
        ye = np.load(os.path.join(CAMP, f"y_ep_te_seed{seed}.npy")).astype(np.int64)
        af = NpLSTM(load_state_dict(os.path.join(MODELS, f"student_seed{seed}.pt")))
        P_af = fixed_policy_probs(af, X)
        P_mheg = gate_replay(P_af, X, tau_d, tau_h)
        systems = {"AF-TOI Fixed": P_af, "AF-TOI+MHEG": P_mheg}
        Xt = torch.tensor(X, device=DEV)
        for arm, lam in (("Skip-RNN matched", MATCHED[seed]), ("Skip-RNN maxsup", SAFELAM)):
            m = SkipLSTM().to(DEV)
            m.load_state_dict(torch.load(os.path.join(HERE, "results", f"skiprnn_seed{seed}_lam{lam}.pt"),
                                         map_location=DEV))
            m.eval()
            with torch.no_grad():
                systems[arm] = m(Xt).cpu().numpy()
        for name, P in systems.items():
            rows.append(dict(seed=seed, system=name, **occupancy(P, yf, ye, THETA)))
        print(f"[ok] seed{seed}")
    with open(os.path.join(HERE, "preonset_occupancy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
    import statistics as st
    print(f"\n{'sistema':<18} {'PreOcc':>8} {'NormOcc':>8} {'Ppre':>7} {'Ppost':>7}   (medias, 4 seeds; theta=0.10)")
    for name in ("AF-TOI Fixed", "AF-TOI+MHEG", "Skip-RNN matched", "Skip-RNN maxsup"):
        sel = [r for r in rows if r["system"] == name]
        print(f"{name:<18} " + " ".join(
            f"{st.mean([r[k] for r in sel]):>8.3f}" if k == "PreOcc" else f"{st.mean([r[k] for r in sel]):>8.3f}"
            for k in ("PreOcc", "NormOcc", "Ppre", "Ppost")))
    print("\nPreOcc  = fracao de frames PRE-onset em alerta (p>=0.10) nos criticos")
    print("NormOcc = fracao de frames em alerta nos episodios normais")
    print("[done] preonset_occupancy.csv")

if __name__ == "__main__":
    main()
