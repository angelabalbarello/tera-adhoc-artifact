#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skip-RNN CONGELADO no grid de escala temporal (analogo exato da Sec. 4.12).
============================================================================
Pergunta: o gate aprendido do Skip-RNN preserva cobertura/pontualidade quando
a duracao da transicao sai do suporte de treino, SEM retreino/retuning?
(O MHEG congelado preserva em agregado — frozen_results.csv da Parte A.)

Protocolo: mesmos conjuntos warped da Sec. 4.12 (build_test_set, 156 eps/celula),
seeds 42-45, grid d = {2,6,10,14,18,22,31,42,50} frames; checkpoints Skip-RNN
CONGELADOS (matched-lambda e max-suppression); metricas na convencao canonica
(m=3 pos-onset, theta=0.10, TTDef = media-detectados + FR*10).

RODAR NESTA PASTA (temporal_scale_generalization_v2):
    python skiprnn_temporal_scale.py
Saida: skiprnn_temporal_scale.csv + comparacao com AF-TOI+MHEG congelado.
"""
import os, sys, csv, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
# gerador TERA-Gen (synthetic_driver_risk_v7) — candidatos locais
for cand in (os.path.join(HERE, "..", "scripts"),
             os.path.join(HERE, "..", "..", "tera-adhoc-artifact", "1-pipeline-tera"),
             os.path.join(HERE, "..", "..", "FGCS", "Inferencia")):
    if os.path.exists(os.path.join(cand, "synthetic_driver_risk_v7.py")):
        sys.path.insert(0, os.path.abspath(cand)); break
else:
    sys.exit("ERRO: synthetic_driver_risk_v7.py nao encontrado nas pastas candidatas.")
from gen_warped import build_test_set          # noqa: E402

import torch, torch.nn as nn                   # noqa: E402
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS   = [42, 43, 44, 45]
DURS    = [2, 6, 10, 14, 18, 22, 31, 42, 50]
DT      = 10.0 / 96
FPS     = 9.6
T_EP    = 10.0
M       = 3
THETA   = 0.10
FRAME_THR = 0.35
H, LAYERS = 64, 2
CKDIR   = os.path.join(HERE, "..", "skiprnn_baseline", "results")
# lambdas por seed: matched-compute (harmonizado) e max-suppression
MATCHED = {42: 5e-05, 43: 5e-05, 44: 5e-05, 45: 0.0001}
SAFELAM = 0.005

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
        probs, us = [], []
        h_out = torch.zeros(B, H, device=x.device)
        for t in range(T):
            u = (u_tilde >= 0.5).float()
            out, (h_new, c_new) = self.lstm(x[:, t:t+1, :], (hs[0], hs[1]))
            h_out = u * out[:, 0, :] + (1 - u) * h_out
            hs[0] = u.unsqueeze(0) * h_new + (1 - u).unsqueeze(0) * hs[0]
            hs[1] = u.unsqueeze(0) * c_new + (1 - u).unsqueeze(0) * hs[1]
            p = torch.sigmoid(self.head(h_out))
            delta_u = torch.sigmoid(self.gate(h_out))
            u_tilde = u * delta_u + (1 - u) * torch.clamp(u_tilde + torch.minimum(delta_u, 1 - u_tilde), 0, 1)
            probs.append(p); us.append(u)
        return torch.cat(probs, 1), torch.cat(us, 1)

def metrics(P, U, yfr_bin, yep, theta):
    N, T = P.shape
    above = P >= theta
    # [auditoria-4] denominador harmonizado com ttd_metrics_binary (common.py):
    # episodio critico SEM frame positivo e EXCLUIDO das metricas de deteccao
    has_pos = (yfr_bin > 0.1).any(axis=1)
    crit = (yep > 0) & has_pos
    onset = np.argmax(yfr_bin > 0.1, axis=1)
    def first_run(row, start):
        run = 0
        for t in range(start, T):
            run = run + 1 if row[t] else 0
            if run >= M: return t - M + 1
        return -1
    det_any = np.array([first_run(above[i], 0) >= 0 for i in range(N)])
    det_post = np.zeros(N, bool); tdet = np.full(N, -1)
    for i in range(N):
        if crit[i]:
            td = first_run(above[i], int(onset[i]))
            det_post[i] = td >= 0; tdet[i] = td
    fr = float(np.mean(~det_post[crit])) if crit.any() else 0.0
    dts = [max(0.0, (tdet[i] - onset[i]) / FPS) for i in range(N) if crit[i] and det_post[i]]
    ttdef = (float(np.mean(dts)) if dts else 0.0) + fr * T_EP
    tp = float(np.sum(det_any & crit)); fp = float(np.sum(det_any & ~crit)); fn = float(np.sum(~det_any & crit))
    f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
    return dict(FR=fr, TTDef=ttdef, SupPct=100.0 * float(1 - U.mean()), F1=f1)

def load_ck(seed, lam):
    p = os.path.join(CKDIR, f"skiprnn_seed{seed}_lam{lam}.pt")
    m = SkipLSTM().to(DEV)
    m.load_state_dict(torch.load(p, map_location=DEV)); m.eval()
    return m

def main():
    rows = []
    models = {(s, "matched"): load_ck(s, MATCHED[s]) for s in SEEDS}
    models.update({(s, "maxsup"): load_ck(s, SAFELAM) for s in SEEDS})
    for seed in SEEDS:
        for d in DURS:
            X, Y, metas = build_test_set(seed, d)
            yep = np.array([1 if m["categoria"] == "Critico" else 0 for m in metas])
            yfr = ((Y > FRAME_THR).astype(np.int64) * yep[:, None])
            Xt = torch.tensor(X, device=DEV)
            for arm in ("matched", "maxsup"):
                with torch.no_grad():
                    P, U = models[(seed, arm)](Xt)
                mm = metrics(P.cpu().numpy(), U.cpu().numpy(), yfr, yep, THETA)
                rows.append(dict(seed=seed, dur_frames=d, dur_s=round(d * DT, 3), arm=arm, **mm))
            print(f"[ok] seed{seed} d={d} ({d*DT:.2f}s)")
    with open("skiprnn_temporal_scale.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]

    # comparacao com AF-TOI+MHEG congelado (Parte A, mesmas seeds/celulas)
    ref = {}
    try:
        for r in csv.DictReader(open("duration_only/frozen_results.csv")):
            if r["config"] == "AF-TOI+MHEG" and int(r["seed"]) in SEEDS:
                ref.setdefault(int(r["dur_frames"]), []).append(
                    (float(r["FR"]), float(r["TTDef"]), float(r["SupPct"])))
    except FileNotFoundError:
        print("[aviso] duration_only/frozen_results.csv nao encontrado — sem coluna de referencia")
    import statistics as st
    print(f"\n{'d(s)':>5} | {'SkipRNN-matched':^30} | {'SkipRNN-maxsup':^30} | {'AF-TOI+MHEG frozen':^28}")
    print(f"{'':>5} | {'FR':>6} {'TTDef':>7} {'%Sup':>6} | {'FR':>6} {'TTDef':>7} {'%Sup':>6} | {'FR':>6} {'TTDef':>7} {'%Sup':>6}")
    for d in DURS:
        line = f"{d*DT:>5.2f} |"
        for arm in ("matched", "maxsup"):
            sel = [r for r in rows if r["dur_frames"] == d and r["arm"] == arm]
            line += f" {st.mean([r['FR'] for r in sel]):>6.3f} {st.mean([r['TTDef'] for r in sel]):>7.3f} {st.mean([r['SupPct'] for r in sel]):>6.1f} |"
        if d in ref:
            line += f" {st.mean([x[0] for x in ref[d]]):>6.3f} {st.mean([x[1] for x in ref[d]]):>7.3f} {st.mean([x[2] for x in ref[d]]):>6.1f}"
        print(line)
    print("\n[done] skiprnn_temporal_scale.csv")

if __name__ == "__main__":
    main()
