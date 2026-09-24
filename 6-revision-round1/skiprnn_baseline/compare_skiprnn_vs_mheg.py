#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparacao HARMONIZADA Skip-RNN vs AF-TOI(+MHEG) — mesma convencao, mesma
particao, mesmos thresholds de deteccao. Elimina a mistura de protocolos.

Protocolo unico (declarado):
  - particao de teste publicada da campanha canonica (seeds 42-45);
  - AF-TOI: checkpoints CONGELADOS da campanha (student_seedXX.pt) via LSTM
    numpy validada (erro zero vs campanha publicada); MHEG = gate_replay com
    (tau_delta, tau_H) congelados de calibration_summary.csv;
  - deteccao: m=3 frames consecutivos p>=theta A PARTIR do onset;
    TTDef = media-sobre-detectados + FR*10; F1 episodico fire-anywhere;
  - theta = theta_TTD canonico da seed (0.10) — o MESMO para os dois sistemas.

Rodar NESTA pasta (skiprnn_baseline), CPU basta:
    python compare_skiprnn_vs_mheg.py
Saida: comparison_skiprnn_mheg.csv + tabela no console.
"""
import os, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TSG  = os.path.join(HERE, "..", "temporal_scale_generalization")
sys.path.insert(0, TSG)
from np_model import load_state_dict, NpLSTM, fixed_policy_probs   # noqa: E402
from train_skiprnn import metrics, M_CONSEC                        # noqa: E402

MODELS = os.path.join(TSG, "frozen_models")
CAMP   = os.path.join(HERE, "..", "..", "tera-adhoc-artifact",
                      "2-campaign-exp_20260519_055950", "data")
SEEDS  = [42, 43, 44, 45]
KIN    = [12, 13, 14]

CAL = {int(r["seed"]): (float(r["theta_ttd"]), float(r["tau_delta"]), float(r["tau_h"]))
       for r in csv.DictReader(open(os.path.join(MODELS, "calibration_summary.csv")))}

def binary_entropy(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def gate_replay(P, X, tau_d, tau_h):
    n, T = P.shape
    out = np.empty_like(P); sup = np.zeros((n, T), bool)
    for i in range(n):
        last = P[i, 0]; out[i, 0] = last
        for t in range(1, T):
            dk = np.abs(X[i, t, KIN] - X[i, t - 1, KIN]).max()
            if (dk >= tau_d) or (binary_entropy(last) >= tau_h):
                last = P[i, t]
            else:
                sup[i, t] = True
            out[i, t] = last
    return out, sup

def load_grid():
    rows = list(csv.DictReader(open(os.path.join(HERE, "results", "skiprnn_grid.csv"))))
    for r in rows:
        for k in ("FR", "TTDef", "SupPct", "F1"):
            r[k] = float(r[k])
        r["seed"] = int(r["seed"]); r["theta"] = float(r["theta"])
    return rows

def pick(rows, seed, lam, theta, split="test"):
    return [r for r in rows if r["seed"] == seed and r["split"] == split
            and abs(r["theta"] - theta) < 1e-9 and float(r["lam"]) == lam][0]

def main():
    grid = load_grid()
    lams = sorted({float(r["lam"]) for r in grid})
    out_rows = []
    print(f"{'seed':>4} {'sistema':<28} {'theta':>5} {'FR':>7} {'TTDef':>7} {'%Sup':>6} {'F1':>6}")
    for seed in SEEDS:
        theta, tau_d, tau_h = CAL[seed]
        X  = np.load(os.path.join(CAMP, f"X_te_seed{seed}.npy")).astype(np.float32)
        yf = np.load(os.path.join(CAMP, f"y_fr_te_seed{seed}.npy")).astype(np.float32)
        ye = np.load(os.path.join(CAMP, f"y_ep_te_seed{seed}.npy")).astype(np.int64)
        model = NpLSTM(load_state_dict(os.path.join(MODELS, f"student_seed{seed}.pt")))
        P = fixed_policy_probs(model, X)
        ones = np.ones_like(P)
        rows = {}
        rows["AF-TOI Fixed"] = metrics(P, ones, yf, ye, theta)
        outP, sup = gate_replay(P, X, tau_d, tau_h)
        rows["AF-TOI+MHEG (frozen)"] = metrics(outP, (~sup).astype(float), yf, ye, theta)
        mheg_sup = rows["AF-TOI+MHEG (frozen)"]["SupPct"]
        # Skip-RNN MATCHED-SUPPRESSION (post-hoc): lambda cujo %Sup de teste mais
        # se aproxima do MHEG. [auditoria] "matched" = fracao de supressao casada,
        # NAO custo/FLOPs identicos (unidades de computacao diferem entre sistemas);
        # a escolha usa a propria particao avaliada (diagnostico post-hoc).
        best_lam = min(lams, key=lambda L: abs(pick(grid, seed, L, theta)["SupPct"] - mheg_sup))
        rows[f"Skip-RNN (matched-sup, lam={best_lam})"] = (pick(grid, seed, best_lam, theta), theta)
        # Skip-RNN safe point (max %Sup com FR<=0.05 na validacao, qualquer theta)
        val_ok = [r for r in grid if r["seed"] == seed and r["split"] == "val" and r["FR"] <= 0.05]
        bs = max(val_ok, key=lambda r: r["SupPct"])
        # [auditoria] theta reportado = o theta REALMENTE usado no ponto seguro
        rows[f"Skip-RNN (safe, lam={bs['lam']})"] = (pick(grid, seed, float(bs["lam"]), bs["theta"]), bs["theta"])
        for name, val in rows.items():
            m, th_used = (val if isinstance(val, tuple) else (val, theta))
            print(f"{seed:>4} {name:<32} {th_used:>5.2f} {m['FR']:>7.3f} {m['TTDef']:>7.3f} "
                  f"{m['SupPct']:>6.1f} {m['F1']:>6.3f}")
            out_rows.append(dict(seed=seed, system=name, theta=th_used, **{k: m[k] for k in ("FR", "TTDef", "SupPct", "F1")}))
        print()
    with open(os.path.join(HERE, "comparison_skiprnn_mheg.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); [w.writerow(r) for r in out_rows]
    # agregados
    import statistics as st
    names = ["AF-TOI Fixed", "AF-TOI+MHEG (frozen)"]
    print("=== mean±sd (4 seeds) ===")
    for base in names + ["Skip-RNN (matched", "Skip-RNN (safe"]:
        sel = [r for r in out_rows if r["system"].startswith(base)]
        if len(sel) < 2: continue
        print(f"{base:<26} FR={st.mean([r['FR'] for r in sel]):.3f}±{st.stdev([r['FR'] for r in sel]):.3f} "
              f"TTDef={st.mean([r['TTDef'] for r in sel]):.3f}±{st.stdev([r['TTDef'] for r in sel]):.3f} "
              f"%Sup={st.mean([r['SupPct'] for r in sel]):.1f}±{st.stdev([r['SupPct'] for r in sel]):.1f} "
              f"F1={st.mean([r['F1'] for r in sel]):.3f}±{st.stdev([r['F1'] for r in sel]):.3f}")
    print("[done] comparison_skiprnn_mheg.csv")

if __name__ == "__main__":
    main()
