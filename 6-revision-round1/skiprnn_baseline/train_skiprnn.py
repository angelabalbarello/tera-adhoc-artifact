#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skip-RNN baseline (Campos et al., ICLR 2018) adaptado ao monitoramento
continuo do TERA — R3 Comment 5: published ADAPTIVE TEMPORAL-COMPUTATION baseline
(recurrent state-update allocation). IMPORTANTE: Skip-RNN NAO e
uncertainty-aware (gate aprendido, nao incerteza preditiva) — nao
apresentar como resposta a categoria (v); ele reforca a categoria (iv).
=======================================================================
Fidelidade ao paper original (Sec. 3 de arXiv:1708.06834):
  - gate binario de atualizacao de estado u_t = round(u~_t) (straight-through);
  - u~_{t+1} = u_t * Delta u(s_t) + (1-u_t) * (u~_t + min(Delta u, 1 - u~_t));
  - custo L_budget = lambda * sum_t u_t  (per-sample cost of computation).
Adaptacoes registradas (stream continuo de monitoramento, nao clipe finito):
  A1. o modelo NUNCA sai ("exit"): quando u_t=0, estado e probabilidade sao
      COPIADOS (state/prob hold), como no protocolo de replay do artigo;
  A2. célula base = LSTM 2 camadas, H=64, D=31 — a MESMA classe do modelo
      implantado do artigo (nao favorece nem prejudica o baseline);
  A3. avaliacao com a MESMA convencao do artigo: deteccao m=3 frames
      consecutivos com p>=theta, FR<=0.05, TTDef = media-sobre-detectados
      + FR*T_ep (T_ep=10s), %Sup = fracao de frames com u_t=0;
  A4. dois pontos de operacao: (i) safe point — maior %Sup com FR<=0.05 na
      validacao (varrendo lambda_skip e theta); (ii) matched point — lambda
      cujo %Sup casa com o do entropy+kinematic gate (Tabela 19).
Seeds 42-45 (mesmas dos demais baselines da Tabela 19). GPU: ~10-20 min/seed
por lambda. Resumivel por (seed, lambda).

USO (GPU da Angela):
    cd TreinamentoNovo\\Experimentos-Artigo1-AdHoc\\skiprnn_baseline
    python train_skiprnn.py            # treina grade de lambdas, 4 seeds
    python train_skiprnn.py --eval     # so agrega/gera tabela
"""
import os, sys, json, glob, math, time
import numpy as np

DATA_DIR = os.path.join("..", "dados_sinteticos")
SEEDS    = [42, 43, 44, 45]
LAMBDAS  = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]   # grade de custo (paper usa ~1e-5..1e-3)
EPOCHS   = 40
LR       = 1e-3
H, LAYERS = 64, 2
T_EP     = 10.0
FPS      = 9.6
M_CONSEC = 3
THETAS   = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
OUT      = "results"

def load_seed(seed):
    import pandas as pd
    d = np.load(os.path.join(DATA_DIR, f"dataset_sintetico_seed{seed}.npz"))
    X = d["X"].astype(np.float32)
    # alvo de frame: curva limpa (mesma convencao do run_v29); fallback observada
    yfr = (d["y_frame_clean"] if "y_frame_clean" in d.files else d["y_frame"]).astype(np.float32)
    # criticidade de episodio: CSV de metadados (categoria == "Critico"),
    # exatamente como o run_v29 (linha 2492); npz["y_episode"] NAO e o rotulo binario
    meta = pd.read_csv(os.path.join(DATA_DIR, f"episodios_seed{seed}.csv"), sep=";")
    ccol = next(c for c in ("class_name", "categoria", "class") if c in meta.columns)
    yep = (meta[ccol].astype(str).values == "Critico").astype(np.int64)
    assert len(yep) == len(X) and yep.sum() > 0, "rotulos de episodio inconsistentes"
    # particao EXATA da campanha canonica (tr/va/te publicados no artefato)
    camp = os.path.join("..", "..", "tera-adhoc-artifact", "2-campaign-exp_20260519_055950", "data")
    p_tr = os.path.join(camp, f"indices_tr_seed{seed}.npy")
    p_va = os.path.join(camp, f"indices_va_seed{seed}.npy")
    p_te = os.path.join(camp, f"indices_te_seed{seed}.npy")
    if all(os.path.exists(p) for p in (p_tr, p_va, p_te)):
        idx_tr, idx_va, idx_te = np.load(p_tr), np.load(p_va), np.load(p_te)
        print(f"[ok] seed{seed}: particao tr/va/te publicada da campanha canonica.")
        return X, yep, yfr, np.asarray(idx_tr), np.asarray(idx_va), np.asarray(idx_te)
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X))
    idx_tr, idx_te = train_test_split(idx, test_size=0.15, random_state=seed, stratify=yep)
    print(f"[warn] seed{seed}: indices publicados nao encontrados; split re-derivado (documentar).")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(idx_tr)
    n_val = max(1, int(0.15 * len(perm)))
    return X, yep, yfr, perm[n_val:], perm[:n_val], np.asarray(idx_te)

def metrics(P, U, yfr, yep, theta):
    """P: probs (N,T) ja com hold; U: updates (N,T) em {0,1}.
    Convencao CANONICA do artigo: para FR/TTD/TTDef, a deteccao de um episodio
    critico so conta A PARTIR do onset (m=3 frames consecutivos p>=theta em
    t>=t0); disparos pre-onset NAO valem como deteccao. F1 episodico usa
    disparo em qualquer ponto (fire-anywhere), como discriminacao de episodio."""
    N, T = P.shape
    above = P >= theta
    crit = yep > 0
    onset = np.argmax(yfr >= 0.35, axis=1)   # frame_thr do pipeline (run_v29)

    def first_run(row, start):
        run = 0
        for t in range(start, T):
            run = run + 1 if row[t] else 0
            if run >= M_CONSEC:
                return t - M_CONSEC + 1
        return -1

    det_any = np.zeros(N, bool)
    det_post = np.zeros(N, bool); tdet = np.full(N, -1)
    for i in range(N):
        det_any[i] = first_run(above[i], 0) >= 0
        if crit[i]:
            td = first_run(above[i], int(onset[i]))
            det_post[i] = td >= 0; tdet[i] = td
    fr = float(np.mean(~det_post[crit])) if crit.any() else 0.0
    dts = [max(0.0, (tdet[i] - onset[i]) / FPS) for i in range(N) if crit[i] and det_post[i]]
    ttd = float(np.mean(dts)) if dts else 0.0
    ttdef = ttd + fr * T_EP
    f1_tp = float(np.sum(det_any & crit)); f1_fp = float(np.sum(det_any & ~crit)); f1_fn = float(np.sum(~det_any & crit))
    f1 = 2 * f1_tp / max(2 * f1_tp + f1_fp + f1_fn, 1e-9)
    return dict(FR=fr, TTDef=ttdef, SupPct=100.0 * float(1 - U.mean()), F1=f1)

def main():
    import torch, torch.nn as nn
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)

    class SkipLSTM(nn.Module):
        def __init__(self, D=31, H=H, L=LAYERS):
            super().__init__()
            self.lstm = nn.LSTM(D, H, num_layers=L, batch_first=True)
            self.head = nn.Linear(H, 1)
            self.gate = nn.Linear(H, 1)   # Delta u(s_t) — paper: funcao do estado
        def forward(self, x):
            # loop explicito com gate binario straight-through (Campos et al., Eq. 1-4)
            B, T, D = x.shape
            hs = [torch.zeros(self.lstm.num_layers, B, H, device=x.device) for _ in range(2)]
            u_tilde = torch.ones(B, 1, device=x.device)     # u~_1 = 1 (primeiro frame atualiza)
            probs, us, us_st = [], [], []
            h_out = torch.zeros(B, H, device=x.device)
            for t in range(T):
                u = (u_tilde >= 0.5).float()
                u_st = u + (u_tilde - u_tilde.detach())     # straight-through
                out, (h_new, c_new) = self.lstm(x[:, t:t+1, :], (hs[0], hs[1]))
                h_cand = out[:, 0, :]
                h_out = u_st * h_cand + (1 - u_st) * h_out                     # hold
                hs[0] = u_st.unsqueeze(0) * h_new + (1 - u_st).unsqueeze(0) * hs[0]
                hs[1] = u_st.unsqueeze(0) * c_new + (1 - u_st).unsqueeze(0) * hs[1]
                p = torch.sigmoid(self.head(h_out))
                delta_u = torch.sigmoid(self.gate(h_out))
                u_tilde = u_st * delta_u + (1 - u_st) * torch.clamp(u_tilde + torch.minimum(delta_u, 1 - u_tilde), 0, 1)
                probs.append(p); us.append(u); us_st.append(u_st)
            # us: gate binario (reporte); us_st: com gradiente ST (custo de orcamento)
            return torch.cat(probs, 1), torch.cat(us, 1), torch.cat(us_st, 1)

    done_p = os.path.join(OUT, "done.json")
    done = json.load(open(done_p)) if os.path.exists(done_p) else {}
    rows = []
    for seed in SEEDS:
        X, yep, yfr, itr, iva, ite = load_seed(seed)
        Xt = torch.tensor(X, device=dev); Yt = torch.tensor(yfr, device=dev)
        for lam in LAMBDAS:
            key = f"{seed}_{lam}"
            ck = os.path.join(OUT, f"skiprnn_seed{seed}_lam{lam}.pt")
            torch.manual_seed(seed); np.random.seed(seed)
            model = SkipLSTM().to(dev)
            if key in done and os.path.exists(ck):
                model.load_state_dict(torch.load(ck, map_location=dev))
            elif "--eval" in sys.argv:
                # [auditoria] nunca avaliar um modelo nao treinado
                sys.exit(f"ERRO: --eval sem checkpoint para {key} ({ck}). Treine primeiro.")
            elif "--eval" not in sys.argv:
                opt = torch.optim.Adam(model.parameters(), lr=LR)
                bce = nn.BCELoss()
                t0 = time.time()
                for ep in range(EPOCHS):
                    perm = np.random.permutation(itr)
                    for b0 in range(0, len(perm), 32):
                        bi = perm[b0:b0+32]
                        p, u, u_st = model(Xt[bi])
                        # custo de orcamento no gate COM gradiente (straight-through),
                        # como em Campos et al.: L = L_task + lam * sum_t u_t
                        loss = bce(p, Yt[bi].clamp(0, 1)) + lam * u_st.mean() * u_st.shape[1]
                        opt.zero_grad(); loss.backward(); opt.step()
                torch.save(model.state_dict(), ck)
                done[key] = True; json.dump(done, open(done_p, "w"))
                with torch.no_grad():
                    _, u_chk, _ = model(Xt[iva])
                print(f"[train] seed{seed} lam={lam} em {time.time()-t0:.0f}s | %skip(val)={100*(1-u_chk.mean().item()):.1f}")
            model.eval()
            with torch.no_grad():
                for split, idx in [("val", iva), ("test", ite)]:
                    P, U, _ = model(Xt[idx]); P, U = P.cpu().numpy(), U.cpu().numpy()
                    for th in THETAS:
                        m = metrics(P, U, yfr[idx], yep[idx], th)
                        rows.append(dict(seed=seed, lam=lam, split=split, theta=th, **m))
    import csv
    with open(os.path.join(OUT, "skiprnn_grid.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
    # ponto seguro por seed: max SupPct s.a. FR<=0.05 na VAL; reporta TEST
    print("\n=== Skip-RNN: safe point (selecionado na validacao, reportado no teste) ===")
    agg = []
    for seed in SEEDS:
        best = None
        for r in rows:
            if r["seed"] == seed and r["split"] == "val" and r["FR"] <= 0.05:
                if best is None or r["SupPct"] > best["SupPct"]: best = r
        if best is None:
            print(f"seed{seed}: nenhum ponto com FR<=0.05"); continue
        te = [r for r in rows if r["seed"] == seed and r["split"] == "test"
              and r["lam"] == best["lam"] and r["theta"] == best["theta"]][0]
        agg.append(te)
        print(f"seed{seed}: lam={best['lam']} theta={best['theta']} -> "
              f"TEST FR={te['FR']:.3f} TTDef={te['TTDef']:.3f} %Sup={te['SupPct']:.1f} F1={te['F1']:.3f}")
    if agg:
        import statistics as st
        print(f"\nmean±sd: FR={st.mean([a['FR'] for a in agg]):.3f}±{st.stdev([a['FR'] for a in agg]):.3f} "
              f"TTDef={st.mean([a['TTDef'] for a in agg]):.3f}±{st.stdev([a['TTDef'] for a in agg]):.3f} "
              f"%Sup={st.mean([a['SupPct'] for a in agg]):.1f}±{st.stdev([a['SupPct'] for a in agg]):.1f}")
    print("\n[done] results/skiprnn_grid.csv — para o matched point, selecionar lambda cujo")
    print("       %Sup de teste case com o do entropy+kinematic gate (Tabela 19) por trajetoria.")

if __name__ == "__main__":
    main()
