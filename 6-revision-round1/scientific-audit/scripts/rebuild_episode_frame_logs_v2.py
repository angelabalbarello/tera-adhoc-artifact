#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstroi episode_frame_logs_v2.csv a partir do log v1 (fix da auditoria 23/09).

Contexto: o gerador original aplicava tau_h=0.95 (calibrado em BITS, Eq. 11)
sobre entropia em NATS, de modo que o ramo entropico do gate nunca disparava e
a mascara de supressao era puramente cinematica (identica entre configs).
Como o ramo cinematico independe do modelo, a mascara antiga fornece exatamente
kin_update(t); o replay hibrido correto e reconstruido deterministicamente
sobre as trajetorias FIXAS armazenadas (mesmo protocolo de replay da Tabela 19):

    update(t) = (t==0) or kin_update(t) or H_bits(buffer) >= 0.95

Verificacao independente: regenerar o log via modelos (generate_episode_frame_logs.py
corrigido, GPU) deve reproduzir este CSV a menos de diferencas de janela de prefixo.
"""
import pandas as pd, numpy as np, math, pathlib

HERE = pathlib.Path(__file__).resolve().parent
L = HERE.parent.parent.parent / "5-robustness-replay-runv29" / "Inferencia"
TAU_H = 0.95  # bits

def Hb(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

def Hn(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))

def main():
    df = pd.read_csv(L / "episode_frame_logs.csv")
    out_rows = []
    for cfg_h, cfg_f in [("baseline_hybrid", "baseline_fixed"),
                         ("afkd_hybrid", "afkd_fixed")]:
        h = df[df.config == cfg_h]
        f = df[df.config == cfg_f]
        fixed_p = {(s, e): g.sort_values("t").p_t.values
                   for (s, e), g in f.groupby(["seed", "episode_id"])}
        for (s, e), g in h.groupby(["seed", "episode_id"]):
            g = g.sort_values("t").copy()
            pf = fixed_p[(s, e)]
            kin_update = (g.is_suppressed.values == 0)
            newp = np.empty(len(g)); sup = np.zeros(len(g), int); last = None
            for t in range(len(g)):
                if (t == 0) or kin_update[t] or (Hb(last) >= TAU_H):
                    last = pf[t]
                else:
                    sup[t] = 1
                newp[t] = last
            g["p_t"] = newp
            g["is_suppressed"] = np.where(g.t0_frame < 0, -1, sup)
            g["H_t"] = [Hn(p) for p in newp]   # coluna mantida em nats (figuras convertem)
            out_rows.append(g)
    new = pd.concat([df[df.config.isin(["baseline_fixed", "afkd_fixed"])]] + out_rows,
                    ignore_index=True)
    new.to_csv(L / "episode_frame_logs_v2.csv", index=False)
    print("episode_frame_logs_v2.csv reconstruido:", len(new), "linhas")

if __name__ == "__main__":
    main()
