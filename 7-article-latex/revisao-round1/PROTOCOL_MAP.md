# PROTOCOL_MAP — mapa de protocolos experimentais (espelho da Tabela 23 do artigo)

| Experimento (local) | Seeds | Treino/retraining | Checkpoint | Partição de teste | Limiar/regra de detecção | Scheduler | Budget matching | ECE | Métricas primárias |
|---|---|---|---|---|---|---|---|---|---|
| Fatorial principal (§4.3–4.7; Tabs. 7–10, 12; Figs. 7–10) | 42–53 | treinado por seed, pipeline TERA congelado | por seed | test 70/15/15 | episódio: média dos últimos k=6 frames vs thr_ep (VAL, F1 máx s.a. FR≤0,05); TTD θ=0,10, m=3 | MHEG τΔ/τ_H por seed | — | ECE_pipe | FR, TTDef, %Sup, F1 |
| Borderline / Movimento 6 (Tab. 13, Fig. 11) | 42 | modelos fatoriais congelados | idem | receitas borderline held-out | p̄_max (máx por episódio) e frações de ativação a θ=0,10 (descritor) | fixo | — | — | p̄_max, frações |
| Calibração pós-hoc (§4.8, Tab. 14, Figs. 12–13) | 42–45 | Baseline congelado + calibradores ajustados | idem | test | regras do pipeline; thresholds MHEG recalibrados por variante | MHEG recalibrado | — | ECE_cal | ECE, FR, TTDef, %Sup |
| Ablação do reference (§4.9, Tab. 15) | 42–45 | retreinado, protocolo simplificado (60 épocas, sem scheduler) | novo | própria | não comparável ao principal | — | — | — | contrastes internos |
| Ablação de mecanismo (§4.6, Tab. 11) | 42–53 | 5 braços retreinados sob protocolo de replicação (run_v29) | novos | run_v29 test | política fixa + MHEG do run_v29 (H em bits) | run_v29 | — | ECE_v29 | F1, FR, ECE, TTDef, %Sup |
| Campanha de estresse (§4.10, Tabs. 16–18) | 42–45 | trajetórias congeladas do protocolo de replicação | abl_A | 9 cenários × 360 eps | FS@onset, cobertura, miss | replay | compute-rate ±2% | — | FS@onset, coverage |
| Gating baselines (§4.11, Tab. 19 + refresh) | 42–45 | trajetórias congeladas | abl_A | replay test | onset do rótulo (y>0,35), θ=0,10, m=3; oracle no teste (declarado) | conf/smooth/mheg/refresh R∈{2,4,8} | matched suppression | — | FR, TTDef, %Sup |
| Skip-RNN (§4.11, Tab. 20) | 42–45 | Skip-RNN treinado end-to-end; AF-TOI(+MHEG) congelado | novos + abl_A | partições da campanha | protocolo harmonizado (idem acima) + PreOcc/NormOcc | gate aprendido vs MHEG | matched suppression | — | FR, TTDef, %Sup, F1, PreOcc, NormOcc |
| Warp temporal (§4.12, Tab. 21, Fig. 14) | 42–53 | modelos e thresholds dos 12 seeds CONGELADOS | fatorial | episódios TERA-Gen warpados (9 durações 0,21–5,21 s) | detecção do onset do rótulo; pontos de operação congelados | MHEG congelado | bisseção até casar compute (±2%, diagnóstico) | — | FR, TTDef, FS@onset |
| Deployment (§4.13, Tab. 22) | — | modelo implantado | student | — | — | — | — | — | params, MACs (1 MAC=2 FLOPs), latência GPU ref., %Sup |

Regras de leitura: valores comparáveis DENTRO de uma linha; nunca entre linhas.
ECE_pipe ≠ ECE_replay ≠ ECE_cal ≠ ECE_v29 (fluxos de probabilidade, partições e binning distintos).
