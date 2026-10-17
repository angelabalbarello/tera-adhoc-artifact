# Diagnóstico exploratório: perda do aviso cinemático no gate (implementação por janelas)

Diagnóstico pós-hoc **exploratório** citado na quinta limitação (§5) e na §4.10
do artigo revisado (v9.4): com pesos e limiares congelados da campanha
`exp_seed12_round1`, zerar o delta cinemático ENTREGUE AO GATE a partir da
segunda decisão (entradas da LSTM intactas) produz supressão persistente e
FR=1,000 nas 24 células (12 seeds × {Baseline, AF-TOI} + MHEG). A condição de
controle (A) reproduz os valores publicados (diff de FR 0,000000).

## Conteúdo (originais da execução de 11/10/2026, preservados sem alteração)

- `DIAGNOSTIC_PREREG.md` — protocolo pré-registrado e sequência de execução;
- `DIAGNOSTIC_REPORT.md` — relatório;
- `windowed_s2.py` — código (implementação fiel de
  `tera_infer.py::infer_stream_hybrid` em numpy, sem torch);
- `s2_windowed_results.csv` — resultados por seed/configuração/condição;
- `logs_ep_*.csv` (48) — logs por episódio; `logs_gate_*.csv.gz` (48) —
  decisões do gate por frame;
- `HASHES_DIAGNOSTIC_FOLDER.txt` — SHA-256 destes 100 arquivos (caminhos
  relativos), gerado na organização do repositório (17/10/2026) e conferido
  contra o manifesto do pacote `evidence_diagnostic_windowed_S2.zip`
  (100/100 idênticos).

Dependência local: `../incremental_engine.py` (leitor de checkpoint sem torch
+ LSTM numpy, validado contra os valores publicados).

## Insumos (todos NO REPOSITÓRIO, caminhos relativos à raiz)

`1-pipeline-tera/results/exp_seed12_round1/`:
`models/{baseline,student}_seed{42..53}.pt`,
`data/{X,y_fr,y_ep}_te_seed{42..53}.npy`,
`calibration/calibration_summary.csv`.
O script resolve esses caminhos sozinho a partir da sua posição na árvore.

## Execução e saídas esperadas

    python3 windowed_s2.py 42 43 44 45 46 47 48 49 50 51 52 53

Dependências: Python ≥3.10, numpy, pandas (sem GPU, sem torch).
Saídas: `s2_windowed_results.csv` (48 linhas) e os 96 logs. Resultados
esperados: condição A reproduz FR/SupPct publicados; condição B dá FR=1,0 e
SupPct ≈98,7–99,0% em todas as células.

Verificação de integridade: `sha256sum -c HASHES_DIAGNOSTIC_FOLDER.txt`

## Estado de verificação (17/10/2026)

- Integridade dos arquivos: VERIFICADA (manifesto acima; zip 101/101).
- Reprodução: realizada na execução original (11/10/2026; condição A conferida
  contra os valores publicados) — NÃO repetida nesta organização, que é
  documental.
