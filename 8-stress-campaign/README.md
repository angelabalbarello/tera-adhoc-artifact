# Campanha de stress — controles negativos e comparação cost-matched (Seção 4.10)

Campanha controlada de 9 cenários onset/ruído (360 episódios por cenário, seeds
42–45) que sustenta a subseção *Negative Controls and Cost-Matched Robustness*
do artigo (`7-article-latex/robustness_section.tex`). Os episódios são gerados
deterministicamente pelos próprios scripts (`src/article1_neg.py`, seeds fixas
via `src/lab.py`); a campanha não depende de nenhum outro dado do repositório.

## Como reproduzir

```
python experiments/run_article1_E1.py
python experiments/run_article1_E1b.py
python experiments/run_article1_negative_controls.py
```

Requisitos: numpy, pandas, matplotlib. Os scripts gravam em `results/`,
`tables/` e `figures/` na raiz deste diretório.

## Rastreabilidade (script → tabela publicada)

| Saída | Script | Uso no artigo |
|---|---|---|
| `article1_A1_T4_statistical_tests.csv` | `run_article1_E1.py` | ablação por canal (Cliff's δ = −0,5 e 0,0, parágrafo final da 4.10) |
| `article1_A1_T5_negative_controls.csv` | `run_article1_negative_controls.py` | sweep dos controles negativos (56,7% lagged; 11,5% vs 33–46% no kinematic_signal_missing; sanity ≈ 0) |
| `article1_A1_T6_cost_matched_comparison.csv` | `run_article1_negative_controls.py` | Tabela cost-matched (±2% de orçamento) |
| `article1_A1_T8_entropy_lag_sensitivity.csv` | `run_article1_negative_controls.py` | Tabela de sensibilidade ao lag entrópico |
| `article1_scenarios_metadata.csv` | `run_article1_negative_controls.py` | Tabela de definição dos 9 cenários |

O cenário `rare_abrupt_critical_events` aparece abreviado como
`rare_abrupt_critical` no manuscrito.

## Validação

Re-execução completa dos três scripts em 31/08/2026 reproduziu os 16 CSVs de
`results/` e `tables/` **byte a byte** contra as cópias originais de
02/07/2026, e todos os valores citados na 4.10 foram conferidos célula a
célula contra `robustness_section.tex`. `tests/test_article1_negative_controls.py`
cobre as invariantes do protocolo (orçamento pareado ±2%, janela de onset,
sanity do cenário sem onset).

## Nota de escopo

Como registrado no próprio manuscrito, as magnitudes desta campanha (compute
rate, TTDef em escala de stress) não são comparáveis aos agregados do fatorial
2×2 da Seção 4.4; o protocolo ranqueia segurança a custo fixo, não re-estima
eficiência.
