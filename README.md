# Artefatos experimentais — "Trustworthy Selective Computation for Resource-Constrained Vehicular Edge Networks"

Submissão **Ad Hoc Networks (ADHOC-D-26-02099)** — ecossistema TERA (TERA-Gen + AF-TOI + MHEG).
Este repositório consolida, em um único lugar, (a) **todas as fontes que geraram os
resultados do artigo submetido** e (b) **os artefatos criados/modificados para os novos
experimentos da revisão (round 1)**. Consolidado em 2026-08-27.

## 1. Rastreabilidade: cada `\input` do artigo → gerador → campanha

| Entrada no `artigo.tex` | Conteúdo | Gerado por | Dados/campanha | Pasta aqui |
|---|---|---|---|---|
| `paper_table2_rows.tex` | Table 8 (estudo fatorial 2×2, 4 seeds) | TERA Pipeline v1.0 (`run_experiment.py` + pacote `tera_pipeline`) | `exp_20260519_055950` (seeds 42–45, config_hash `d5327a24d1d2f7f9`, generator_hash `65bbec4a23050639`) | `1-pipeline-tera/`, `2-campaign-exp_20260519_055950/` |
| `paper_metrics_macros.tex` | macros numéricas do estudo principal (FR, TTDef, %Sup etc.) | idem | idem | idem |
| `paper_table_stratified_rows.tex` | estratificação progressivo/abrupto | idem | idem | idem |
| `paper_table_seed_coverage.tex` | cobertura por seed | idem | idem | idem |
| Tabela 9 (Wilcoxon/Cliff, valores no corpo do texto) | estatística pareada n=4 (p-piso 0,0625) | derivada de `results_all_seeds.csv` | idem | `2-campaign.../metrics/` |
| `paper_calibration_macros.tex` | experimento de calibração pós-hoc (TS/beta) | `tera_posthoc_calibration_experiment.py` | mesma campanha + `calibration/` | `3-calibration-posthoc/` |
| `paper_macros_ablacao_teacher.tex`, `table_teacher_ablation.tex` | ablação arquitetural do reference não-causal (BiLSTM × Transformer) — protocolo simplificado, **não comparável** ao fatorial principal | `run_ablation_teacher_study.py` (e `run_ablation_v8.py`) | `resultados_ablacao_teacher/`, `resultados_exp_ablacao_v8/` | `4-teacher-ablation/` |
| `robustness_section.tex` | campanha de robustez: 9 cenários de stress, orçamento pareado, FS@onset (protocolo replay, **escala distinta** do fatorial — ver nota de escopo no próprio arquivo) | protocolo replay sobre trajetórias do `run_v29_ablacao_ttdef_ajuste_gatting.py` (abl_A) | frame_probs .npy seeds 42–45 em `Inferencia/` | `5-robustness-replay-runv29/` |
| Figuras 4–11 (entropia, traces, heatmap, tradeoff, TTDef, multiseed, prob_dist) | figuras do estudo principal | `tera_pipeline/figures/tera_figures.py` + `generate_episode_frame_logs.py` / `generate_borderline_frame_logs.py` | `exp_20260519_055950/figures` + logs | `2-campaign.../`, `5-robustness.../Inferencia/`, `7-article-latex/figuras-fonte/` |

**Gerador de dados sintéticos (TERA-Gen):** `synthetic_driver_risk_v7.py`
(SHA-256/16 = `65bbec4a23050639`, idêntico ao da campanha publicada) e o wrapper
`tera_pipeline/dataset/tera_gen.py`.

### Nota sobre a preparação do código para publicação

Antes da publicação deste repositório, comentários, docstrings e mensagens de
console dos scripts passaram por uma revisão de estilo (remoção de molduras
decorativas e padronização de avisos). A revisão foi verificada
automaticamente: a árvore sintática de cada arquivo é idêntica à da versão
que executou as campanhas, exceto por literais de string de saída, e todos os
módulos compilam e importam normalmente. `synthetic_driver_risk_v7.py` e
`configs/experiment_config.yaml` não foram tocados — seus hashes
(`65bbec4a23050639` e `d5327a24d1d2f7f9`) continuam idênticos aos registrados
nos manifests das campanhas.

### Lacunas de rastreabilidade conhecidas (registradas, não resolvidas)

1. **O script que computou os números da campanha de 9 cenários** (tabela de
   robustez com FS@onset e orçamento pareado) **não foi localizado em disco** —
   apenas o `robustness_section.tex` final com os valores. As trajetórias-fonte
   (frame_probs por seed) estão em `5-robustness-replay-runv29/Inferencia/`.
2. A cópia "viva" do `tera_pipeline` em `FGCS/Arquitetura/` **divergiu após a
   campanha publicada** (`tera_utils.py` modificado em 25/06, `tera_figures.py`
   em 02/06). Este repositório contém a **cópia congelada** (estado ≤ 19/05,
   dia da campanha), que é a que deve ser usada em qualquer re-execução.

## 2. O que está sendo criado/modificado para a revisão (round 1)

| Artefato | Status | O que é |
|---|---|---|
| `6-revision-round1/baselines_novos/run_gating_baselines.py` | **novo** (27/08) | baselines pedidos pelo R3.5: gate por confiança e gate por entropia suavizada (w=5), replay sobre trajetórias armazenadas, seeds 42–45, limiares oráculo favoráveis aos baselines |
| `6-revision-round1/baselines_novos/run_matched_sup.py` + `matched_sup.json` | **novo** (27/08) | ponto de operação com supressão pareada (%Sup igual ao gate entrópico+cinemático) |
| `6-revision-round1/baselines_novos/gating_baselines_results.csv`, `summary.json` | **novo** | resultados dos baselines acima |
| `6-revision-round1/seed-expansion-deprecated-runv29/` | **obsoleto — não usar** | primeiro launcher da expansão 4→12 seeds; executava o `run_v29` (pipeline **errado**: não é o que gerou a Table 8). Mantido para registro. Interrompido em 27/08 após a seed 42. |
| `6-revision-round1/seed-expansion-tera/run_seed_expansion_tera.py` | **novo — launcher correto** | expansão 4→12 seeds no TERA Pipeline congelado, com verificação de hashes antes de executar |
| `6-revision-round1/seed-expansion-tera/check_reproduction_42_45.py` | **novo** | validação crítica: seeds 42–45 regeneradas devem reproduzir a campanha publicada antes de qualquer integração |
| `6-revision-round1/seed-expansion-tera/stats_12seeds.py` | **novo** | agrega as 12 seeds, Wilcoxon exato + Cliff's δ, comparação n=4 × n=12, observações individuais — **não altera o artigo** |
| `6-revision-round1/seed-expansion-tera/regen_figs_12seeds_en.py`, `regen_dense_figs_en.py` | **novos** | regeneram as figuras do artigo (multiseed/trade-off em 12 seeds; figuras densas com fontes maiores, R2.6) |
| `6-revision-round1/seed-expansion-tera/df_episodios_12seeds.csv` | **novo** | TTDef por episódio das 12 seeds (validado contra o CSV canônico), fonte das Figs. 8–9 |
| `7-article-latex/revisao-round1/artigo.tex`, `rebuttal.tex` | **revisão integrada** (31/08) | n=12 aplicado às Tabelas 8–10/12 e Figuras 7–10, mudanças em vermelho com referência de página no rebuttal; pendem itens da autora (nome do editor, decisão R3.11, Data statement, Highlights) |

### Protocolo congelado da expansão de seeds

Seeds **42–53 (n=12), fixadas antes de qualquer p-valor**. O n não será alterado em
função do resultado; todas as 12 seeds serão reportadas. A expansão responde ao
poder estatístico (R2.5); **não** substitui a questão de generalização temporal
(durações de onset, R3.4), que é dimensão ortogonal e será decidida separadamente.
Os resultados n=12 não autorizam conclusões sobre direção naturalística — a
limitação de validade externa permanece.

## 3. Como re-executar

```
# 12 seeds no pipeline congelado (~1,5–2 h em RTX 3050 Ti)
cd 6-revision-round1/seed-expansion-tera
python run_seed_expansion_tera.py

# validação obrigatória antes de qualquer uso dos resultados
python check_reproduction_42_45.py

# relatórios estatísticos (não tocam no artigo)
python stats_12seeds.py
```

Requisitos: Python 3.10+, torch (CUDA), numpy, pandas, scikit-learn, matplotlib,
scipy, pyyaml.

## 4. Estrutura

```
1-pipeline-tera/                  pipeline congelado (run_experiment.py, tera_pipeline/,
                                  configs/experiment_config.yaml, synthetic_driver_risk_v7.py)
2-campaign-exp_20260519_055950/   campanha publicada completa (manifest, metrics, models,
                                  data, calibration, latex, figures, logs)
3-calibration-posthoc/            experimento de calibração pós-hoc + macros
4-teacher-ablation/               ablação BiLSTM × Transformer + resultados
5-robustness-replay-runv29/       run_v29, gerador, dados sintéticos, frame_probs,
                                  logs de frames e robustness_section.tex
6-revision-round1/                artefatos novos da revisão (baselines R3.5,
                                  expansão de seeds TERA, launcher obsoleto p/ registro)
7-article-latex/                  fontes LaTeX:
                                  00-versao-SUBMETIDA/ — a versão que os revisores leram
                                    (artigo.tex 2028 linhas, do zip ___submited.zip; ver
                                    MAPA_DE_VERSOES.md, artigo_ORIGINAL_submetido.pdf e
                                    ADHOC-D-26-02099_versao_oficial_EM.pdf)
                                  versao-final-segura/ — variante pós-submissão (jul/2026,
                                    ajustes de fontes/ORCID; 2063 linhas — NÃO é a submetida)
                                  revisao-round1/ — revisão em andamento + rebuttal
                                  figuras-fonte/ — fontes das figuras
docs/                             notas de planejamento experimental
```
