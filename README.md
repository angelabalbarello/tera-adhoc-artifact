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

**Gerador de dados sintéticos (TERA-Gen):** `synthetic_driver_risk_v7.py` é o
gerador propriamente dito — receitas, dinâmicas calibradas e `quick_generate()`;
seu hash (SHA-256/16 = `65bbec4a23050639`) é idêntico ao registrado nos manifests
das campanhas. `tera_pipeline/dataset/tera_gen.py` **não gera dados**: é o wrapper
do pipeline que chama o gerador e materializa os arquivos consumidos pelo
treino/avaliação — verificação do hash do gerador, cache do `.npz`, splits
determinísticos 70/15/15 estratificados por receita, binarização dos rótulos
(`y_fr = (y_frame_clean > 0.35)·y_ep`), flag de progressivo e
`dataset_metadata_seed{N}.json`. Os dois são necessários para reprodução exata.

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

### Notas de rastreabilidade

1. A campanha de 9 cenários da subseção de robustez (FS@onset e orçamento
   pareado) está em `8-stress-campaign/`, com scripts, resultados e testes.
   A re-execução completa reproduz os 16 CSVs byte a byte (validado em
   31/08/2026); os episódios são gerados deterministicamente pelos próprios
   scripts, sem dependência de dados externos.
2. A cópia "viva" do `tera_pipeline` em `FGCS/Arquitetura/` **divergiu após a
   campanha publicada** (`tera_utils.py` modificado em 25/06, `tera_figures.py`
   em 02/06). Este repositório contém a **cópia congelada** (estado ≤ 19/05,
   dia da campanha), que é a que deve ser usada em qualquer re-execução.

## 2. O que foi criado/modificado para a revisão (round 1)

| Artefato | Status | O que é |
|---|---|---|
| `6-revision-round1/baselines_novos/run_gating_baselines.py` | **novo** (27/08) | baselines pedidos pelo R3.5: gate por confiança e gate por entropia suavizada (w=5), replay sobre trajetórias armazenadas, seeds 42–45, limiares oráculo favoráveis aos baselines |
| `6-revision-round1/baselines_novos/run_matched_sup.py` + `matched_sup.json` | **novo** (27/08) | ponto de operação com supressão pareada (%Sup igual ao gate entrópico+cinemático) |
| `6-revision-round1/baselines_novos/gating_baselines_results.csv`, `summary.json` | **novo** | resultados dos baselines acima |
| `6-revision-round1/seed-expansion-tera/run_seed_expansion_tera.py` | **novo — launcher correto** | expansão 4→12 seeds no TERA Pipeline congelado, com verificação de hashes antes de executar |
| `6-revision-round1/seed-expansion-tera/check_reproduction_42_45.py` | **novo** | validação crítica: seeds 42–45 regeneradas devem reproduzir a campanha publicada antes de qualquer integração |
| `6-revision-round1/seed-expansion-tera/stats_12seeds.py` | **novo** | agrega as 12 seeds, Wilcoxon exato + Cliff's δ, comparação n=4 × n=12, observações individuais — **não altera o artigo** |
| `6-revision-round1/seed-expansion-tera/regen_figs_12seeds_en.py`, `regen_dense_figs_en.py` | **novos** | regeneram as figuras do artigo (multiseed/trade-off em 12 seeds; figuras densas com fontes maiores, R2.6) |
| `6-revision-round1/seed-expansion-tera/df_episodios_12seeds.csv` | **novo** | TTDef por episódio das 12 seeds (validado contra o CSV canônico), fonte das Figs. 8–9 |

### Protocolo congelado da expansão de seeds

Seeds **42–53 (n=12), fixadas antes de qualquer p-valor**. O n não será alterado em
função do resultado; todas as 12 seeds serão reportadas. A expansão responde ao
poder estatístico (R2.5); **não** substitui a questão de generalização temporal
(durações de onset, R3.4), que é dimensão ortogonal e será decidida separadamente.
Os resultados n=12 não autorizam conclusões sobre direção naturalística — a
limitação de validade externa permanece.

## 2b. Rodadas de auditoria da revisão (v5–v9.4) e diagnóstico pós-hoc

Após a integração inicial da revisão, o manuscrito passou por rodadas de auditoria
científica dirigida (v5–v9.2). Material público correspondente:

| Pasta/arquivo | Conteúdo |
| --- | --- |
| `6-revision-round1/scientific-audit/` | scripts de verificação (leitor de checkpoint sem torch, reprodução seed 42, semântica de inferência), testes-trava do manuscrito (21 testes), CSVs de reconciliação (Tab. 13/FR), proveniência de figuras |
| `6-revision-round1/incremental-campaign/diagnostic-windowed-S2/` | **diagnóstico exploratório citado na §5 (quinta limitação) e na §4.10 do artigo v9.4**: perda do aviso cinemático no gate da implementação por janelas (protocolo pré-registrado, código numpy sem torch, resultados por seed, 96 logs, README_REPO.md e manifesto sha256) |
| `6-revision-round1/incremental-campaign/evidence_diagnostic_windowed_S2.zip` | pacote do diagnóstico para o suplemento da submissão (manifesto sha256 verificado 101/101; os insumos — checkpoints, arrays de teste e calibração — estão neste repositório em `1-pipeline-tera/results/exp_seed12_round1/`) |
| `1-pipeline-tera/results/exp_seed12_round1/data/` | arrays de teste `{X,y_fr,y_ep}_te_seed{42..53}.npy` (insumos do diagnóstico; ~23 MB) |
| `7-article-latex/revisao-round1/` | fontes LaTeX finais da revisão: `artigo.tex`, `rebuttal.tex`, `robustness_section.tex`, macros/linhas de tabela, figuras (incl. `image/fig3_mheg_componentes.tex` e `image/2-fig_tera_pipeline_final_v2.png`, corrigidas para a semântica avaliada) e PDFs de referência (`artigo_round1_final.pdf`, `rebuttal_round1_final.pdf`) |
| `7-article-latex/revisao-round1/audit-trail/` | trilha de auditoria: matrizes v6–v9, diffs entre versões, relatórios de verificação (inclui `VERIFICACAO_CODIGO_SEMANTICA.md`, que amarra código↔resultados↔texto) |
| `7-article-latex/revisao-round1/_superseded/` | builds e fontes intermediários preservados para histórico (não usar) |

Correções relevantes dessas rodadas (todas declaradas no rebuttal): descrição do
Algoritmo 1/§3.4/Figs. 2–3 alinhada à semântica avaliada (janela causal W=96
recomputada; a variante single-step de estado persistente é declarada não validada);
Tab. 13/Fig. 11 regeneradas após correção do logger; nomenclatura das variantes de
ECE; unidade de entropia unificada em bits.

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
                                  expansão de seeds TERA)
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
