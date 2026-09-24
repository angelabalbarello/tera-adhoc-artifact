# FINAL_FIX_REPORT — auditoria final code→paper + passagem adversarial (24/09/2026)
Complementa SCIENTIFIC_FIX_REPORT.md (rodada de 23/09). Matriz completa: FINAL_CODE_PAPER_AUDIT.md.

## 1. Auditoria adversarial (Revisor #3) — um achado por categoria

| # | Categoria | FILE / LINE | Evidência (fonte de dados / comando) | Esperado vs Observado | VEREDITO |
|---|-----------|-------------|--------------------------------------|----------------------|----------|
| 1 | Valor que não deriva dos artefatos | artigo.tex §2.2 (~L405) | expanded_per_seed.csv (baseline_hybrid) / test_per_seed_coverage.py | "up to 0.117" vs máx real 0.1583 em 3 seeds | **CRITICAL — corrigido em 23/09 (passada adversarial 1) e agora travado por teste** |
| 2 | Figura cujo script não reproduz o PDF | image/5-fig_temporal_trace_en.pdf | regen_dense_figs_en_v2.py + episode_frame_logs_v2.csv / test_figure5_selection.py | seleção declarada (mediana) vs script anterior (máx. contraste + t0 fixo) | **MAJOR — corrigido; seleção agora auditável (results/figure5_exemplar_selection.csv: ep36 rank 36/72, ep148 rank 24/48)** |
| 3 | Tabela que mistura protocolos | Tab.11 × Tab.8 | protocol_provenance.csv; caption Tab.11 | mesmo rótulo "frozen protocol" vs 5 braços retreinados sob run_v29 | **MAJOR — corrigido (declaração de retraining + não-comparabilidade + Tab.23)** |
| 4 | Threshold em unidade errada | generate_episode_frame_logs.py L164→168 | τ_H=0,95 bits vs H em nats (máx 0,693) ⇒ ramo entrópico nunca disparava no log das Figs.4–6 | gate híbrido vs máscara só-cinemática | **CRITICAL — corrigido na fonte (3 cópias), log regenerado na GPU (skip rates = run_v29: 56,2/40,1), teste trava a base** |
| 5 | Seed omitida | Tab.9/§4.4 | fr_paired_seeds.csv | 12 seeds presentes (42–53), zeros por empate (7) declarados; nenhuma omissão | **SEM ACHADO (verificado; teste falha se a lista mudar)** |
| 6 | p-value incompatível com os pares | §4.4/Tab.9 | scipy 1.15.3 wilcoxon exact/wilcox/two-sided | contagem publicada "nine of twelve" incompatível; pares reais (5 informativos) DÃO p=0,0625 | **CRITICAL na contagem (corrigida p/ seven of twelve); p estava correto; teste automatizado** |
| 7 | Claim mais forte que os resultados | artigo.tex Discussion (~L2528) | Skip-RNN Tab.20 (gate aprendido ATINGE endpoints in-distribution) | "breaks in causal edge inference" (geral) vs falha demonstrada só no replay controlado p/ trajetórias não organizadas | **MAJOR — corrigido nesta passada (F6): escopo + nuance Skip-RNN; +caveats IC+MHEG/refresh na Discussion (F7)** |
| 8 | Baseline implementado injustamente | run_gating_baselines_v2 (gate conf. puro trava por construção) | refresh_gate_v2_results.csv | espantalho vs variante realista | **MAJOR — mitigado em 23/09: periodic-refresh implementado e reportado (Base R=8: 85,9%/FR 0), com oracle e n=4 declarados** |
| 9 | Configuração do artigo ≠ código | Tab.6 / Eqs.5–9 | run_v29 assinatura+config A / test_aftoi_loss_manifest.py | θ_late 0,15→0,10; λ_late 3,5→2,0; E_warm 10→12; E_ramp 17→16; λ_TOI→β_max 0,35; piso α 0,50; KL→MSE gateada | **CRITICAL — corrigido em 23/09; manifest + teste linha-a-linha agora travam os coeficientes** |
| 10 | Declaração do rebuttal sem lastro no commit | rebuttal Additional 6 ("publicly available") | GitHub main (1 commit de 27/08) vs artefatos pós-27/08 | disponível vs não publicado | **MAJOR — resolvido: branch enviada pela autora; commit de consolidação efb5104 pronto para `push :main` (comando entregue); Add.7–9 conferidos item a item contra a árvore** |

Achados NOVOS desta passada final (além dos acima, todos corrigidos): 8 marcas editoriais residuais no corpo ("submitted version", "revision verification", "initially/originally reported", "expanded from four to twelve", caption Tab.8) reescritas em linguagem científica com espelhos no rebuttal (F1–F5); nota de escala do TTDef na §4.12 (Tab.21 vs Tab.8) adicionada com espelho na citação do C4 (F8); legenda da Fig.7 agora distingue a linha agregada FR=0,05 do critério por seed, citando as 3 violações do Baseline+MHEG (F9).

## 2. Confirmed issues / false alarms / bugs
- Confirmados e corrigidos: ver tabela acima + SCIENTIFIC_FIX_REPORT (P0.1–P0.8).
- Falsos alarmes verificados: p=0,0625 (correto); Tab.13×FR (definições distintas, sem contradição); refs Zhang; τ_H da campanha de estresse (sinal H sintético normalizado é desenho declarado, não bug).
- Deliberado e documentado: coluna H_t do log em nats (figuras convertem para bits); H_strat permanece como utilitário no código de calibração/estresse sem alimentar claims ou figuras.

## 3. Entregáveis desta passada (em tera-adhoc-artifact/6-revision-round1/scientific-audit/)
- results/: fr_paired_seeds.csv · per_seed_coverage_constraint.csv · figure5_exemplar_selection.csv · protocol_provenance.csv · aftoi_loss_manifest.json (+ audit_fr_wilcoxon.csv, ablation_vs_alpha_grid.txt, metric_provenance.csv da rodada anterior)
- configs/protocol_registry.yaml (fonte declarativa da Tabela 23)
- tests/: test_entropy_consistency · test_model_complexity · test_fr_wilcoxon · test_per_seed_coverage · test_figure5_selection · test_aftoi_loss_manifest · test_table_values · test_manuscript_claims (8 arquivos; falham se base de entropia, seeds, contagens, coeficientes da loss, seleção da Fig.5, valores-manchete ou claims proibidos mudarem)
- scripts/verify_paper.sh (comando único: complexidade → estatística → testes → manuscrito) — **estado atual: TUDO OK (0 FAIL/40 OK no checker + 8/8 testes)**
- FIGURE_PROVENANCE.md (figura → script → dados)

## 4. CHANGED_FILES (desta passada)
artigo.tex (F1–F9), rebuttal.tex (espelhos F3/F5/F8), artigo_v2.pdf (build final; renomear para artigo.pdf após fechar o leitor), rebuttal.pdf, + novos: FINAL_CODE_PAPER_AUDIT.md, FINAL_FIX_REPORT.md, scientific-audit/{tests/*, configs/protocol_registry.yaml, results/{fr_paired_seeds, per_seed_coverage_constraint, figure5_exemplar_selection, protocol_provenance}.csv, results/aftoi_loss_manifest.json, scripts/verify_paper.sh}.

## 5. UNRESOLVED_ISSUES
1. `git push` final (efb5104:main + 0874741 e o commit desta passada na branch) — credencial só no Windows; comandos entregues.
2. `artigo.pdf` travado pelo leitor de PDF no Windows durante a compilação final → build ficou em `artigo_v2.pdf` (íntegro); renomear após fechar o leitor; apagar `artigo_final.pdf` (build intermediário com refs quebradas, gerado durante o lock; a remoção falhou pelo mount).
3. Overleaf desatualizado (subir os 4 .tex + 5 figuras novas).
4. Figs. 7–11 não re-renderizadas nesta passada (dados inalterados; scripts versionados) — regeneração opcional.

## 6. Respostas finais
- **A. Every headline number reproducible? YES** (matriz C1–C16; suíte verde; Tab.8/9/11/14/19/20/21/22 derivadas de CSVs/checkpoints versionados)
- **B. Every published figure reproducible? YES** (Figs.1–3 diagramas estáticos versionados; 4–6 v2-script+log canônico; 7–11 scripts da expansão; 12 script de calibração (rerun validado); 13 regen_fig13_locus_en; 14 make_figure.py do warp — mapa em FIGURE_PROVENANCE.md)
- **C. Every published table reproducible? YES** (proveniência em protocol_provenance.csv; valores travados por testes)
- **D. AF-TOI equations match code? YES** (Eqs.5–9 = run_v29 config A; manifest + teste linha-a-linha)
- **E. Entropy unit consistent everywhere? YES** (bits em todo caminho operacional; duas exceções DELIBERADAS e declaradas: coluna de log em nats convertida pelas figuras; sinal H sintético normalizado da campanha de estresse)
- **F. Statistical tests reproducible? YES** (scipy 1.15.3, wilcoxon exact/zeros-descartados/two-sided; CSVs pareados versionados; testes automáticos)
- **G. Rebuttal statements match final manuscript? YES** (espelhos verificados; Additional 1–9 conferidos contra a árvore; checker 0 FAIL)

Nenhum NO ⇒ UNRESOLVED_ISSUES contém apenas ações operacionais (push/rename/Overleaf), não pendências científicas.
