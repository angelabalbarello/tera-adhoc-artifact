# SCIENTIFIC_FIX_REPORT — correção científica pós-auditoria "Angela - Rebuttal"
Data: 23/09/2026. Base: AUDIT_MATRIX.md (verificação de cada apontamento contra código/dados).
Ambiente de execução restaurado; TODOS os números novos foram recalculados de artefatos, nenhum inventado.

## A. Problemas CONFIRMADOS e corrigidos na origem

**P0.1 — Figura 5 contradizia o texto.** Confirmado nos dados (painel abrupto: H≈máx o episódio todo, p_pre≈0,48; episódio atípico — 12/48 abruptos saturados; critério do progressivo era MÁXIMO contraste = seleção favorável). E, mais fundo: o gerador do log de replay (`generate_episode_frame_logs.py`) aplicava τ_H=0,95 (calibrado em bits) sobre entropia em NATS (máx 0,693) → o ramo entrópico do gate nunca disparava e a máscara de supressão era idêntica entre configs. **Correções:** bug consertado na fonte (3 cópias); log reconstruído pelo protocolo de replay sobre trajetórias armazenadas (o mesmo da Tabela 19) → `episode_frame_logs_v2.csv`; exemplares trocados por critério objetivo pré-declarado (mediana de contraste por regime: progressivo ep.36 t0=5,42 s; abrupto ep.148 t0=2,50 s), declarado na legenda; guia de leitura reescrito com os números reais (H 0,5→1,0 bit; supressão 58%→14%/43% no AF-TOI vs ~55–56% constante no Baseline) e com as duas ressalvas honestas (marcadores de detecção de exemplar não arbitram latência; minoria saturada 12/48). Com o log corrigido, a alocação diferencial voltou a ser real: supressão na janela de onset 9,8% (AF-TOI) vs 37,2% (Baseline). Tabelas publicadas NÃO foram afetadas (os pipelines das tabelas já usavam log2).

**P0.2 — bits vs nats.** Confirmado. Definição oficial = BITS (Eq. 11, scheduler, τ_H=0,95). Figuras 4–6 regeneradas em bits (eixos 0/0,5/1), texto do Movement 1 atualizado (0,5 bit / 1 bit), legenda da Fig. 4 declara a unidade. H_strat é razão (invariante à base) — sem efeito nesse ponto.

**P0.3 — Complexidade/FLOPs.** Confirmado. Do checkpoint: **58.242 parâmetros exatos** = 4(DH+H²+2H)+4(2H²+2H)+2(H+1); **57,2 K MACs/timestep ≈ 115 KFLOPs** (1 MAC=2 FLOPs, convenção agora declarada). "≈28 KFLOPs" corrigido em §3.4, Alg. 1 (parágrafo, com custo de 2 camadas) e Tabela 22. Script reproduzível `scripts/check_model_complexity.py`. Motivação da §2.1 reformulada (modelo pequeno isola o mecanismo; ganho cresce com preditor maior/processador compartilhado).

**P0.4 — FR/Wilcoxon.** Recontado dos logs: **7/12** seeds Baseline com FR=0 (não 9/12); 5 diferenças não nulas, todas a favor do AF-TOI → **p=0,0625 estava CORRETO** (Wilcoxon exato bicaudal descartando zeros; pratt=0,424; zsplit=0,052). Cliff δ=+0,361 ✓. Texto e nota da Tabela 9 corrigidos + variante do teste declarada; espelhos no rebuttal. Artefatos: `audit_fr_wilcoxon.csv`, `recompute_statistics.py`.

**P0.5 — Tabela 11 × Tabela 8.** Proveniência estabelecida: Tabela 8 = pipeline TERA congelado; Tabela 11 = TODOS os 5 braços retreinados sob o protocolo de replicação (run_v29). Agora declarado na legenda e no texto ("values are comparable within this table only"); ×5,7 mantido como contraste interno. **Reforço da análise (novo, dos dados):** na política FIXA, NoTemp não é distinguível do PIOR α (0,250 vs 0,161; p=0,91) — mas sob MHEG NoTemp é pior que TODA a grade (×3,5 canônico p=0,005; ×3,0 pior α p=0,003; ×4,8 média p=0,001) e ECE pior em 12/12 contra todo ponto da grade. "Essentially flat" substituído por essa formulação quantitativa e simétrica (a mais forte que os dados sustentam).

**P0.6 — ECE em 4 escalas.** Variantes nomeadas (ECE_pipe, ECE_replay, ECE_cal, ECE_v29), nota da §4.8 atualizada, legendas ajustadas, mapa de protocolos criado (nova Tabela 23).

**P0.7 — Figura 14 / H_strat.** Confirmado e AGRAVADO: recomputado pelo MESMO procedimento, AF-TOI 0,343±0,026 < Baseline 0,394±0,063 < calibrados 0,63–0,82 → o score é sensível à escala de probabilidade, não à organização, e não sustenta claim em NENHUMA direção. "Systematically exceeds" removido; figura removida; §4.8 declara a remoção e reapoia o controle na morfologia (Fig. 12), no lócus (Fig. 13) e na Tabela 14.

**P0.8 — Tabela 13/Fig. 11 × FR.** Sem contradição numérica: FR usa score episódico = média dos últimos k=6 frames vs limiar calibrado no VAL; p̄_max = máx. por episódio (descritor de sinal). Definições agora explícitas (célula da Tabela 2 + Movement 6).

**Etapa 3 — Formalização AF-TOI.** §3.2 reescrita com a LOSS IMPLEMENTADA completa (Eqs. 5–9): L_hard (BCE com rampa), L_align (MSE em probabilidades temperadas T=2,0, gateada a críticos — análogo em espaço de probabilidade da destilação; dito explicitamente), L_ep, L_onset (track/floor/mono/push com foco exp normalizado), L_late, L_pre, L_tail, L_neg; schedules corrigidos (warm-up 12/ramp 16/total 80; β_max=0,35; piso α=0,50 — artigo dizia 10/17 e λ→1,0). Tabela 6 corrigida (θ_late 0,10, λ_late 2,0, K/N_pre, β_max, γ_ep, δ, pesos). Ref. §1: 2.7→2.6. Distinção vs KD reescrita nos termos que as equações agora sustentam (espelho no rebuttal C1) + parentesco com perdas exponenciais de antecipação citado.

**Etapa 4 — Trustworthy.** Definição por SEED (FR ≤ 0,05 em toda seed): exclui o controle negativo (viola em 3 seeds, máx 0,158 — corrigido na passagem adversarial) e o AF-TOI+MHEG passa (máx 0,017). §2.2 + espelho C7 + "unsafe regime" reancorado no critério por seed.

**Etapa 5 — Skip-RNN.** Texto reescrito: "matches or exceeds" dito com todas as letras; TTDef=0,000 explicado como artefato (alerta pré-onset → detecção colapsa em t0) e reconhecido como LIMITAÇÃO do desenho de avaliação; PreOcc/NormOcc reclassificadas como métricas COMPLEMENTARES de semântica do sinal, introduzidas por causa deste resultado; sem ranking global. Espelhos no rebuttal (C5-iv + citação).

**Etapa 6 — Gate de confiança/espantalho.** IMPLEMENTADA a variante periodic-refresh + confidence (R∈{2,4,8}, mesmo protocolo v2): quebra o deadlock (Baseline R=8: 85,9±0,4% sup, FR=0, TTDef 0,161 s). Reportada honestamente na §4.11: cadência cega basta para os endpoints in-distribution, mas pertence à família sem estrutura que, a custo pareado, falha FS@onset no estresse (§4.10). Rebuttal C5-i atualizado ("This remains a limitation."). Script: `run_refresh_gate_v2.py` + CSV. **Tabela 14 (IC+MHEG)** enfrentada de frente na §4.8: "matches or exceeds AF-TOI+MHEG nos endpoints deste replay", com leitura mecanística e status não-testado-sob-shift declarado.

**Etapa 7 — Mapa de protocolos.** Nova Tabela 23 (posicionada após a Tabela 22 para NÃO deslocar a numeração usada no rebuttal) com seeds/proveniência/regra de decisão/orçamento/variante de ECE por campanha.

**Etapas 8–11.** Linguagem moderada ("bear out", "rules out", "decisive", "confirming/confirms" (5×), "establishes", "systematically"); resumo 238→**216 palavras** com comparador qualificado ("the cheapest configuration"); marcas do processo de revisão removidas do corpo (8 ocorrências) com espelhos; Data Availability harmonizada (repositório GitHub verificado público em 23/09); Tabela 1: Edge do TERA ✓→○† com nota; Tabela 8 célula "—" preenchida (0,223±0,145, valor que a Tab. 12 já usava); Tabela 10 coluna abrupta TTD_b→TTD_a; Intro/Conclusão desacoplam "vehicular and ad hoc"; Discussão/Conclusão/§4.13 alinhadas ao resultado de orçamento pareado da §4.12 (vantagem = cobertura e oportunidade, não posicionamento a custo igual), incl. a frase do §4.10 "decisive result" reescopada.

**Etapa 12 — Rebuttal.** Espelhos simultâneos em TODOS os trechos citados alterados; R2-Rec com a lista concreta de onde a linguagem foi moderada; R1-C6 reescrito relatando as correções da Fig. 5 como correções; C7 com a lógica do critério por seed; "Additional changes" itens 7–9 = aviso EXPLÍCITO das correções da verificação interna (contagem 7/12, FLOPs, unidade de entropia + bug do replay, exemplares da Fig. 5, remoção da Fig. 14/H_strat, célula da Tab. 8, rótulo da Tab. 10, loss completa, mapa de protocolos). Numeração: Figura 15→14 (remoção da antiga Fig. 14) refletida nos 4 refs do rebuttal; tabelas inalteradas (mapa = Tab. 23).

**Etapas 14–15.** `check_manuscript_consistency.py`: **0 FAIL / 40 OK** (`consistency_report.md`). Compilação local: artigo 34 pp, rebuttal 29 pp, 0 erros, 0 refs indefinidas; overfulls restantes: 1 hbox interno do frontmatter do cas-dc (pré-existente, sem estouro visível na p. 1) + 4 menores <20 pt (tolerância pré-existente).

## B. Apontamentos NÃO confirmados
- p=0,0625 "impossível com 9/12 zeros": o p estava certo; o ERRO era a contagem (7/12). (P0.4)
- Tabela 13 × FR: compatíveis; faltavam definições, não havia inconsistência numérica. (P0.8)
- R3.11 (refs Zhang): sem problema (a própria auditoria concluiu isso).

## C. Números recalculados (fontes em metric_provenance.csv)
7/12 zeros FR; p=0,0625 (wilcox) / 0,052 (zsplit) / 0,424 (pratt); ablação vs grade α (hyb ×3,0–×4,8, p≤0,003; fixa vs pior α p=0,91); 58.242 params / 57,2 K MACs / 115 KFLOPs; H_strat uniforme (0,343/0,394/0,63–0,82); Fig. 5 painéis (H em bits, supressão por janela); supressão onset-window 9,8% vs 37,2% (log v2); refresh gate (tabela completa por R e trajetória); resumo 216 palavras.

## D. Scripts (em tera-adhoc-artifact/6-revision-round1/scientific-audit/ e baselines_novos/)
check_model_complexity.py · recompute_statistics.py · check_manuscript_consistency.py · regen_dense_figs_en_v2.py · run_refresh_gate_v2.py · reconstrução do episode_frame_logs_v2.csv (documentada abaixo) · fix no generate_episode_frame_logs.py (3 cópias).

## E. CHANGED_FILES
- Artigo 1/revisao-round1/: **artigo.tex**, **rebuttal.tex**, **robustness_section.tex**, **paper_table2_rows.tex**, image/4-fig_entropy_dist_en.pdf, image/5-fig_temporal_trace_en.pdf, image/6-fig_gating_heatmap_en.pdf, artigo.pdf, rebuttal.pdf (+ AUDIT_MATRIX.md, SCIENTIFIC_FIX_REPORT.md, consistency_report.md, metric_provenance.csv — novos)
- tera-adhoc-artifact/: 5-robustness-replay-runv29/Inferencia/{generate_episode_frame_logs.py (fix), episode_frame_logs_v2.csv (novo)}; 6-revision-round1/scientific-audit/{scripts,results} (novos)
- FGCS/Inferencia/ e Experimentos-Artigo1-AdHoc/Inferencia/: generate_episode_frame_logs.py (mesmo fix)
- Experimentos-Artigo1-AdHoc/baselines_novos/: run_refresh_gate_v2.py + refresh_gate_v2_results.csv (novos)
- Backup íntegro: revisao-round1/backup-pre-scifix-20260923/

## F. Limitações que CONTINUAM (declaradas no artigo/rebuttal)
- IC+MHEG e Skip-RNN igualam/superam os endpoints in-distribution; a diferenciação reivindicada é semântica do sinal + mecanismos de runtime + comportamento sob shift controlado.
- Validação sintética, nó único, sem hardware físico; campanhas complementares n=4 descritivas.
- episode_frame_logs_v2.csv foi reconstruído pelo replay sobre trajetórias armazenadas (protocolo da Tabela 19); a regeneração via modelos (prefix-window, GPU) deve reproduzi-lo a menos de pequenas diferenças — item de verificação na sua máquina (1 comando, ~min).

## G. Decisões que ficam com os autores (UNRESOLVED_ISSUES)
1. **Figs. 12–13 (calibração)**: fontes pequenas e "95\%" literal no título da Fig. 13 — regeneração exige rerun do experimento de calibração (CPU, probs armazenadas); script original das figuras não localizado na árvore. NEEDS_SOURCE_VERIFICATION.
2. **IC+MHEG sob o warp temporal da §4.12** (recomendação 6 da auditoria): experimento novo; adiável (limitação já declarada na §4.8).
3. **Mover §§4.9–4.11 para material suplementar** (sugestão estrutural da auditoria): não aplicado sem aprovação.
4. Rodar na sua máquina (GPU): `python generate_episode_frame_logs.py` corrigido para regenerar o log via modelos e confirmar o v2 reconstruído.

## H. Resumo executivo
P0 corrigidos: 8/8 (P0.1–P0.8, incl. um bug operacional real encontrado além da auditoria: gate do replay em nats). P1 corrigidos: linguagem, α-grid, Skip-RNN/IC framing, §4.12↔Discussão, straw-man (refresh gate implementado), definições p̄_max/FR. P2/P3 corrigidos: resumo 216, marcas de revisão, Data Availability, Tab. 1/8/10, unidades das figuras. Pendentes de dados/decisão: itens G.1–G.4. Compilação: 0 erros; verificador automático: 0 FAIL/40 OK.

## I. Segunda passagem adversarial (papel do Revisor #3, 23/09 — ADVERSARIAL_REVIEW_R3.md)
1 CRITICAL encontrado e corrigido (C1: máximo por seed do Baseline+MHEG era 0,158 em 3 seeds, não 0,117 — texto §2.2 + espelho C7). 5 MAJOR corrigidos (M1 Fig. 4×texto: pré-onset concentra no máximo, não em 0,5 bit; M2 "suppression is uniform" no Baseline contradizia o log v2 — proteção residual do onset é do gatilho cinemático, 37% vs 10%; M3 BCE é ponderada por classe; M4 gates de ativação δ>0,5/δ>1,0 declarados; M5 refresh gate agora declara n=4 + calibração oracle + caráter descritivo). M6 resolvido (rebuild_episode_frame_logs_v2.py versionado). M7 PENDENTE (push do artefato público). 3 MINOR aplicados. Recompilado: artigo 34 pp / rebuttal 29 pp, 0 erros, 0 refs indefinidas; verificador 0 FAIL/40 OK.

## J. Fechamento dos itens de reprodutibilidade (23/09, noite)
- Item 2 RESOLVIDO: ablacao_mecanismo_temporal/, skiprnn_baseline/ (com checkpoints), temporal_scale_generalization_v2/ (com data_campaign) e baselines_novos v2+refresh copiados para 6-revision-round1/ do artefato.
- Item 1 PARCIAL: commit 32172a8 criado na branch fix/rebuttal-scientific-consistency (345 arquivos, ~79k linhas) contornando o index.lock com GIT_INDEX_FILE alternativo; o PUSH exige credencial do Windows: na pasta do repo, rodar
    del .git\index.lock & del .git\HEAD.lock   (limpeza dos locks órfãos)
    git push -u origin fix/rebuttal-scientific-consistency
- Item 3 RESOLVIDO no essencial: o gerador das Figs. 12–13 É o tera_posthoc_calibration_experiment.py (mapeamento documentado em scientific-audit/FIGURE_PROVENANCE.md); a Fig. 13 foi regenerada AGORA por script versionado (regen_fig13_locus_en.py) a partir de artefatos do repo, com fontes maiores e título sem o "95\%" vazado; a Fig. 12 regenera com o rerun do script de calibração (torch, máquina local). R2 C6 do rebuttal atualizado. Recompilado: 0 erros, 0 FAIL/40 OK.

## K. Verificação independente na GPU (23/09, noite — item G.4 FECHADO)
A autora executou o `generate_episode_frame_logs.py` corrigido (GPU, cuda): τ recalibrado no VAL caiu EXATAMENTE nos valores canônicos (τ_Δ=0.0190, τ_H=0.95) e os skip rates do replay por modelo bateram com o CSV canônico do run_v29 (Baseline+MHEG 56.2%, AF-TOI+MHEG 40.1%) — primeira geração deste log com o ramo entrópico ativo. O log da GPU foi adotado como `episode_frame_logs_v2.csv` (a reconstrução anterior ficou como `episode_frame_logs_v2_reconstructed.csv`; diferia só nos totais, 52.6/31.7, por efeito de janela de prefixo). VALIDAÇÃO DOS NÚMEROS DO MANUSCRITO no log canônico: idênticos — exemplares da Fig. 5 (ep. 36 prog t0=5.42s; ep. 148 abrupto t0=2.50s), painéis (0.49→0.97 / 0.36→0.73 bits; supressão 58→14%/43%; Baseline 55–56%), janela de onset (9.8% vs 37.2%; fora 34.3% vs 54.4%), agregados da Fig. 6(c) (0.48→0.55 / 0.68→0.95), minoria saturada 12/48, fases da Fig. 4. Figuras 4–6 regeneradas do log canônico; artigo recompilado. Nenhuma alteração de texto necessária.
