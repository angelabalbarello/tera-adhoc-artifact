# V6 — Reconciliação científica final (11/10/2026)

Builds: **artigo_v6.pdf** (36 pp, 0 refs indefinidas) e **rebuttal_v6.pdf** (23 pp, 0 overfull). Fonte canônico do rebuttal: `rebuttal.tex` (`rebuttal_final_v2.tex` é cópia idêntica de entrega; os dois enviados antes eram de fato idênticos). Cópia da v5 preservada em `artigo_v5_BACKUP.tex`/`rebuttal_v5_BACKUP.tex` + PDFs v5. Suíte: **0 FAIL/40 OK + 9/9 testes** (novo: `test_borderline_consistency.py`, semântico e numérico, não textual).

## 1. PRIORIDADE MÁXIMA — Tabela 13 / Figura 11 / FR: RESOLVIDO COM EVIDÊNCIA POR EPISÓDIO

**A incompatibilidade apontada era real e tinha causa identificável: ERRO DE GERAÇÃO DE ARTEFATO.**

Investigação direta (sem aceitar nenhum relatório como verdade):
- Fonte das linhas Normal/Crítico da Tab. 13: `exp_seed12_round1/logs/borderline_frame_logs.csv`, gerado por `tera_log.py::_infer_episode`, que rodava o modelo **stateless** — cada frame como sequência isolada de comprimento 1, estado recorrente reinicializado a cada frame. Protocolo INVÁLIDO e distinto do publicado.
- Protocolo publicado (todas as métricas): `tera_infer.py::infer_stream_*` com **janela-prefixo causal completa** (`make_prefix_window`, frames 0..t com pad).
- No stream inválido: 35/120 críticos (29,2%) com p_max<θ e, aplicando a regra de detecção estável a ele, FR=0,300 — exatamente a contradição aritmética apontada (FR publicado 0,000).

**Reconstrução do stream correto, sem GPU e sem torch utilizável no sandbox**: checkpoint lido por unpickler próprio (formato zip do torch) e LSTM de 2 camadas reimplementado em numpy, protocolo prefix-window idêntico, mesmos checkpoints congelados (`student/baseline_seed42.pt`) e mesma partição de teste (`X_te_seed42.npy`, 157 episódios, 120 críticos).

**Validação do pipeline de reconstrução contra valores publicados (4 casas):** FR = 0,000/0,000 ✓; TTD_b = 0,0825 (baseline) / 0,0156 (AF-TOI) ✓ idênticos ao `expanded_per_seed.csv` seed 42; A_p = 0,583 / 0,8565 ✓ idênticos à coluna TTD_Progressive. O reconstrutor reproduz o protocolo publicado exatamente.

**Valores corretos da Tab. 13 (stream válido):** AF-TOI Fixed — Normal 0,000±0,000 (0,0%); Critical **0,940±0,083 (100,0%)**. Baseline Fixed — Normal 0,001±0,000 (0,0%); Critical 0,977±0,077 (100,0%). Atenção/Alerta inalterados (a análise gpu_v23 usa prefixo completo — protocolo válido — mas outro checkpoint: declarado na nota da tabela como comparação qualitativa).

**Entregas deste ponto:** `scientific-audit/results/tab13_fr_reconciliation_seed42.csv` (por episódio: episode_id, seed, regime, checkpoint, probability_stream, onset_frame, θ, m, p_max, first_valid_detection_frame, detected, missed); `borderline_frame_logs_v2.csv` (stream corrigido frame a frame); Figura 11 regenerada; verificação semântica implementada e travada em teste: **FR ≥ fração(p_max<θ)** na mesma população/stream, e FR do log == FR publicado. `tera_log.py` corrigido NA FONTE (prefix-window, com nota do fix). Frase residual da §4.7 ("episode-classification rule that defines FR") corrigida para a regra de detecção estável.

**Impacto nas conclusões:** a leitura da §4.7 MUDA — a ordenação dos quatro níveis agora é monotônica em p̄max (0,000<0,048<0,380<0,940) e em fração de ativação (0<5,8<48,8<100%), com a cautela explícita de que Atenção/Alerta vêm de outro checkpoint/protocolo. O argumento anterior ("ordering is not monotonic... activation frequency more stable") baseava-se no artefato inválido e foi reescrito.

## 2. A_p e latências — CONFIRMADO COM EXEMPLOS POR TRAJETÓRIA

Fórmula verificada no código e reproduzida numericamente: A_p = média de max(0, t₀−t_det_full)·Δt sobre críticos progressivos, busca estável (θ=0,10, m=3) desde o frame 0, não detectados contam 0. Exemplos reais extraídos do stream validado (seed 42, AF-TOI): ep3 = antecipação que persiste até o onset (det_full=59, onset=60, delay pós-onset 0); **ep137 = ativação pré-onset (frame 30) que NÃO persiste e reentra com atraso pós-onset (det 52 > onset 51)** — confirmando que detecção pré-onset NÃO entra automaticamente com atraso zero; ep16 = detecção pós-onset; nenhum crítico não detectado em seed 42. O texto do §4.4 já apresenta A_p e TTD_b como quantidades em referências temporais distintas que não se compõem aritmeticamente. Convenção do marcador m=3: o instante reportado é o primeiro frame da tripla (retrospectivo); a confirmação causal ocorre no terceiro — nota adicionada na definição. "episode-rule TTD_ef" removido da §4.12 (artigo e espelho): a Tab. 8 usa a regra de detecção estável; a não-comparabilidade vem de episódios warped e pipeline distintos.

## 3. Calibração — ARGUMENTO FECHADO

Mantida a correção matemática (§2.5, §4.8); removidas as sobras: "frontier structurally distinct"/"reduces FAR and TTD_ef simultaneously" substituídos por descrição dos **pontos medidos** (FAR não é definido no artigo — removido); parágrafo da Discussion agora "none of the evaluated calibrators induced temporal differentiation in the measured trajectories" (observação, não teorema); título da Figura 13 neutralizado ("ECE vs. TTD_ef operational locus (95% CI, seeds 42–45)"), anotação "AF-TOI frontier"→"AF-TOI", figura ampliada e regenerada do script versionado. IC+MHEG competitivo mantido explícito; não testado sob shift mantido explícito. Binning das 4 ECEs declarado (10 bins equilargos; replay descarta bins<10 com denominador total preservado).

## 4. Contradições residuais — REMOVIDAS COM NÚMEROS EXATOS

"identical FR positions" → "nearly coincident FR positions (0.033 vs. 0.035)"; "as often as on redundant ones" → "suppression also lands on emerging-risk timesteps" com os 37%/54% vs 10%/34% já adjacentes; claim de alternativas sob shift → caracterização desigual explícita (Skip-RNN testado descritivamente no warp preservando endpoints; isotônico e refresh NÃO testados sob shift); imputação ao refresh já removida na rodada anterior (leitura de família mantida com a ressalva "not rerun under that stress protocol"); PreOcc/NormOcc permanecem complementares; ocupação acima de θ=0,10 nunca é equiparada aos estados Attention/Alert/Critical (limiares 0,15/0,38/0,50 distintos — Eq. 14).

## 5. Tabela 11 — COMPLETA COM DADOS REAIS

Blocos **Fixed** (FR, TTD_ef) e **MHEG** (FR, TTD_ef, %Sup) para as 5 condições, mean±sd sobre as 12 seeds, direto de `mechanism_ablation_per_seed.csv` (nenhum valor reconstruído): NoTemp 0,180±0,104 vs canônico 0,051±0,097 (×3,5), pior-α agendado 0,059±0,091 (×3,0), média da grade 0,037 (×4,8) — exatamente os contrastes do texto; p-values e r_rb mantidos; multiplicidade declarada na legenda (Wilcoxon exato bicaudal, sem correção, secundário sob a taxonomia da §4.1).

## 6. Algoritmo, perda e implementação

Alg. 1: condição explícita **(t=1) ∨ (Δx≥τΔ) ∨ (H≥τH)** — espelha `tera_infer` (`if t == 0 or last_p is None`), que já forçava a primeira atualização (erro documental, não mudança de execução); 0·log 0 ≡ 0 com clip 1e-7 declarado; 1.024 bytes = estado h,c das duas camadas (2×2×64 FP32), "recurrent state only, not total runtime memory". Colisões de notação resolvidas em nota (T temperatura×comprimento; K janela×conjunto cinemático; α foco×peso α(e)); "pure hard supervision" corrigido na Tab. 6 para "β(e)=0 (sem alinhamento; termos fixos e onset-shaping ativos nos pesos iniciais)" — era erro documental (δ inicia em 0,30 e os λ fixos estão ativos desde a época 1, conforme manifest da loss). "Same frozen pipeline": sustentado por `manifest.json` da campanha e pelos hashes de configuração/gerador já citados no Data Availability (nenhum hash novo inventado).

## 7–8. Figuras, linguagem e rebuttal

Fig. 11 e Fig. 13 regeneradas (fontes `.pre-v6` preservadas); marcas editoriais do corpo já removidas nas rodadas anteriores e travadas por teste. Rebuttal sincronizado: espelhos das frases alteradas (§4.12 scale-note, §6 single-node/GPU), "none changes a conclusion" requalificado (itens (a)–(f) não mudam conclusão; itens 10–11 mudam leituras específicas e dizem quais), **Additional 11** documenta o bug da Tab. 13 com a reconciliação completa e reproduz a Tab. 13 corrigida; a Tab. 11 completa é reproduzida no R3 C6. Fonte canônico único declarado.

## Verificações (semânticas, numéricas, visuais)

- Semântica: FR ≥ fração(p_max<θ) no stream unificado — teste automatizado.
- Numéricas: FR/TTD_b/A_p reconstruídos == publicados (4 casas); Tab. 11 == CSV por seed; Tab. 13 == log corrigido; abstract 226.
- Visuais: Fig. 11 e Fig. 13 inspecionadas após regeneração; **Fig. 12 permanece com títulos de painéis sobrepostos — PENDÊNCIA** (ver abaixo).

## PENDÊNCIAS RESTANTES (nenhuma silenciada)

1. **Fig. 12 (fig_calibration_entropy_temporal): cosmética, requer artefato ausente.** O gerador é o experimento de calibração completo (`tera_posthoc_calibration_experiment.py`, função `fig_entropy_temporal_curves`), que consome as trajetórias H(t) em memória; as trajetórias por frame das variantes calibradas não estão arquivadas como CSV. Sem elas não regenero a figura sem re-executar o experimento (GPU da Angela; mesmo script, ajustando apenas `suptitle`/espaçamento). Impede: apenas a correção visual dos títulos.
2. **Refresh periódico e IC sob o protocolo de stress/shift: não testados** — agora declarados como tal no texto; executá-los seria experimento novo (fora do escopo desta rodada, a afirmação correspondente foi restringida em vez de sustentada).
3. **Efeito padronizado da ablação de canais da campanha de estresse**: per-seed FS não arquivado (REQUIRES RECOMPUTATION registrado desde o STATISTICAL_AUDIT).
4. As Figs. 4–6/8–10 não foram re-renderizadas nesta rodada (dados inalterados; scripts versionados).

## Veredito

Os quatro bloqueios científicos da auditoria (Tab. 13/FR, calibração residual, contradições de narrativa, Tab. 11) estão **resolvidos com evidência reproduzível**; o que não pôde ser sustentado foi **restringido no texto, não defendido**. Pendências remanescentes são: uma correção visual dependente de rerun (Fig. 12) e limitações declaradas — nenhuma incompatibilidade dado↔texto conhecida permanece. Não declaro "ready" por busca textual: o critério novo é o teste semântico por episódio, e ele passa.
