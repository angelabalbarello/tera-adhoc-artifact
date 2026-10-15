# Rodada dirigida pós-auditoria externa (11/10/2026) — resolução ponto a ponto

Referência: Auditoria_ADHOC_TERA_2026-10-11.pdf. Ordem seguida: reconciliar métricas → calibração → escopo → sincronização → linguagem, como recomendado.

## 1. PROBLEMA PRINCIPAL — latências Tab. 8 × Tab. 10 (RESOLVIDO PELO CÓDIGO, correção textual)

Fui ao código, não à interpretação. Em `tera_eval.py`:
- A coluna publicada como "TTD_p" é `_ttd_progressive_anticipation()` (L453): **antecipação pré-onset**, max(0, t₀−t_det)·Δt, busca desde o frame 0, misses contam 0 — **maior = melhor**. O docstring do próprio código avisa: "NÃO é usado diretamente no TTDef".
- TTD_a (`_ttd_subset_mean`, L507) e o TTD_b global (`compute_ttd_list`) são atraso pós-onset sobre detectados, a partir do onset.
- Verificação numérica (expanded_per_seed.csv): TTD_Progressive médio = 1,1997 (aftkd_fixed) e 0,6568 (baseline_fixed) — exatamente os valores da tabela.

**Não há inconsistência aritmética**: a conta 0,60×1,200 ≥ 0,720 s não se aplica porque 1,200 s não é atraso nem componente da média global; detecções pré-onset entram no TTD_b global com atraso ≈0 (por isso 0,069 s). O problema real era outro, e pior: **o texto do §4.4 interpretava antecipação como atraso** ("can delay classification consolidation") e a seta da coluna era ↓.

Correções: coluna renomeada **A_p (s) ↑** nas Tabs. 8 e 10, com definição explícita nas duas notas ("mean pre-onset anticipation... higher is better... not a component of TTD_b or TTD_ef"); parágrafo Regime asymmetry reescrito com a leitura correta e a **reconciliação explícita** entre colunas por regime e globais (referências temporais diferentes); legenda da Tab. 10 corrigida ("analogous to TTD_p" removido). Nenhum número alterado.

## 2. FR: regra de classificação × regra temporal (RESOLVIDO PELO CÓDIGO)

`tera_eval.py` L387–395 (comentário explícito no código) e `tera_calibrate.py` L136–155: **o FR publicado E o FR da calibração de limiares usam a regra de detecção estável** (p_t ≥ θ_TTD por m=3 frames consecutivos do onset), não o escore dos 6 frames finais. O escore k=6 define apenas a classificação de episódio consumida pelo F1. **Era a definição da Tabela 2 que estava errada** — corrigida para a regra temporal, com o escore k=6 reatribuído ao F1; célula do fatorial na Tab. 23 corrigida igualmente. As regras não podem discordar no FR publicado porque o FR nunca usou a regra de escore.

## 3. Calibração monotônica × entropia (CORRIGIDO MATEMATICAMENTE)

§2.5 e §4.8 reescritos: um rescaling monotônico preserva a ordem das probabilidades, **mas não, em geral, a das entropias binárias** (H cresce até p=0,5 e decresce depois); a ausência de reorganização temporal passa a ser **resultado medido para os calibradores avaliados** (Fig. 12), não impossibilidade matemática. Nota das ECEs ganhou o binning verificado no código: 10 bins equilargos em todas as variantes; ECE_replay descarta bins com <10 frames.

## 4. Extrapolações limitadas aos testes feitos

- Skip-RNN sob warp: qualificado "in that descriptive four-seed replay" (§4.11 + espelho).
- Refresh periódico × stress: agora declarado como **leitura de família** ("the periodic-refresh gate itself was not rerun under that stress protocol") no artigo e no rebuttal.
- IC sob shift: já estava explícito ("has not been tested here"); inalterado.

## 5. Sincronização carta ↔ manuscrito

- Escopo inferencial corrigido nos DOIS: "confined to the twelve-seed campaigns (the factorial study and the retrained mechanism-removal ablation of Table 11)" + warp exploratório (§4.1 e prosa R2).
- Marcas editoriais removidas do corpo: "in the corrected replay logs"→"in the replay logs"; "introduced in response to this result" removido; H_strat em presente neutro ("is not reported: ... it is sensitive to the probability scale"); nota da Tab. 14 neutralizada.

## 6. "Unchanged" → números medidos

Abstract: "leaves the miss rate unchanged" → "does not improve coverage" (**226 palavras**; contagem, rebuttal ×2 e locks atualizados). §1: "does not improve coverage (FR 0.033→0.035)". §4.5 já trazia 47 vs 50. Rebuttal R3 C5(ii) e matriz espelhados.

## 7. Linguagem (as 4 formulações da auditoria)

- "Two caveats are owed to the reader." → "These exemplars do not establish aggregate latency differences: ..." (+ ancoragem no agregado da Fig. 6, fechando a observação sobre generalizações do Baseline).
- "we state the result directly rather than by implication" → removido; a frase abre direto com a Tab. 14.
- "informative rather than flattering" → "reflects pre-onset threshold activation" (artigo + espelho).
- "a ceiling effect rather than an absent effect" → "only these five nonzero pairs contribute, and the contrast is not significant at the 0.05 level" (artigo + espelho; Limitações §5 mantém a leitura de teto factual).

## 8. Pendência da Tab. 11

Legenda agora aponta os valores do contraste agendado reportados no texto (0,051 vs 0,180 s).

## Registro no rebuttal

Novo item **Additional changes 10** descreve toda a rodada (renomeação A_p com reconciliação, regra de FR, correção da claim de calibração, binning das ECEs, números do controle negativo, leitura de família do refresh, escopo inferencial, Tab. 11).

## Estado

- artigo_v5.pdf: 36 pp, 0 refs indefinidas (17 overfulls pré-existentes, mesmos de antes).
- rebuttal_v5.pdf: 22 pp, 0 erros, 0 overfull.
- Suíte: 0 FAIL / 40 OK + 8/8, com locks novos (proíbe "miss rate unchanged", "informative rather than flattering", "a ceiling effect rather", "introduced in response to this result", "corrected replay logs", "owed to the reader", "rather than by implication", "cannot reorganize the temporal structure" no artigo; exige "pre-onset anticipation"); contagem oficial do abstract = 226 (pdftotext no build mais recente, agora resolvido por mtime).
- Nenhum número experimental alterado; todas as correções são de rótulo, definição, escopo ou leitura, verificadas contra código e CSVs.
