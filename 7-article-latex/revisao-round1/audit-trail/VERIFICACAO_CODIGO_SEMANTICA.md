# Verificação: qual código gerou os resultados e qual é a sua semântica (13/10/2026)

Pergunta da autora: "será que não foi usado um código diferente para deduzir que a
inferência recomputa a janela W=96?" — Varredura nas pastas conectadas
(TreinamentoNovo, Safe-Drive-Project, Artigo 1, ArtigoAdHocNetwork,
versao-final-segura). Cópias de inferência encontradas e examinadas:

## 1. run_v29 (ORIGINAL — base do protocolo da submissão)
`tera-adhoc-artifact/5-robustness-replay-runv29/Inferencia/run_v29_ablacao_ttdef_ajuste_gatting.py`
- `make_prefix_window` (linha 564): janela causal até t, pad à esquerda pelo 1º frame;
- `infer_stream_fixed` e `infer_stream_hybrid`: a CADA atualização,
  `x_slice = make_prefix_window(X[i], t, window)` e `model(xt)` — **nenhum estado é
  passado entre chamadas** (o nn.LSTM do torch parte de h=c=0 a cada janela);
- supressão: `frame_probs[i,t] = last_p` (só a probabilidade emitida);
- a janela fatia `x_seq[:t+1]` do stream completo ⇒ frames suprimidos são
  relidos pelas janelas seguintes.
**Semântica: janela recomputada.** ✔

## 2. TERA Pipeline congelado (rodou as DUAS campanhas publicadas)
`tera-adhoc-artifact/1-pipeline-tera/tera_pipeline/inference/tera_infer.py`
- Docstring declara: "os dois laços de streaming seguem as definições do protocolo
  de replay (run_v29...)"; mesmo `make_prefix_window`, mesma supressão por `last_p`,
  nenhum estado passado. **Semântica: janela recomputada.** ✔
- Proveniência: manifest da campanha submetida (exp_20260519_055950, 19/05, seeds
  42–45) e da expansão (exp_seed12_round1, seeds 42–53) com os MESMOS hashes
  (config d5327a24d1d2f7f9, gerador 65bbec4a23050639); o launcher da expansão
  verifica os hashes antes de executar.

## 3. Cópia "viva" FGCS (datada de 19/05/2026, o dia da campanha)
`FGCS/Arquitetura/tera_pipeline/inference/tera_infer.py`
- Cabeçalho documenta: "[FIX-1] infer_stream_fixed e infer_stream_hybrid usam
  make_prefix_window ... idêntico ao run_v29 linha 571".
  **Semântica: janela recomputada.** ✔ — e o próprio "[FIX-1]" registra que houve
  uma versão anterior SEM janela, corrigida ANTES da campanha. Isso explica a
  origem da descrição antiga do artigo: o texto herdou o desenho pré-FIX
  (single-step), enquanto o código executado já era o de janelas.

## 4. Única exceção conhecida
`tera_pipeline/logging/tera_log.py` fazia inferência stateless (frame isolado) —
afetava SOMENTE a Tabela 13/Figura 11, foi detectado e corrigido na rodada v6, com
o rastreio de alcance documentado (FIGURE_PROVENANCE.md; rebuttal Add. 12).

## 5. Prova numérica (independe de leitura de código)
Com os checkpoints e limiares publicados: a reimplementação numpy da semântica de
JANELA reproduz os valores publicados por seed com diferença 0,000000 em FR e
≤5×10⁻⁴ p.p. em supressão (24 células; TTDef médio aftkd 0,0989 = publicado); a
semântica single-step dá TTDef 0,170 vs 0,023 (seed 42). Só uma das duas semânticas
reproduz os números do artigo — a de janela. Auditor externo replicou os quatro
resultados da seed 42 de forma independente (diferença <5e-6).

## 6. E o que foi ENVIADO à revista?
A versão submetida (`7-article-latex/00-versao-SUBMETIDA/artigo.tex`) já continha a
contradição: o Algoritmo 1 do PDF que os revisores leram diz
`(p_t, h_t) ← LSTM(x_t, h_{t−1})` e, na supressão, `p_t ← p_{t−1}; h_t ← h_{t−1}`
(linhas 903/906) — ou seja, descrevia a variante single-step persistente — enquanto
TODOS os números submetidos vinham do código por janelas (itens 1–2). A revisão não
mudou a arquitetura nem os experimentos: corrigiu uma descrição que já estava
errada na submissão, exposta quando o R1 C5 pediu a análise de complexidade do
Algoritmo 1.

**Conclusão:** não há cópia alternativa do código que sustente a descrição
single-step para os resultados publicados; todas as implementações que geraram
números do artigo (submissão e revisão) recomputam a janela causal com estado
reinicializado e releitura dos frames suprimidos.
