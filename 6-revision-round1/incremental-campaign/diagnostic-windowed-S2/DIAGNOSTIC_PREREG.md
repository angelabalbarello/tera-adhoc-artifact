# Pré-registro — Diagnóstico exploratório S2 na semântica de JANELA (artigo v9)

**Data:** 11/10/2026, registrado ANTES da execução. **Definido APÓS a campanha
incremental** (motivado pelo colapso S2 observado lá). **Exploratório.**

## Objetivo
Verificar se a implementação PRINCIPAL do artigo v9 (janela causal W=96 recomputada,
`make_prefix_window`, pad à esquerda pelo frame 0, supressão reutiliza a última
probabilidade emitida; `tera_infer.py::infer_stream_hybrid`) também deixa de atualizar
quando o aviso cinemático do gate é retirado — i.e., se o colapso não é exclusivo da
variante incremental.

## Condições (12 seeds, TESTE, checkpoints e limiares PUBLICADOS, sem recalibração)
- **A (normal):** gate publicado: ativa se max|Δx_kin(t,t−1)| ≥ τ_δ OU H(p_emitida) ≥ τ_h;
  primeira atualização forçada em t=0. Deve reproduzir os valores publicados por seed
  (expanded_per_seed.csv) — divergência será investigada antes de interpretar B.
- **B (sem aviso cinemático):** idêntica, mas o delta cinemático ENTREGUE AO GATE é
  fixado em 0 a partir da segunda decisão (t≥1). **Entrada da LSTM intacta** — só o
  sinal de decisão é retirado. Regra da primeira atualização preservada.

Configs: baseline_hybrid e aftkd_hybrid (Baseline+MHEG, AF-TOI+MHEG).

## Mecanismo verificado no código antes de rodar
`tera_infer.py` NÃO contém atualização periódica obrigatória, timeout ou fallback
(busca por refresh/timeout/fallback/periodic/force: só a inicialização em t=0).
Nenhuma proteção nova será adicionada neste diagnóstico.

## Métricas por seed/condição
FR (detecção estável m=3) + nº de críticos perdidos; TTD e TTDef (definições
publicadas); % de atualizações suprimidas; maior sequência consecutiva sem
atualização; nº de episódios congelados até o fim após uma supressão.
Logs por episódio: decisões do gate, entropia usada, p emitida, onset, instantes de
atualização e detecção.

## Verificação da cadeia causal
dk=0 → H(p emitida)<τ_h → supressão → p e H congeladas → nenhuma atualização futura.
Se ocorrer também na janela, o problema NÃO será atribuído exclusivamente à execução
incremental.

## Limites
Conclusões restritas a este cenário (ausência do aviso cinemático no gate); sem
generalização para outras falhas de sensores. Nenhuma alteração ao artigo, aos
limiares ou ao algoritmo nesta etapa; proteção eventual será proposta separadamente.
