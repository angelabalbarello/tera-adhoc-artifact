# Diagnóstico S2 na semântica de janela (artigo v9) — relatório

**Data:** 11/10/2026 · **Exploratório, definido após a campanha incremental**
(pré-registro: DIAGNOSTIC_PREREG.md, escrito antes da execução). 12 seeds, TESTE,
implementação publicada (janela causal W=96 recomputada, pad frame 0, supressão reusa
a última p emitida, 1ª atualização forçada em t=0), checkpoints e limiares publicados,
sem recalibração. Nenhuma alteração ao artigo, limiares ou algoritmo.

## Sanidade (condição A, funcionamento normal)

A condição A reproduz os valores publicados por seed (`expanded_per_seed.csv`):
max|diff| FR = 0,000000; max|diff| supressão = 0,0005 p.p.; médias aftkd_hybrid
TTDef 0,0989 / Sup 48,13% = publicadas. A implementação testada é a do artigo.

## Resultado (condição B: delta cinemático do gate = 0 a partir da 2ª decisão; entrada da LSTM intacta)

| Config | Cond | FR (m=3) | Críticos perdidos | TTDef (s) | Sup % | Maior seq. sem update | Congelados até o fim |
|---|---|---|---|---|---|---|---|
| AF-TOI+MHEG | A | 0,0014 | 0,17/120 | 0,099 | 48,1 | 55,5 | 54,8* |
| AF-TOI+MHEG | **B** | **1,000** | **120/120** | **10,0** | 98,96 | 95 | 157/157 |
| Baseline+MHEG | A | 0,0347 | 4,2/120 | 0,570 | 61,0 | 55,5 | 108,5* |
| Baseline+MHEG | **B** | **1,000** | **120/120** | **10,0** | 98,94 | 95 | 157/157 |

\* Em A, "congelado até o fim" é benigno (quiescência no final do episódio, após
detecção); em B é o episódio inteiro a partir de t=1. Médias sobre 12 seeds; valores
por seed em `s2_windowed_results.csv`; logs por episódio (`logs_ep_*.csv`) e por
decisão do gate (`logs_gate_*.csv.gz`: dk, entropia, ativação por frame).

## Cadeia causal verificada

Nos logs (ex.: seed 42, ep 0, B): dk entregue ao gate = 0 em todos os t≥1; H(p emitida
em t=0) = 0,0013 < τ_h; atualização suprimida; p e H permanecem congeladas; **zero
ativações até o fim**. Em 3.766/3.768 episódios-config (12 seeds × 2 configs × 157
episódios) não há nenhuma atualização após t=0 — os 2 restantes tinham H(p₀) ≥ τ_h,
único caso em que o ramo entrópico reativa sozinho.

**Por que o gate fica fechado:** o ramo entrópico avalia a entropia da probabilidade
*emitida* (congelada durante a supressão), não da evidência nova. Supressão ⇒ p
constante ⇒ H constante abaixo de τ_h ⇒ o gate nunca reabre. É um ponto absorvente da
regra OR(dk≥τ_δ, H(p_emitida)≥τ_h) quando o aviso cinemático desaparece — e **não
depende da semântica de execução**: ocorre igualmente na janela recomputada (aqui) e
na variante incremental (campanha, estresse S2).

## Respostas

1. **A versão principal também pode ficar sem atualizar?** Sim. Sob ausência do aviso
   cinemático no gate, a implementação por janelas do v9 congela após t=0 em
   praticamente todos os episódios (3.766/3.768), identicamente à variante incremental.
   O problema é da regra de gating, não da execução incremental.
2. **Quanto prejudica a detecção neste cenário?** Totalmente: FR = 1,000 nas 12 seeds
   e nas 2 configs (120/120 críticos perdidos por seed; TTDef = 10,0 s pela convenção
   publicada FR·T_ep).
3. **Existe proteção já implementada?** Não. `tera_infer.py` não contém atualização
   periódica obrigatória, timeout nem fallback; a única atualização forçada é a
   inicialização (t=0 / last_p None). Nada foi adicionado neste diagnóstico.
4. **Limitação a declarar no artigo:** o MHEG publicado não possui mecanismo de
   reativação independente do canal cinemático; se o sinal de mudança cinemática
   entregue ao gate for perdido/zerado, a supressão é absorvente e a detecção falha
   por completo. Escopo: este cenário (ausência do aviso cinemático no gate), estes
   checkpoints/limiares; **não** generaliza a outras falhas de sensores (a entrada da
   LSTM permaneceu intacta aqui).

## Proposta separada (NÃO implementada, NÃO avaliada nesta etapa)

Proteção candidata: atualização obrigatória a cada k frames suprimidos (refresh
determinístico, ex.: k=16 ⇒ supressão máx. ~94%), preservando o gate publicado.
Experimento necessário: recalibrar (τ_δ, τ_h) na validação COM o refresh ativo,
reavaliar A e B nas 12 seeds e reportar o custo (queda de supressão em A) vs o ganho
(FR em B). Só após decisão sua — os resultados do método original permanecem como estão.
