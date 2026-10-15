# TEMPORAL MECHANISM AUDIT (Fase 1) — 08/09/2026

## Eq. 8 no artigo
`artigo.tex`, label `eq:ponderacao_aftoi` (§3.2): w(t) = w_pre na janela pré-onset
[t0−k, t0); exp(−α(t−t0)) para t ≥ t0; 1 fora. É um ESQUEMA da implementação abaixo.

## Implementação correspondente (source of truth)
`FGCS/Inferencia/run_v29_ablacao_ttdef_ajuste_gatting.py`, config do artigo =
`ABLATION_MODE="A"` (abl_A). α = `onset_exp_alpha` (default 0.15). A loss do aluno
(linhas 1688–1697):

```
loss = α_curr·l_hard + β_curr·l_soft_gated + γ_ep·l_ep
     + δ·early_pen + 1.20·tail_pen + 2.00·noncrit_tail_pen
     + aux_now_weight·l_aux_now + late_pen + pre_pen + γ_hidden·l_hidden
```

Termos ANCORADOS AO ONSET (a "organização temporal"):
- **early_pen** (linhas 1537–1603): janela [t0, t0+K), curva-alvo derivada dos rótulos
  suaves referenciados ao onset, penalidades de tracking/floor/monotonicidade/push,
  todas ponderadas por `time_focus = exp(−α·steps)` **normalizado para média 1**
  (linha 1559).
- **late_pen** (M3, linhas 1608–1626): pós-onset, ponderação exp(−α·t) normalizada.
  Na config A, θ_late=0.10 é trivialmente satisfeito → termo ≈ inativo (comentário
  v27/A2 no próprio código).
- **pre_pen** (M6, linhas 1631–1655): janela pré-onset de 8 frames em episódios
  progressivos (o "w_pre" da Eq. 8), ativo quando δ>1.0.

Termos NÃO ancorados ao onset (permanecem em qualquer braço): l_hard, l_soft_gated
(KD MSE aluno↔referência, restrito a episódios críticos, uniforme no tempo), l_ep,
tail_pen/noncrit_tail_pen (últimos K_AGG frames, condicionados a episódio, não a t0),
l_aux_now; l_hidden desativado na config A.

## Papel matemático de α
α controla a CONCENTRAÇÃO do peso dentro da janela de onset: quanto maior α, mais
o gradiente se concentra nos primeiros frames após t0. Como `time_focus` é
renormalizado para média 1, α NÃO altera a massa total do peso — apenas o perfil.

## Comportamento de α=0
exp(0)=1 para todos os steps; após a normalização, `time_focus ≡ 1`: perfil
UNIFORME dentro da janela. Mas a janela continua selecionada por t0, as curvas-alvo
continuam referenciadas ao onset, a monotonicidade continua exigida, e pre_pen
continua ativo.

## Resposta à pergunta da Fase 1
**Does α=0 remove the onset-conditioned temporal organization while leaving all
other components unchanged? — NO.** α=0 remove apenas o decaimento intra-janela;
todo o condicionamento ao onset permanece.

## Condição escolhida para "no temporal organization"
`TEMPORAL_ORG_SCALE = 0`: multiplica por zero os TRÊS termos onset-anchored
(δ·early_pen, late_pen, pre_pen) e nada mais. O braço resultante é
"destilação crítica genérica": mesmo dado, mesma arquitetura, mesma referência,
mesmo KD (l_soft_gated), mesmo scheduler de λ, mesma calibração de thresholds —
sem qualquer informação de ONDE no tempo o risco emerge.

Justificativa: é a menor intervenção que zera exatamente o conjunto de termos que
usam t0, preservando idênticos todos os componentes não-temporais. Escada causal
resultante: Baseline (sem alinhamento) → NoTemp (alinhamento sem ancoragem) →
AF-TOI completo. A comparação NoTemp vs AF-TOI atribui o efeito especificamente à
ancoragem temporal; Baseline vs NoTemp atribui o que vem da destilação genérica.

## Distinção para o artigo (Fase 8)
- Tabela 11 atual = PARAMETER SENSITIVITY (α>0, seed 42, valores ≈).
- Nova Tabela 11 = MECHANISM REMOVAL (NoTemp) + sensitivity (α∈{0.05,0.15,0.30,0.50}),
  12 seeds, valores exatos, Wilcoxon exato pareado + Cliff's δ vs braço canônico.
- Braço α=0.15 REUSA exp_seed12_round1 (não retreinar).

## Execução (GPU da Angela)
```
cd FGCS\Inferencia
python ..\..\Experimentos-Artigo1-AdHoc\ablacao_mecanismo_temporal\run_mechanism_ablation.py
python ..\..\Experimentos-Artigo1-AdHoc\ablacao_mecanismo_temporal\analyze_mechanism_ablation.py
```
4 braços novos × 12 seeds ≈ 12–24 h de GPU (resumível por seed/braço; `--with-a0`
adiciona o braço α=0 como curiosidade diagnóstica). O analyze localiza os resumos
por seed automaticamente; se o nome das colunas do summary diferir, ajustar o
dicionário `ALIASES` (1 linha).
