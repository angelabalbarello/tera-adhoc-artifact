# Temporal-Scale Generalization v2 — Relatório Final (decision gate)

Experimento autorizado ("autorizo", 07/set/2026). Executado 07–08/set/2026.
**Status: PARADO no decision gate (Part J). artigo.tex e rebuttal.tex NÃO foram tocados.**

## 1. Objetivo
Testar se o AF-TOI+MHEG generaliza para durações de onset fora do suporte de treino,
eliminando os dois confounders do v1: (i) **forma** — v1 usava rampas lineares
(mudava forma E duração juntas); v2 usa time-warp shape-preserving das curvas
ORIGINAIS do gerador; (ii) **orçamento** — v1 comparava FS@onset com computações
diferentes; v2 adiciona controle iso-custo (protocolo da Tabela 17).

## 2. Setup congelado
- 12 seeds canônicas (42–53), checkpoints de exp_seed12_round1, thresholds de
  calibration_summary.csv (θ=0.1; τΔ, τ_H por seed). Nada retreinado/recalibrado.
- Grid: d ∈ {2,6,10,14,18,22,31,42,50} frames = {0.21…5.21}s.
  Suporte de treino auditado: abrupt 0.42–1.04s, progressive 2.50–4.06s;
  gap 1.15–2.40s = interpolação OOD; 0.21s e 4.38–5.21s = extrapolação OOD.
- 156 episódios/célula (10 receitas críticas + 3 normais × 12), pareados entre
  durações via ep_seed = seed·10⁶ + k. 108 células × 4 configs.
- Métricas na convenção canônica (TTDef = média-sobre-detectados + FR×10;
  FS@onset original: janela [t0−5, t0+5], harmful = pos & suprimido & held<θ).

## 3. Validações (pré-requisito para confiar nos números)
- **Âncora exata**: com os índices de teste publicados (indices_te_seed42–45.npy),
  a maquinaria reproduz a campanha canônica com erro ZERO (dFR=0, dTTDef=0,
  dSkip=0.00pp nas 16 células). Seeds 46–53 (split re-derivado): dFR≤0.0084.
  → qualquer efeito abaixo é da intervenção de duração, não do ferramental.
- **Shape-preservation**: todos os testes passam — identidade em d=d0 exata
  (0.00e+00), round-trip exato, t0 invariante, canais de esqueleto invariantes,
  ponto de 90%-do-pico em s+d, intensidade terminal exata sob compressão.
  (Atenuação de pico sob expansão ≤0.164 por suavização: inerente, documentada.)
- **Iso-custo**: 108/108 células casadas com desvio ≤1.9% (tol 2%), todas via
  bisseção em τ_H (fallback em τΔ nunca foi necessário).

## 4. Resultado 1 — Generalização de escala temporal (thresholds congelados)
AF-TOI mantém a restrição FR≤0.05 em TODO o grid; o Baseline viola nos extremos:

| d (s) | regime | FR Base Fixed | FR AF Fixed | TTDef Base | TTDef AF |
|---|---|---|---|---|---|
| 0.21 | OOD-extrap | **0.097** | 0.000 | 2.12 | 0.52 |
| 0.62–1.04 | ID (abrupt) | 0.010–0.041 | ≤0.001 | 0.89–1.33 | 0.54 |
| 1.46–2.29 | OOD-interp | 0.004–0.006 | 0.001 | 0.48–0.64 | 0.43–0.54 |
| 3.23 | ID (progr.) | 0.011 | 0.010 | 0.44 | 0.40 |
| 4.38 | OOD-borda | 0.036 | 0.024 | 0.72 | 0.45 |
| 5.21 | OOD-extrap | **0.085** | 0.024 | 1.24 | 0.41 |

- Cruzamento FR=0.05 (média entre seeds): Baseline Fixed em 0.21s e 5.21s; AF-TOI
  **nunca** (máx da média 0.026). Por célula (seed×duração): AF-TOI excede 0.05 em
  7/108 células (todas em d≥3.23s, seeds 42–44, máx 0.108) vs 22/108 do Baseline
  Fixed (máx 0.250) e 32/108 do Baseline+MHEG (máx 0.333). Redigir o claim como
  restrição satisfeita EM AGREGADO por duração, reportando as violações por seed.
- TTDef do AF-TOI quase plano (0.40–0.66s) vs Baseline em U (0.44→2.12s).
- Coerente com o v1 (rampas lineares): o efeito NÃO era artefato de forma.

## 5. Resultado 2 — Qualidade de alocação em iso-custo (o resultado honesto)
Com Baseline+MHEG casado ao orçamento do AF-TOI+MHEG por célula (±2%):

- **FS@onset: SEM vantagem sistemática do AF-TOI.** 47 vitórias / 8 empates /
  53 derrotas em 108 células; diff médio +0.014 (levemente CONTRA o AF).
  Wilcoxon exato n=12: significativo só em d=0.21s (p=0.027, favorecendo o
  BASELINE: AF 0.098 vs Base 0.069, Cliff δ=+0.36; nesse mesmo ponto o Baseline
  casado tem FR=0.111, i.e. viola a cobertura); p≥0.21 nas demais durações;
  em d≥3.23s o AF-TOI é numericamente menor (n.s., δ até −0.16).
  [CORREÇÃO 08/09: versão anterior deste relatório invertia a direção.]
  → A diferença de FS@onset vista com thresholds congelados (0.10–0.14 vs
  0.12–0.31) era majoritariamente **efeito de orçamento**: o Baseline+MHEG
  congelado suprime mais (55–67% vs 47–51%) e por isso suprime mais no onset.
- **FR e TTDef em iso-custo: vantagem do AF-TOI persiste, concentrada no OOD.**
  FR: p=0.008 (0.21s), p=0.031 (0.62s), p=0.042 (5.21s); TTDef: p=0.003 (0.21s),
  p=0.021 (0.62s), p=0.007 (5.21s). Em 0.21s: FR 0.006 vs 0.111, TTDef 0.66 vs
  2.29. No miolo ID (1.46–3.23s): sem diferença significativa (p≥0.5).
- FS_excess (Parte C, diagnóstico): ambos os gates suprimem MENOS no onset do
  que globalmente (excess negativo em todo o grid), i.e. a supressão perto do
  onset é sub-proporcional nos dois sistemas.

## 6. Parte G — seeds fracas de F1
F1 (AF-TOI Fixed, média nas 9 durações): 43:0.03, 45:0.21, 51:0.26, 48:0.29 —
a fraqueza das seeds 43/45 do v1 PERSISTE sob warp shape-preserving, logo é
propriedade dos checkpoints (variância de treino), não do formato da rampa.
FR/TTDef dessas seeds permanecem dentro da restrição; o F1 episódico com θ
congelado é a métrica sensível. Não afeta as conclusões agregadas (estatística
pareada n=12 já as inclui).

## 7. Respostas às perguntas A–E do spec
- **A. Confounding de forma eliminado?** Sim — warp preserva forma normalizada
  exatamente (testes exatos); identidade em d=d0 é bit-exact; âncora zero-erro.
- **B. Confounding de orçamento eliminado?** Sim — 108/108 células casadas ≤1.9%
  via o próprio protocolo do artigo (bisseção em τ_H, tol 2%).
- **C. AF-TOI generaliza em escala temporal?** Sim, para a garantia central:
  FR≤0.05 mantido em todo o grid com thresholds congelados (máx 0.026), enquanto
  o Baseline viola nos dois extremos; TTDef estável. Vale em frozen E iso-custo.
- **D. FS@onset menor em custo igual?** **Não.** Em iso-custo os dois gates são
  estatisticamente indistinguíveis em 8 de 9 durações, e a única diferença
  significativa (d=0.21s) favorece o Baseline — num ponto em que o Baseline
  casado viola FR. A vantagem de FS@onset vista com thresholds congelados era
  efeito de orçamento.
- **E. Justifica mudança no manuscrito?** Sim, com enquadramento dual e honesto:
  (1) reivindicar generalização de escala temporal para FR/TTDef (forte, robusta,
  significativa no OOD); (2) NÃO reivindicar superioridade de FS@onset em custo
  igual — se mencionado, como resultado nulo/diagnóstico. Isso responde
  diretamente ao R3 Comment 4(ii) (durações de onset não variadas como fator):
  agora foram variadas, com controle de forma e custo.

## 8. Proposta condicionada (aguardando autorização)
1. Nova subseção 4.XX no artigo (3 parágrafos, em vermelho): protocolo (warp
   shape-preserving + iso-custo), Resultado 1 (FR/TTDef), Resultado 2 (nulo de
   FS@onset em iso-custo, declarado como limitação honesta) + Figura
   fig_temporal_scale_v2 + tabela table_temporal_scale_v2.
2. Reescrever R3 Comment 4 item (ii) no rebuttal com citação literal da nova
   subseção, substituindo "not varied as an independent experimental factor".

## 9. Artefatos
- `duration_only/frozen_results.csv` (432 linhas) + 216 npz de célula
- `iso_cost/iso_cost_results.csv` (216) + `paired_stats.json` + `full_paired_stats.json`
- `figures/fig_temporal_scale_v2.{pdf,png}` — 4 painéis com sombreado ID/gap/OOD
- `table_temporal_scale_v2.tex`
- `audit/anchor_validation_v2.csv`, `audit/original_curve_audit.json`
- `provenance/{model_hashes.json (24), frozen_thresholds.json, duration_grid.json}`
- Reprodução: `python3 -B run_partA.py && python3 -B run_partB.py` (resumíveis)
