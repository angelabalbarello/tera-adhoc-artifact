# OBSOLETO — NÃO USAR

Este launcher (`run_seed_expansion.py`) executava a expansão 4→12 seeds no
`run_v29_ablacao_ttdef_ajuste_gatting.py` (FGCS/Inferencia). Auditoria de
27/08/2026 mostrou que **a Table 8 do artigo não foi gerada por esse pipeline**,
e sim pelo TERA Pipeline v1.0 (campanha `exp_20260519_055950`) — os regimes de
métricas são incompatíveis (ex.: F1 Baseline 0,357 na Table 8 vs 0,85 no run_v29).

A execução foi interrompida em 27/08 após a seed 42 (que, aliás, reproduziu
fielmente o resultado arquivado do abl_A — o run_v29 é reprodutível, mas
reproduz a campanha errada).

Use no lugar: `../seed-expansion-tera/run_seed_expansion_tera.py`.

Mantido apenas para registro da trilha de auditoria da revisão.
