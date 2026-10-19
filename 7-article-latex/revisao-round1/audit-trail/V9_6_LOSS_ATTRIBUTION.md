# Evidência da atribuição das duas funções de perda (v9.6, 19/10/2026)

Cadeia de evidência que sustenta a dupla linhagem declarada na §3.2/Tab. 6:

1. **Campanha principal (Tabelas 8–10, 12).** O treinador é
   `1-pipeline-tera/tera_pipeline/training/tera_train.py` (este repositório;
   sha256/16 `fd1e58b34d10cfc6`), funcionalmente idêntico (diff só de molduras
   decorativas de comentário) à cópia "viva" que executou a campanha
   (FGCS/Arquitetura, mtime 19/05/2026 05:51 — a campanha exp_20260519 iniciou
   05:59). Perda implementada: `aftkd_loss` = BCE episódica + cross-entropy
   ponderada em probabilidades temperadas (τ=2,0), com pesos w_i(t): 1 nos
   não-críticos (NÃO normalizados), decaimento exp pós-onset ×(λ_late=3,5 se
   σ(z_s)<θ_late=0,15) ×3,0 nos 8 frames pré-onset nos críticos (normalizados
   por episódio). Schedule λ_KD: 0 (10 ép.) → 0,20–1,00 (17) → 1,00.
   Config congelada: `configs/experiment_config.yaml` (hash d5327a24d1d2f7f9,
   amarrado pelos manifests das duas campanhas).
2. **Protocolo de replicação (replay §4.10–4.11; ablação Tab. 11).** Perda
   completa (L_hard rampa + L_align MSE + L_ep + composto 0,60/1,60/1,20/1,50
   + L_late/L_pre/L_tail/L_neg; warmup 12/ramp 16) em
   `5-robustness-replay-runv29/Inferencia/run_v29_ablacao_ttdef_ajuste_gatting.py`
   (config A); teste-trava `test_aftoi_loss_manifest.py` amarra §3.2/Tab. 6 a
   esse código e, desde a v9.6, também amarra a linhagem do pipeline
   (test_pipeline_objective_lineage).
3. **Separação dos checkpoints.** Modelos da campanha (19/05, reutilizados na
   expansão 27/08; ex.: student_seed42 sha/12 9f58c7611ec3) ≠ abl_A de abril
   (29/04; 31b42b2ecb55; maxdiff de pesos 0,83). Assinaturas comportamentais
   distintas sob a MESMA avaliação numpy por janelas (seed 42, teste):
   não-críticos last-6 médio 0,0001 vs 0,095; fração pós-onset ≥0,78: 45,7%
   vs 0%.
4. **Braço canônico da ablação.** `mechabl_a015/student_seed42.pt` ==
   `abl_A/student_seed42.pt` BIT A BIT (reuso do treino original do mesmo
   protocolo; declarado na Tab. 11, no texto e no protocol map desde a v9.6).

**Limites da verificação (declarados):** o manifest da campanha registra
config_hash e generator_hash, não um hash do código de treino; a identidade
do treinador vivo é estabelecida por mtime + equivalência funcional do
código, e a separação de linhagens por hashes de checkpoint e assinaturas
comportamentais — não há log de console do treino preservado. Nada disso
altera resultados; define a que modelos cada formalização se refere.
