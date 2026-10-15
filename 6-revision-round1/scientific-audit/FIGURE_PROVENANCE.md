# Proveniência das figuras do artigo (revisão round 1)

| Figura no artigo | Arquivo em image/ | Gerador (neste repositório) | Dados |
|---|---|---|---|
| Figs. 4–6 | 4-fig_entropy_dist_en / 5-fig_temporal_trace_en / 6-fig_gating_heatmap_en | scientific-audit/scripts/regen_dense_figs_en_v2.py | 5-robustness-replay-runv29/Inferencia/episode_frame_logs_v2.csv (CANÔNICO: gerado na GPU em 23/09 pelo generate_episode_frame_logs.py corrigido; skip rates 56.2%/40.1% batem com o CSV do run_v29; a reconstrução determinística ficou em episode_frame_logs_v2_reconstructed.csv, com rebuild_episode_frame_logs_v2.py) |
| Figs. 7–10 | 7-fig_tradeoff / 8–10 | 6-revision-round1/seed-expansion-tera/regen_figs_12seeds_en.py e regen_dense_figs_en.py | expanded_per_seed.csv, df_episodios_12seeds.csv |
| Fig. 11 | 11-fig_prob_dist_en | regen_dense_figs_en*.py (fig_prob_dist) | borderline logs |
| Fig. 12 | fig_calibration_entropy_temporal.pdf | 3-calibration-posthoc/tera_posthoc_calibration_experiment.py (fig_entropy_temporal_curves; salva com o MESMO nome) | rerun do experimento de calibração (requer torch; probs armazenadas) |
| Fig. 13 | fig_neg_control_operating_curve.pdf | scientific-audit/scripts/regen_fig13_locus_en.py (novo, 23/09 — título/fontes corrigidos) OU o rerun acima (fig_ece_vs_ttdef → fig_calibration_ece_vs_ttdef.pdf, renomear) | paper_calibration_macros.tex + expanded_per_seed.csv (seeds 42–45) |
| (ex-Fig. 14) | fig_neg_control_preonset_ap.pdf | REMOVIDA do artigo (score H_strat não sustenta claim; ver SCIENTIFIC_FIX_REPORT P0.7) | — |
| Fig. 14 (atual) | fig_temporal_scale_v2.pdf | temporal_scale_generalization_v2/make_figure.py | temporal_scale_generalization_v2/ |
