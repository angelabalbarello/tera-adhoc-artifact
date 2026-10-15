Figuras geradas a partir de episode_frame_logs.csv e, opcionalmente, borderline_frame_logs(2).csv.

Arquivos principais:
- fig_temporal_trace_combined.png/pdf: trajetória média de p_t ao redor do onset, por regime.
- fig_temporal_trace_progressive.png/pdf: versão isolada para regime progressive.
- fig_temporal_trace_abrupt.png/pdf: versão isolada para regime abrupt.
- fig_entropy_dist_all.png/pdf: boxplot da entropia H(p_t) por fase temporal.
- fig_entropy_dist_progressive.png/pdf: entropia por fase no regime progressive.
- fig_entropy_dist_abrupt.png/pdf: entropia por fase no regime abrupt.
- fig_entropy_dist_density.png/pdf: distribuição global de incerteza frame-level.
- fig_gating_heatmap_combined.png/pdf: heatmap de supressão das duas configurações híbridas.
- fig_gating_heatmap_baseline_hybrid.png/pdf: heatmap isolado do baseline hybrid.
- fig_gating_heatmap_afkd_hybrid.png/pdf: heatmap isolado do AF-KD hybrid.

Arquivo opcional:
- fig_prob_dist_borderline_optional.png/pdf: gerado do arquivo borderline enviado, caso queira comparar com a figura já produzida.

Tabelas auxiliares:
- entropy_phase_summary.csv
- gating_phase_summary.csv

Configurações encontradas no log:
config
baseline_fixed     14976
afkd_fixed         14976
baseline_hybrid    14976
afkd_hybrid        14976

Regimes encontrados:
regime
progressive    27648
abrupt         18432
normal         13824
