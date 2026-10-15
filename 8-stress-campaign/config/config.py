"""Configuracao central: seeds, tamanhos e limiares dos experimentos."""
SEEDS = [42, 43, 44, 45]            # reprodutibilidade (paired entre metodos)
GATING = dict(L=60, tau_h=0.35, tau_delta=0.30, theta=0.5, n_episodes=200)
RETRIEVAL = dict(dim=32, prototypes=8, per_class=2, per_proto=40, queries=300)
LONGITUDINAL = dict(drivers=30, episodes=96, drift=0.03, n_anom=2, z_threshold=2.5)
POLICY = dict(drivers=300, weak_brs_frac=0.18, degrade_frac=0.08)
SYNTHETIC_DISCLAIMER = (
    "Os resultados desta execucao sao baseados em dados sintetico-controlados e devem ser "
    "interpretados como validacao metodologica preliminar, NAO como evidencia empirica em frota real."
)
