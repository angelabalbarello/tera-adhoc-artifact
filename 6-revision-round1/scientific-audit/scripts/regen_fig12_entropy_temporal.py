#!/usr/bin/env python3
"""Regenera fig_calibration_entropy_temporal.pdf a partir das saidas ARQUIVADAS
da execucao original (frame_probs_*_seed42.npy + rotulos), usando o modulo
corrigido do artefato (titulo neutro). Nenhuma inferencia ou treino."""
import sys, importlib.util
from pathlib import Path
import numpy as np

ART = Path("/sessions/zealous-lucid-ride/mnt/TreinamentoNovo/tera-adhoc-artifact/3-calibration-posthoc/tera_posthoc_calibration_experiment.py")
EXP = Path("/sessions/zealous-lucid-ride/mnt/TreinamentoNovo/FGCS/Arquitetura/pipeline_completo/results")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home()/ "fig12/fig_calibration_entropy_temporal.pdf"

spec = importlib.util.spec_from_file_location("tpce", ART)
mod = importlib.util.module_from_spec(spec)
sys.modules["tpce"] = mod
spec.loader.exec_module(mod)

y_fr_te = np.load(EXP/"data/y_fr_te_seed42.npy")
y_ep_te = np.load(EXP/"data/y_ep_te_seed42.npy")
crit = y_ep_te == 1
trajectories = {}
for cid in mod.CALIB_LABELS:
    p = EXP/f"metrics/frame_probs_{cid}_seed42.npy"
    if p.exists():
        fp = np.load(p)[crit]
        trajectories[cid] = mod.build_entropy_trajectory(fp, y_fr_te[crit], y_ep_te[crit], frame_thr=mod.FRAME_THR)
print("configs:", sorted(trajectories))
mod.fig_entropy_temporal_curves(trajectories, OUT, seed_viz=42)
print("ok ->", OUT)
