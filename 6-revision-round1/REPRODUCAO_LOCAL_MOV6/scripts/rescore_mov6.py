"""Movement 6 (Table 13, Attention/Alert rows): inference only with the frozen
checkpoints in ../modelos_salvos. No training, no threshold change.

Usage (from the package root):
    python scripts/rescore_mov6.py 42            # Table 13 uses seed 42
    python scripts/rescore_mov6.py 42 43 44 45   # optional, all archived seeds
Output: out/rescored_episode_level_<seeds>.csv
"""
import sys, os, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "out"
WORK.mkdir(exist_ok=True)
# borderline_analysis_gpu_v23.py imports the generator from the cwd and reads
# checkpoints from ./modelos_salvos; run it from a work dir laid out that way.
for f in ("borderline_analysis_gpu_v23.py", "synthetic_driver_risk_v7.py"):
    shutil.copy2(ROOT / "src" / f, WORK / f)
if not (WORK / "modelos_salvos").exists():
    shutil.copytree(ROOT / "modelos_salvos", WORK / "modelos_salvos")
os.chdir(WORK); sys.path.insert(0, str(WORK))
import numpy as np, pandas as pd, torch
import borderline_analysis_gpu_v23 as B
gen = B.gen
assert Path(gen.__file__).resolve().parent == WORK
dev = torch.device("cpu")
seeds = [int(s) for s in sys.argv[1:]] or [42]
rows = []
for SEED in seeds:
    ck = B.MODEL_DIR / f"student_seed{SEED}.pt"
    st = B.MultiTaskLSTM(B.D, B.H, bi=False).to(dev)
    st.load_state_dict(torch.load(ck, map_location=dev)); st.eval()
    B.set_seed(SEED)
    X_b, _, _, _, metas = gen.build_dataset(gen.RECIPES_BORDERLINE, window=B.T,
        per_recipe=B.PER_RECIPE, seed=SEED, shuffle=False, SENSOR_LEVEL=4)
    fp = B.infer_episodes(st, X_b, dev)
    for i, m in enumerate(metas):
        m["_seed"] = SEED; m["episode_idx"] = i
    df = B.analyse_borderline(fp, metas, B.THETA, B.M_DETECT); df["seed"] = SEED
    rows.append(df)
out = WORK / ("rescored_episode_level_" + "_".join(map(str, seeds)) + ".csv")
pd.concat(rows, ignore_index=True).to_csv(out, index=False)
print("written", out)
