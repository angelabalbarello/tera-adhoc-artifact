"""Compare a rescored CSV with expected/borderline_episode_level.csv and print
the Table 13 summaries (mean, SD with ddof=1, fraction with max_prob >= 0.10)."""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
new = pd.read_csv(sys.argv[1])
ref = pd.read_csv(ROOT / "expected" / "borderline_episode_level.csv")
m = new.merge(ref, on=["seed", "episode_idx"], suffixes=("_new", "_ref"))
assert (m.recipe_id_new == m.recipe_id_ref).all()
d = (m.max_prob_new - m.max_prob_ref).abs().max()
print(f"episodes compared: {len(m)}; max |delta max_prob| = {d:.2e}")
for name, df in (("expected", ref[ref.seed.isin(new.seed.unique())]), ("rescored", new)):
    for cat in ("Atencao", "Alerta"):
        x = df[df.categoria == cat].max_prob
        print(f"{name:9s} {cat:8s} n={len(x):4d} mean={x.mean():.3f} sd={x.std(ddof=1):.3f} "
              f"frac>=0.10={100*(x >= 0.10).mean():.1f}%")
