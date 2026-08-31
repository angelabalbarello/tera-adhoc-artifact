# fig_ttdef_dist.py
# Requer: results_all_seeds.csv com colunas [config, seed, episode_id, ttdef]
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

df = pd.read_csv("results_all_seeds_unified.csv")

configs = [
    ("LSTM-Baseline",  "Baseline Fixo",    "#4C72B0"),
    ("Baseline+Gating","Baseline+MHEG",    "#DD8452"),
    ("LSTM-AF-TKD",    "AF-TKD Fixo",      "#55A868"),
    ("AF-TKD+Gating",  "AF-TKD+MHEG",      "#C44E52"),
]

fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharey=True, sharex=True)
bins = np.linspace(0, 10.5, 22)

for ax, (key, label, color) in zip(axes, configs):
    data = df[df["config"] == key]["ttdef"].values
    ax.hist(data, bins=bins, color=color, alpha=0.85, edgecolor="white",
            linewidth=0.4)
    ax.axvline(10.0, color="red", linestyle="--", linewidth=0.8,
               label="Penalidade\n(não detectado)")
    ax.set_title(label, fontsize=8, fontweight="bold")
    ax.set_xlabel(r"TTD$_\mathrm{ef}$ (s)", fontsize=7)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.tick_params(labelsize=7)

axes[0].set_ylabel("Episódios críticos", fontsize=8)
plt.tight_layout(w_pad=0.3)
plt.savefig("image/fig_ttdef_dist_pt.pdf", bbox_inches="tight", dpi=300)