import pandas as pd, numpy as np

df = pd.read_csv("borderline_frame_logs.csv")

# Só AFKD (configuração adotada), máx de p_t por episódio
kd = df[df["config"] == "afkd_fixed"]
ep = kd.groupby(["risk_level", "episode_id"])["p_t"].max().reset_index()

for lvl in ["Normal", "Atenção", "Alerta", "Crítico"]:
    sub = ep[ep.risk_level == lvl]["p_t"]
    frac = (sub >= 0.10).mean() * 100
    print(f"{lvl}: {sub.mean():.3f} ± {sub.std():.3f}  frac≥θ={frac:.0f}%")