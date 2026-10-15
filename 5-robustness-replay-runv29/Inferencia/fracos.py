import pandas as pd
df = pd.read_csv("borderline_frame_logs.csv")
kd = df[df["config"]=="afkd_fixed"]
ep = kd.groupby(["risk_level","episode_id"])["p_t"].max().reset_index(name="p_max")
for lvl in ["Normal","Atenção","Alerta","Crítico"]:
    sub = ep[ep.risk_level==lvl]["p_max"]
    frac = (sub >= 0.10).mean() * 100
    print(f"{lvl}: frac≥θ={frac:.0f}%")