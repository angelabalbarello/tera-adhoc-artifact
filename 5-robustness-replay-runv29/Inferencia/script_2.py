import pandas as pd

df = pd.read_csv("borderline_frame_logs.csv")
kd = df[(df["config"]=="afkd_fixed") & (df["risk_level"]=="Normal")]

# Separar por origem da receita
print("=== Normal por receita (p_max por episódio) ===")
ep = kd.groupby(["recipe","episode_id"])["p_t"].max().reset_index()
print(ep.groupby("recipe")["p_t"].agg(["mean","std","count"]).round(3))