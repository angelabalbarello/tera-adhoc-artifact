import pandas as pd

df = pd.read_csv("borderline_frame_logs.csv")
kd = df[df["config"] == "afkd_fixed"]

# p_last6 = média dos últimos 6 frames (= episode aggregator do treino)
ep = (
    kd.groupby(["risk_level", "episode_id"])
    .apply(lambda g: g.nlargest(6, "t")["p_t"].mean())
    .reset_index(name="p_last6")
)

print("=== p_max vs p_last6 por nível ===")
ep_pmax = kd.groupby(["risk_level","episode_id"])["p_t"].max().reset_index(name="p_max")
ep = ep.merge(ep_pmax, on=["risk_level","episode_id"])

for lvl in ["Normal", "Atenção", "Alerta", "Crítico"]:
    sub = ep[ep.risk_level == lvl]
    print(f"\n{lvl} (n={len(sub)}):")
    print(f"  p_max   = {sub.p_max.mean():.3f} ± {sub.p_max.std():.3f}")
    print(f"  p_last6 = {sub.p_last6.mean():.3f} ± {sub.p_last6.std():.3f}")
    print(f"  frac_max≥0.10  = {(sub.p_max >= 0.10).mean()*100:.0f}%")
    print(f"  frac_last6≥0.10 = {(sub.p_last6 >= 0.10).mean()*100:.0f}%")