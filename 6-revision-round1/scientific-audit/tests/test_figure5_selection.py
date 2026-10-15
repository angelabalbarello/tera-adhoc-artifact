"""Fig. 5: criterio mediano reprodutivel -> prog ep36 (t0 5.42s), abrupto ep148 (t0 2.50s)."""
import pandas as pd, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
def test_median_selection():
    log=pd.read_csv(ROOT/"5-robustness-replay-runv29/Inferencia/episode_frame_logs_v2.csv")
    dv=log[(log.seed==42)&(log.is_critical==1)&(log.config=='afkd_hybrid')]
    exp={"abrupt":(148,2.50,(2.0,4.0)),"progressive":(36,5.42,(4.0,7.0))}
    for regime,(eid,t0e,(lo,hi)) in exp.items():
        cand=[]
        for ep,g in dv[dv.regime==regime].groupby('episode_id'):
            g=g.sort_values('t'); t0=float(g.t0_time_s.iloc[0])
            if not lo<=t0<=hi: continue
            pre=g[g.time_s<t0]
            if len(pre)<3: continue
            cand.append((ep,t0,float(g[g.time_s>=t0].H_t.mean())-float(pre.H_t.mean())))
        cand.sort(key=lambda x:x[2]); ep,t0,_=cand[len(cand)//2]
        assert ep==eid and abs(t0-t0e)<0.05, (regime,ep,t0)
if __name__ == "__main__":
    test_median_selection(); print("figure5 selection OK")
