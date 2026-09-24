"""Valores-manchete das Tabelas 8, 11 e 20 derivam dos CSVs arquivados."""
import pandas as pd, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
def test_table8_headline():
    s=pd.read_csv(ROOT/"6-revision-round1/seed-expansion-tera/expanded_summary.csv").set_index('config_id')
    af=s.loc['aftkd_hybrid']; bf=s.loc['baseline_fixed']; bh=s.loc['baseline_hybrid']
    assert abs(af.F1_mean-0.846)<5e-4 and abs(af.FailRate_mean-0.001)<5e-4
    assert abs(af.TTDef_mean-0.099)<5e-4 and abs(af.Cost_ms_per_frame_mean-0.340)<5e-4
    assert abs(af.SkipPct_mean-48.130)<5e-3
    assert abs(bf.TTDef_mean-0.500)<5e-4 and abs(bh.TTDef_mean-0.570)<5e-4
    assert abs(bh.TTD_mean-0.223)<5e-4  # celula preenchida na Tab. 8
    assert abs((bh.TTDef_mean-bf.TTDef_mean)/bf.TTDef_mean-0.139)<0.002  # 13.9%
def test_table11_headline():
    m=pd.read_csv(ROOT/"6-revision-round1/ablacao_mecanismo_temporal/mechanism_ablation_per_seed.csv")
    g=m.groupby('arm').mean(numeric_only=True)
    assert abs(g.loc['mechabl_notemp','TTDef']-0.250)<5e-3
    assert abs(g.loc['mechabl_a015','TTDef']-0.043)<5e-3
    assert abs(g.loc['mechabl_notemp','TTDef_hyb']-0.180)<5e-3
    assert abs(g.loc['mechabl_a015','TTDef_hyb']-0.051)<5e-3
    assert abs(g.loc['mechabl_notemp','ECE']-0.642)<5e-3
    assert abs(g.loc['mechabl_notemp','SupPct']-59.0)<0.1
def test_table20_headline():
    c=pd.read_csv(ROOT/"6-revision-round1/skiprnn_baseline/comparison_skiprnn_mheg.csv")
    po=pd.read_csv(ROOT/"6-revision-round1/skiprnn_baseline/preonset_occupancy.csv")
    af=c[c.system=="AF-TOI+MHEG (frozen)"]
    assert abs(af.FR.mean()-0.004)<1e-3 and abs(af.TTDef.mean()-0.101)<1e-3
    assert abs(af.SupPct.mean()-48.7)<0.1 and abs(af.F1.mean()-0.998)<1e-3
    m=c[c.system.str.contains("matched-sup")].sort_values("seed")  # 1 linha por seed
    assert len(m)==4 and abs(m.SupPct.mean()-50.9)<0.2
    assert m.FR.max()==0.0 and m.TTDef.mean()<0.001 and abs(m.F1.mean()-0.998)<1e-3
    mx=c[c.system.str.contains("safe, lam=0.005")]
    assert abs(mx.SupPct.mean()-93.6)<0.1 and abs(mx.F1.mean()-0.866)<1e-3
    agg=po.groupby("system")[["PreOcc","NormOcc"]].mean()
    assert abs(agg.loc["Skip-RNN matched","PreOcc"]-0.211)<0.005
    assert abs(agg.loc["Skip-RNN maxsup","PreOcc"]-0.375)<0.005
    assert abs(agg.loc["Skip-RNN maxsup","NormOcc"]-0.109)<0.005
    assert abs(agg.loc["AF-TOI+MHEG","PreOcc"]-0.051)<0.005
if __name__ == "__main__":
    test_table8_headline(); test_table11_headline(); test_table20_headline(); print("table values OK")
