"""Trustworthy POR SEED: Baseline+MHEG viola em 3/12 (max 0.158); AF-TOI+MHEG passa 12/12 (max 0.017)."""
import pandas as pd, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
def test_per_seed_constraint():
    df=pd.read_csv(ROOT/"6-revision-round1/seed-expansion-tera/expanded_per_seed.csv")
    bh=df[df.config_id=='baseline_hybrid'].FailRate; ah=df[df.config_id=='aftkd_hybrid'].FailRate
    assert int((bh>0.05).sum())==3 and abs(bh.max()-0.1583)<1e-4
    assert int((ah>0.05).sum())==0 and abs(ah.max()-0.0167)<1e-4
    assert bh.mean()<=0.05  # o agregado sozinho NAO pode classificar seguranca
if __name__ == "__main__":
    test_per_seed_constraint(); print("per-seed coverage OK")
