"""Tabela 9: 7/12 zeros, 5 pares informativos, p=0.0625 (wilcox), delta=+0.361; TTDef p=0.021."""
import pandas as pd, pathlib
from scipy import stats
ROOT = pathlib.Path(__file__).resolve().parents[3]
CSV = ROOT/"6-revision-round1/seed-expansion-tera/expanded_per_seed.csv"
def load():
    df=pd.read_csv(CSV)
    b=df[df.config_id=='baseline_fixed'].set_index('Seed'); a=df[df.config_id=='aftkd_fixed'].set_index('Seed')
    return b,a
def test_fr_contrast():
    b,a=load(); d=(b.FailRate-a.FailRate)
    assert sorted(b.index)==list(range(42,54))
    assert int((d==0).sum())==7 and int((d>0).sum())==5 and int((d<0).sum())==0
    r=stats.wilcoxon(b.FailRate.values,a.FailRate.values,zero_method='wilcox',alternative='two-sided',mode='exact')
    assert abs(r.pvalue-0.0625)<1e-9
    gt=sum(x>y for x in b.FailRate for y in a.FailRate); lt=sum(x<y for x in b.FailRate for y in a.FailRate)
    assert abs((gt-lt)/144-0.3611)<0.001
def test_ttdef_contrast():
    b,a=load()
    r=stats.wilcoxon(b.TTDef.values,a.TTDef.values,zero_method='wilcox',alternative='two-sided',mode='exact')
    assert abs(r.pvalue-0.021)<0.001
if __name__ == "__main__":
    test_fr_contrast(); test_ttdef_contrast(); print("wilcoxon OK  [scipy]", stats.__name__)
