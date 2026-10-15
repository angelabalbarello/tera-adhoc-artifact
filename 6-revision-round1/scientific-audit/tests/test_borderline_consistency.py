"""Tab.13/Fig.11 x FR: reconciliacao semantica e numerica (seed 42, stream prefix-window).

Verifica, na MESMA populacao e MESMO stream:
  1. FR >= fracao de episodios com p_max < theta (relacao semantica);
  2. FR do log corrigido == FR publicado (expanded_per_seed, seed 42);
  3. valores da Tabela 13 (Normal/Critical) == estatisticas do log corrigido;
  4. os valores do artefato invalido (0.271/70.8%) nao retornam ao artigo.
"""
import pathlib, sys
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[3]
ART  = pathlib.Path(sys.argv[1] if len(sys.argv)>1 else "/sessions/relaxed-lucid-thompson/mnt/Artigo 1/revisao-round1")
REC  = ROOT/"6-revision-round1/scientific-audit/results/tab13_fr_reconciliation_seed42.csv"
PS   = ROOT/"6-revision-round1/seed-expansion-tera/expanded_per_seed.csv"

def test_semantic_and_values():
    df=pd.read_csv(REC)
    ps=pd.read_csv(PS)
    for cfg in ["baseline_fixed","aftkd_fixed"]:
        c=df[(df.config==cfg)&(df.is_critical==1)]
        assert len(c)==120
        fr=float(c.missed.mean()); frac=float((c.p_max<0.10).mean())
        assert fr >= frac - 1e-9                      # relacao semantica
        pub=float(ps[(ps.Seed==42)&(ps.config_id==cfg)].FailRate.iloc[0])
        assert abs(fr-pub) < 1e-9                     # == FR publicado
    c=df[(df.config=="aftkd_fixed")&(df.is_critical==1)]
    assert abs(c.p_max.mean()-0.940)<0.0015 and abs(c.p_max.std(ddof=1)-0.083)<0.0015
    assert float((c.p_max>=0.10).mean())==1.0

def test_article_values():
    art=(ART/"artigo.tex").read_text(errors="replace")
    assert "0.940 \\pm 0.083" in art
    assert "$0.271 \\pm 0.178$" not in art and "70.8" not in art
    assert "stateless" not in art  # historico pertence ao rebuttal

if __name__=="__main__":
    test_semantic_and_values(); test_article_values(); print("borderline consistency OK")
