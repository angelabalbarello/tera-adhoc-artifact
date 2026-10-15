"""
gera_borderline_macros.py  — TERA Pipeline
Uso: python gera_borderline_macros.py borderline_frame_logs.csv
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path

THETA_TTD = 0.10
M_DETECT  = 3

def fmt(v):
    return f"{v:.3f}".replace(".", "{,}")

def would_alert(probs, theta, m):
    count = 0
    for p in probs:
        count = count + 1 if p >= theta else 0
        if count >= m:
            return True
    return False

def compute_category(df, cat_value):
    sub = df[df["risk_level"] == cat_value]
    if sub.empty:
        return {}
    mx, mn, fr, wa = [], [], [], []
    for (_, _), g in sub.groupby(["seed", "episode_id"]):
        p = g["p_t"].values
        mx.append(p.max())
        mn.append(p.mean())
        fr.append((p >= THETA_TTD).mean())
        wa.append(would_alert(p, THETA_TTD, M_DETECT))
    return {
        "n":  len(mx),
        "mx": round(float(np.mean(mx)), 3),
        "sd": round(float(np.std(mx)),  3),
        "mn": round(float(np.mean(mn)), 3),
        "fr": round(float(np.mean(fr)) * 100, 1),
        "pc": round(float(np.mean(wa)) * 100, 1),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arquivo", nargs="?", default=None)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()
    caminho = args.arquivo or args.log
    if not caminho:
        parser.error("Uso: python gera_borderline_macros.py borderline_frame_logs.csv")

    df = pd.read_csv(caminho)
    print(f"Lido: {len(df):,} linhas")
    print(f"risk_level valores: {sorted(df['risk_level'].unique())}")
    print()

    niveis = df["risk_level"].unique()
    cat_at = next((v for v in niveis if "ten" in str(v).lower()), None)
    cat_al = next((v for v in niveis if "lert" in str(v).lower()), None)

    if not cat_at or not cat_al:
        print(f"ERRO: categorias nao encontradas. Valores: {list(niveis)}")
        raise SystemExit(1)

    print(f"Atenção → '{cat_at}'  |  Alerta → '{cat_al}'")
    print()

    at = compute_category(df, cat_at)
    al = compute_category(df, cat_al)

    print("% ── Borderline analysis macros ─────────────────────────────")
    print(f"\\newcommand{{\\BLnAt}}{{{at['n']}}}")
    print(f"\\newcommand{{\\BLnAl}}{{{al['n']}}}")
    print(f"\\newcommand{{\\BLmxAt}}{{{fmt(at['mx'])}}}")
    print(f"\\newcommand{{\\BLsdAt}}{{{fmt(at['sd'])}}}")
    print(f"\\newcommand{{\\BLmxAl}}{{{fmt(al['mx'])}}}")
    print(f"\\newcommand{{\\BLsdAl}}{{{fmt(al['sd'])}}}")
    print(f"\\newcommand{{\\BLfrAt}}{{{fmt(at['fr'])}}}")
    print(f"\\newcommand{{\\BLfrAl}}{{{fmt(al['fr'])}}}")
    print(f"\\newcommand{{\\BLpcAt}}{{{fmt(at['pc'])}}}")
    print(f"\\newcommand{{\\BLpcAl}}{{{fmt(al['pc'])}}}")
    print()

    print("NÚMEROS COMPLETOS:")
    for nome, d in [("ATENÇÃO", at), ("ALERTA", al)]:
        print(f"  [{nome}] n={d['n']}  mx={d['mx']}±{d['sd']}  "
              f"fr={d['fr']}%  alerta={d['pc']}%")

    print()
    mx_at, mx_al = at['mx'], al['mx']
    if mx_at < mx_al:
        print(f"✓ Ordenamento correto: Atenção({mx_at}) < Alerta({mx_al})")
    else:
        print(f"⚠  Atenção({mx_at}) >= Alerta({mx_al}) — adicione footnote no artigo")

if __name__ == "__main__":
    main()
