#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Confere se as seeds 42-45 da campanha nova reproduzem a campanha publicada.

Compara results_all_seeds.csv de exp_seed12_round1 com o de
exp_20260519_055950, por (seed, config). Metricas deterministicas (F1, ECE,
FR, TTD, TTDef, SkipPct e estratificacoes) precisam coincidir; Lat_ms e
Cost_ms_per_frame medem tempo real de execucao e variam com a carga da
maquina, entao sao reportadas a parte sem reprovar.

Criterio sobre as deterministicas: PASS com diferenca <= 1e-9; WARN ate 1e-3
(drift de versao torch/driver, discutir antes de integrar); FAIL acima disso.
So relata; nao altera nenhum arquivo do artigo.
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
NEW_DIR = ROOT / "1-pipeline-tera" / "results" / "exp_seed12_round1"
REF_DIR = ROOT / "2-campaign-exp_20260519_055950"
SEEDS_REF = ["42", "43", "44", "45"]
WALL_CLOCK = {"Lat_ms", "Cost_ms_per_frame"}


def load_rows(p: Path) -> dict:
    rows = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Seed"] in SEEDS_REF:
                rows[(r["Seed"], r["config_id"])] = r
    return rows


def main() -> None:
    new_csv = NEW_DIR / "metrics" / "results_all_seeds.csv"
    ref_csv = REF_DIR / "metrics" / "results_all_seeds.csv"
    for p in (new_csv, ref_csv):
        if not p.exists():
            sys.exit(f"{p} nao encontrado")

    for name, d in (("nova", NEW_DIR), ("publicada", REF_DIR)):
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        print(f"manifest {name}: exp={m['experiment_id']}  "
              f"gen={m['generator_hash']}  cfg={m['config_hash']}")

    ref = load_rows(ref_csv)
    new = load_rows(new_csv)

    missing = sorted(set(ref) - set(new))
    if missing:
        sys.exit(f"FAIL: combinacoes ausentes na nova campanha: {missing}")

    worst = 0.0
    worst_key = None
    n_det = 0
    diffs = []
    wall_diffs = []
    for key, r_ref in sorted(ref.items()):
        r_new = new[key]
        for col, v in r_ref.items():
            if col in ("Seed", "config_id", "Modelo", "Gating"):
                continue
            try:
                a, b = float(v), float(r_new[col])
            except (ValueError, KeyError):
                continue
            d = abs(a - b)
            if col in WALL_CLOCK:
                if d > 1e-9:
                    wall_diffs.append(d)
                continue
            n_det += 1
            if d > worst:
                worst, worst_key = d, (key, col, a, b)
            if d > 1e-9:
                diffs.append((key, col, a, b, d))

    print(f"\n{n_det} valores deterministicos comparados em {len(ref)} linhas.")
    if wall_diffs:
        print(f"nota: {len(wall_diffs)} valores de relogio de parede divergem "
              f"(esperado; maior diferenca {max(wall_diffs):.3f} ms)")

    if not diffs:
        print("\nPASS: reproducao exata das metricas deterministicas (<= 1e-9).")
        print("As 12 seeds podem ser integradas como replicacoes homogeneas; "
              "a coluna ms/f do agregado deve usar apenas a campanha nova.")
        return

    print(f"\n{len(diffs)} diferencas acima de 1e-9; maior: {worst:.3e} em {worst_key}")
    for key, col, a, b, d in sorted(diffs, key=lambda x: -x[4])[:25]:
        print(f"  seed={key[0]} config={key[1]:>16s} {col:>22s}: "
              f"ref={a:.6f} novo={b:.6f} diff={d:.3e}")
    if worst <= 1e-3:
        print("\nWARN: drift numerico pequeno; discutir antes de integrar.")
    else:
        print("\nFAIL: o pipeline nao reproduz a campanha publicada; "
              "nao integrar antes de investigar.")
        sys.exit(1)


if __name__ == "__main__":
    main()
