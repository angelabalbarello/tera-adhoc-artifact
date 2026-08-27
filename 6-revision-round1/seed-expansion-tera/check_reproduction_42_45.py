#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validacao critica pre-integracao (item 6 da auditoria da revisao):
as seeds 42-45 regeneradas pela nova campanha (exp_seed12_round1) devem
reproduzir os valores da campanha publicada exp_20260519_055950, que formou
a Table 8 do artigo submetido.

Compara, por (seed, config_id), todas as metricas numericas de
results_all_seeds.csv. Tambem confere generator_hash entre manifests.

As metricas dividem-se em duas classes:
  DETERMINISTICAS (F1, ECE, FailRate, TTD*, TTDef*, SkipPct, estratificacoes):
    dependem apenas de seed + codigo + dados -> exige-se reproducao exata.
  RELOGIO DE PAREDE (Lat_ms, Cost_ms_per_frame): medem tempo real de execucao
    e dependem do estado da maquina no momento (carga, termica) -> drift e
    esperado entre campanhas; sao reportadas a parte, sem reprovar.

Criterio (apenas sobre as deterministicas):
  PASS   — diferenca absoluta <= 1e-9 (determinismo pleno)
  WARN   — diferenca <= 1e-3 (drift numerico de versao torch/GPU; discutir)
  FAIL   — diferenca  > 1e-3 (NAO integrar; investigar)

Nao altera nenhum arquivo do artigo. Apenas relata.
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
WALL_CLOCK = {"Lat_ms", "Cost_ms_per_frame"}  # medidas de tempo real: drift esperado


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
            sys.exit(f"ERRO: {p} nao encontrado.")

    # hashes dos manifests
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
    diffs = []       # deterministicas fora de 1e-9
    wall_diffs = []  # relogio de parede (informativo)
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
                    wall_diffs.append((key, col, a, b, d))
                continue
            n_det += 1
            if d > worst:
                worst, worst_key = d, (key, col, a, b)
            if d > 1e-9:
                diffs.append((key, col, a, b, d))

    print(f"\n{n_det} valores DETERMINISTICOS comparados em {len(ref)} linhas "
          f"(seed x config).")
    if wall_diffs:
        print(f"[informativo] {len(wall_diffs)} valores de relogio de parede "
              f"(Lat_ms/Cost_ms_per_frame) divergiram — esperado entre campanhas "
              f"(dependem da carga da maquina). Maior: "
              f"{max(w[4] for w in wall_diffs):.3f} ms.")

    if not diffs:
        print("\nPASS: reproducao exata das metricas deterministicas "
              "(todas as diferencas <= 1e-9).")
        print("As seeds 42-45 da nova campanha reproduzem a Table 8. "
              "Pode-se integrar as 12 seeds como replicacoes homogeneas.")
        print("Nota: a coluna ms/f (custo) do agregado n=12 deve usar apenas a "
              "campanha nova (medicao homogenea), e a diferenca de escala vs a "
              "Table 8 publicada deve ser mencionada na integracao.")
        return

    print(f"\n{len(diffs)} diferencas deterministicas acima de 1e-9. "
          f"Maior: {worst:.3e} em {worst_key}")
    for key, col, a, b, d in sorted(diffs, key=lambda x: -x[4])[:25]:
        print(f"  seed={key[0]} config={key[1]:>16s} {col:>22s}: "
              f"ref={a:.6f} novo={b:.6f} diff={d:.3e}")
    if worst <= 1e-3:
        print("\nWARN: drift numerico pequeno (<= 1e-3). Provavel diferenca de "
              "versao torch/driver. Discutir antes de integrar; nao editar o artigo.")
    else:
        print("\nFAIL: diferencas relevantes (> 1e-3). NAO integrar as 12 seeds. "
              "O pipeline nao esta reproduzindo a campanha publicada; investigar "
              "antes de qualquer uso.")
        sys.exit(1)


if __name__ == "__main__":
    main()
