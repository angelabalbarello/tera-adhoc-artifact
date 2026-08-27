#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPANSAO DE SEEDS 4 -> 12 NO PIPELINE CORRETO (TERA Pipeline v1.0)
===================================================================
Revisao Ad Hoc Networks (ADHOC-D-26-02099), resposta a R2.5/R3.6.

Este launcher substitui o antigo seed-expansion (run_v29), que executava o
pipeline ERRADO: a Table 8 do artigo foi gerada pelo TERA Pipeline v1.0
(campanha exp_20260519_055950), nao pelo run_v29_ablacao de FGCS/Inferencia.

PROTOCOLO CONGELADO (nao alterar):
  * Seeds 42..53, n=12, definidas ANTES de conhecer qualquer p-valor.
  * O n NAO sera aumentado nem reduzido em funcao do resultado.
  * Todas as 12 seeds serao reportadas, qualquer que seja o desfecho.
  * Nenhuma tabela do artigo ou rebuttal e alterada automaticamente:
    apos a campanha, rode check_reproduction_42_45.py e stats_12seeds.py
    e leve os relatorios para decisao humana.

GARANTIAS DE PIPELINE CONGELADO (verificadas antes de executar):
  * generator_hash (synthetic_driver_risk_v7.py) == 65bbec4a23050639
  * config_hash (configs/experiment_config.yaml) == d5327a24d1d2f7f9
  * usa a copia CONGELADA de tera_pipeline (1-pipeline-tera/), estado de
    19/05/2026 — a copia "viva" de FGCS/Arquitetura divergiu apos a campanha
    (tera_utils.py 25/06, tera_figures.py 02/06) e NAO deve ser usada.

USO (a partir desta pasta, com o venv que tem torch+CUDA):
    python run_seed_expansion_tera.py
Tempo estimado: campanha original de 4 seeds levou ~36 min; 12 seeds ~1,5-2 h.
Validacao apos o termino:
    python check_reproduction_42_45.py
    python stats_12seeds.py
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE_DIR = HERE.parent.parent / "1-pipeline-tera"
EXPECTED_GENERATOR_HASH = "65bbec4a23050639"
EXPECTED_CONFIG_HASH = "d5327a24d1d2f7f9"
SEEDS = list(range(42, 54))  # 42..53 — CONGELADO. Nao editar em funcao do p-valor.
EXP_ID = "exp_seed12_round1"


def sha16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def main() -> None:
    if not (PIPELINE_DIR / "run_experiment.py").exists():
        sys.exit(f"ERRO: {PIPELINE_DIR}/run_experiment.py nao encontrado.")

    # 1) gerador congelado
    gen = PIPELINE_DIR / "synthetic_driver_risk_v7.py"
    gh = sha16(gen.read_bytes())
    if gh != EXPECTED_GENERATOR_HASH:
        sys.exit(f"ABORTADO: generator_hash={gh}, esperado {EXPECTED_GENERATOR_HASH}. "
                 "O gerador TERA-Gen nao e o mesmo da campanha publicada.")

    # 2) config congelada
    try:
        import yaml
    except ImportError:
        sys.exit("ERRO: pyyaml nao instalado (pip install pyyaml).")
    cfg = yaml.safe_load((PIPELINE_DIR / "configs" / "experiment_config.yaml")
                         .read_text(encoding="utf-8"))
    ch = sha16(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode())
    if ch != EXPECTED_CONFIG_HASH:
        sys.exit(f"ABORTADO: config_hash={ch}, esperado {EXPECTED_CONFIG_HASH}. "
                 "experiment_config.yaml nao e o mesmo da campanha publicada.")

    # 3) campanha ja existente?
    exp_dir = PIPELINE_DIR / "results" / EXP_ID
    if (exp_dir / "manifest.json").exists():
        sys.exit(f"ABORTADO: {exp_dir} ja contem um manifest.json. "
                 "Para repetir, mova/apague a pasta manualmente (decisao humana).")

    print(f"[seed-expansion-tera] hashes OK (gen={gh}, cfg={ch})")
    print(f"[seed-expansion-tera] SEEDS = {SEEDS} (protocolo congelado)")
    print(f"[seed-expansion-tera] exp-id = {EXP_ID}")
    print(f"[seed-expansion-tera] resultados em: {exp_dir}")

    cmd = [sys.executable, "run_experiment.py",
           "--seeds", ",".join(str(s) for s in SEEDS),
           "--exp-id", EXP_ID]
    print("[seed-expansion-tera] executando:", " ".join(cmd))
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    ret = subprocess.call(cmd, cwd=str(PIPELINE_DIR), env=env)
    if ret != 0:
        sys.exit(f"run_experiment.py terminou com codigo {ret}")

    print("\n[seed-expansion-tera] concluido.")
    print("Proximos passos (nesta pasta):")
    print("  1. python check_reproduction_42_45.py   # valida 42-45 vs Table 8")
    print("  2. python stats_12seeds.py              # relatorio 12 seeds (sem tocar no artigo)")


if __name__ == "__main__":
    main()
