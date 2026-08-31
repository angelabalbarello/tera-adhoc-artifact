#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Roda o fatorial com as seeds 42-53 no pipeline congelado (1-pipeline-tera).

Revisao ADHOC-D-26-02099, resposta a R2.5/R3.6. As seeds foram fixadas antes
de qualquer p-valor e o n nao muda em funcao do resultado; todas as seeds sao
reportadas. Antes de executar, o launcher confere os hashes do gerador e da
configuracao contra os da campanha publicada (exp_20260519_055950) e aborta
em caso de divergencia. Apos o termino, rodar check_reproduction_42_45.py e
stats_12seeds.py; nenhum arquivo do artigo e alterado por estes scripts.

Uso: python run_seed_expansion_tera.py   (venv com torch+CUDA; ~2-3 h)
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
SEEDS = list(range(42, 54))
EXP_ID = "exp_seed12_round1"


def sha16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def main() -> None:
    if not (PIPELINE_DIR / "run_experiment.py").exists():
        sys.exit(f"run_experiment.py nao encontrado em {PIPELINE_DIR}")

    gen = PIPELINE_DIR / "synthetic_driver_risk_v7.py"
    gh = sha16(gen.read_bytes())
    if gh != EXPECTED_GENERATOR_HASH:
        sys.exit(f"gerador divergente ({gh} != {EXPECTED_GENERATOR_HASH}); abortando")

    try:
        import yaml
    except ImportError:
        sys.exit("pyyaml nao instalado")
    cfg = yaml.safe_load((PIPELINE_DIR / "configs" / "experiment_config.yaml")
                         .read_text(encoding="utf-8"))
    ch = sha16(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode())
    if ch != EXPECTED_CONFIG_HASH:
        sys.exit(f"config divergente ({ch} != {EXPECTED_CONFIG_HASH}); abortando")

    exp_dir = PIPELINE_DIR / "results" / EXP_ID
    if (exp_dir / "manifest.json").exists():
        sys.exit(f"{exp_dir} ja contem manifest.json; mova ou apague manualmente "
                 "antes de repetir a campanha")

    print(f"hashes ok (gen={gh}, cfg={ch}); seeds={SEEDS}; exp-id={EXP_ID}")
    cmd = [sys.executable, "run_experiment.py",
           "--seeds", ",".join(str(s) for s in SEEDS),
           "--exp-id", EXP_ID]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    ret = subprocess.call(cmd, cwd=str(PIPELINE_DIR), env=env)
    if ret != 0:
        sys.exit(f"run_experiment.py terminou com codigo {ret}")

    print("campanha concluida; proximos passos:")
    print("  python check_reproduction_42_45.py")
    print("  python stats_12seeds.py")


if __name__ == "__main__":
    main()
