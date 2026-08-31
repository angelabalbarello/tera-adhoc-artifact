# -*- coding: utf-8 -*-
"""
run_experiment.py — TERA Pipeline v1.0
Ponto de entrada único e determinístico para os experimentos do artigo.

Avalia as quatro configurações do fatorial nas seeds pedidas, sem patches
manuais nem reruns parciais; a Baseline Hybrid é nativa do pipeline. O fluxo
vai do dataset ao export (CSV canônico e macros LaTeX) e termina gravando um
manifest.json com hashes de configuração e do gerador, para rastreabilidade.

Execução completa: python run_experiment.py
Estágios ou seeds específicos: --stages eval,export / --seeds 42
Reuso de artefatos existentes: --reuse-data --reuse-models
A configuração fica em configs/experiment_config.yaml.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Garantir que o pacote tera_pipeline está no path
sys.path.insert(0, str(Path(__file__).parent))

from tera_pipeline.utils.tera_utils import (
    load_config,
    set_global_seed,
    setup_logging,
    get_device,
)
from tera_pipeline.dataset.tera_gen import DatasetManager
from tera_pipeline.training.tera_train import TrainingManager
from tera_pipeline.calibration.tera_calibrate import CalibrationManager
from tera_pipeline.inference.tera_infer import InferenceManager
from tera_pipeline.evaluation.tera_eval import EvaluationEngine
from tera_pipeline.logging.tera_log import FrameLogger
from tera_pipeline.figures.tera_figures import FigureGenerator
from tera_pipeline.export.tera_export import ResultsExporter
from tera_pipeline.latex.tera_latex import LatexGenerator

# Configurações canônicas do fatorial 2×2
ALL_CONFIGS = ["baseline_fixed", "baseline_hybrid", "aftkd_fixed", "aftkd_hybrid"]
ALL_SEEDS   = [42, 43, 44, 45]
ALL_STAGES  = ["dataset", "training", "calibration", "inference",
               "evaluation", "logging", "figures", "export"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TERA Pipeline v1.0 — Framework Experimental Reproduzível"
    )
    p.add_argument(
        "--stages",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=ALL_STAGES,
        help=f"Estágios a executar (default: todos). Opções: {','.join(ALL_STAGES)}",
    )
    p.add_argument(
        "--seeds",
        type=lambda s: [int(x) for x in s.split(",")],
        default=ALL_SEEDS,
        help=f"Seeds a processar (default: {ALL_SEEDS})",
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Arquivo de configuração YAML",
    )
    p.add_argument(
        "--exp-id",
        type=str,
        default=None,
        help="ID do experimento (default: timestamp automático)",
    )
    p.add_argument(
        "--reuse-data",
        action="store_true",
        help="Reutilizar datasets existentes (pula geração)",
    )
    p.add_argument(
        "--reuse-models",
        action="store_true",
        help="Reutilizar modelos treinados existentes (pula treinamento)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida configuração sem executar nenhum estágio",
    )
    return p.parse_args()


def compute_config_hash(cfg: dict) -> str:
    """Hash SHA-256 do arquivo de configuração para rastreabilidade."""
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def get_generator_hash() -> str:
    """Hash do gerador de dataset para detectar mudanças."""
    gen_path = Path("synthetic_driver_risk_v7.py")
    if gen_path.exists():
        return hashlib.sha256(gen_path.read_bytes()).hexdigest()[:16]
    return "not_found"


def build_experiment_dir(exp_id: str, cfg: dict) -> Path:
    """Cria e retorna o diretório do experimento."""
    results_root = Path(cfg.get("results_root", "results"))
    exp_dir = results_root / exp_id
    for subdir in ["data", "models", "calibration", "metrics",
                   "logs", "figures", "latex"]:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
    return exp_dir


def write_manifest(
    exp_dir: Path,
    exp_id: str,
    cfg: dict,
    args: argparse.Namespace,
    stages_completed: list[str],
    timing: dict[str, float],
) -> None:
    manifest = {
        "experiment_id":    exp_id,
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "pipeline_version": "1.0",
        "config_hash":      compute_config_hash(cfg),
        "generator_hash":   get_generator_hash(),
        "seeds":            args.seeds,
        "stages_requested": args.stages,
        "stages_completed": stages_completed,
        "reuse_data":       args.reuse_data,
        "reuse_models":     args.reuse_models,
        "results_csv":      str(exp_dir / "metrics" / "results_all_seeds.csv"),
        "macros_tex":       str(exp_dir / "latex" / "paper_metrics_macros.tex"),
        "timing_seconds":   timing,
        "paper_claim": (
            f"All results generated by TERA Pipeline v1.0, {exp_id}. "
            "No manual post-processing was applied."
        ),
    }
    path = exp_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Atualiza link simbólico "latest"
    latest = exp_dir.parent / "latest"
    if latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(exp_dir.name)
    except OSError:
        pass  # Windows pode não suportar symlinks sem privilégios


def validate_seeds(seeds: list[int]) -> None:
    """Interrompe se seeds não cobrem o conjunto canônico [42,43,44,45]."""
    missing = set(ALL_SEEDS) - set(seeds)
    if missing:
        print(
            f"\nAVISO: Seeds faltantes: {sorted(missing)}\n"
            f"   Resultados com seeds incompletas não são publicáveis.\n"
            f"   Use --seeds 42,43,44,45 para o experimento completo.\n"
        )


def main() -> None:
    args  = parse_args()
    cfg   = load_config(args.config)
    exp_id = args.exp_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    device = get_device()

    # Validações iniciais
    validate_seeds(args.seeds)

    log = setup_logging(name="tera_pipeline", level="INFO")
    log.info("=" * 72)
    log.info(f"  TERA Pipeline v1.0  |  Experimento: {exp_id}")
    log.info(f"  Seeds:   {args.seeds}")
    log.info(f"  Estágios: {args.stages}")
    log.info(f"  Device:  {device}")
    log.info("=" * 72)

    if args.dry_run:
        log.info("DRY RUN: configuração validada. Nenhum estágio executado.")
        return

    exp_dir          = build_experiment_dir(exp_id, cfg)
    stages_completed = []
    timing           = {}

    # Stage 1: DATASET
    if "dataset" in args.stages:
        t0 = time.time()
        log.info("\n[Stage 1/8] DATASET")
        dm = DatasetManager(cfg=cfg, exp_dir=exp_dir, seeds=args.seeds)
        dm.run(reuse=args.reuse_data)
        stages_completed.append("dataset")
        timing["dataset"] = round(time.time() - t0, 1)
        log.info(f"  Dataset concluído ({timing['dataset']:.0f}s)")

    # Stage 2: TRAINING
    if "training" in args.stages:
        t0 = time.time()
        log.info("\n[Stage 2/8] TRAINING")
        tm = TrainingManager(cfg=cfg, exp_dir=exp_dir, seeds=args.seeds,
                             device=device)
        tm.run(reuse=args.reuse_models)
        stages_completed.append("training")
        timing["training"] = round(time.time() - t0, 1)
        log.info(f"  Treinamento concluído ({timing['training']:.0f}s)")

    # Stage 3: CALIBRATION
    if "calibration" in args.stages:
        t0 = time.time()
        log.info("\n[Stage 3/8] CALIBRATION")
        cm = CalibrationManager(cfg=cfg, exp_dir=exp_dir, seeds=args.seeds,
                                device=device)
        cm.run()
        stages_completed.append("calibration")
        timing["calibration"] = round(time.time() - t0, 1)
        log.info(f"  Calibração concluída ({timing['calibration']:.0f}s)")

    # Stage 4: INFERENCE
    if "inference" in args.stages:
        t0 = time.time()
        log.info("\n[Stage 4/8] INFERENCE")
        log.info("  Configs: " + ", ".join(ALL_CONFIGS))
        im = InferenceManager(cfg=cfg, exp_dir=exp_dir, seeds=args.seeds,
                              device=device)
        im.run(configs=ALL_CONFIGS)
        stages_completed.append("inference")
        timing["inference"] = round(time.time() - t0, 1)
        log.info(f"  Inferência concluída ({timing['inference']:.0f}s)")

    # Stage 5: EVALUATION
    if "evaluation" in args.stages:
        t0 = time.time()
        log.info("\n[Stage 5/8] EVALUATION")
        ee = EvaluationEngine(cfg=cfg, exp_dir=exp_dir)
        results_df = ee.run(seeds=args.seeds, configs=ALL_CONFIGS)
        ee.print_summary(results_df)
        stages_completed.append("evaluation")
        timing["evaluation"] = round(time.time() - t0, 1)
        log.info(f"  Avaliação concluída ({timing['evaluation']:.0f}s)")

    # Stage 6: FRAME LOGGING
    if "logging" in args.stages:
        t0 = time.time()
        log.info("\n[Stage 6/8] FRAME LOGGING")
        seed_viz = cfg.get("seed_visualization", 42)
        log.info(f"  [Frame logs] seed_viz={seed_viz} (apenas figuras)")
        fl = FrameLogger(cfg=cfg, exp_dir=exp_dir, device=device)
        fl.generate_episode_logs(seeds=[seed_viz])
        fl.generate_borderline_logs(seeds=[seed_viz])
        stages_completed.append("logging")
        timing["logging"] = round(time.time() - t0, 1)
        log.info(f"  Frame logs concluídos ({timing['logging']:.0f}s)")

    # Stage 7: FIGURES
    if "figures" in args.stages:
        t0 = time.time()
        log.info("\n[Stage 7/8] FIGURES")
        fg = FigureGenerator(cfg=cfg, exp_dir=exp_dir)
        fg.run()
        stages_completed.append("figures")
        timing["figures"] = round(time.time() - t0, 1)
        log.info(f"  Figuras geradas ({timing['figures']:.0f}s)")

    # Stage 8: EXPORT
    if "export" in args.stages:
        t0 = time.time()
        log.info("\n[Stage 8/8] EXPORT")
        ex = ResultsExporter(cfg=cfg, exp_dir=exp_dir)
        ex.run()
        lx = LatexGenerator(cfg=cfg, exp_dir=exp_dir)
        lx.run()
        stages_completed.append("export")
        timing["export"] = round(time.time() - t0, 1)
        log.info(f"  Export concluído ({timing['export']:.0f}s)")

    # Manifesto final
    write_manifest(exp_dir, exp_id, cfg, args, stages_completed, timing)

    total = sum(timing.values())
    log.info("\n" + "=" * 72)
    log.info(f"  TERA Pipeline concluído  |  {exp_id}")
    log.info(f"  Estágios: {stages_completed}")
    log.info(f"  Tempo total: {total:.0f}s ({total/60:.1f}min)")
    log.info(f"  Resultados: {exp_dir / 'metrics' / 'results_all_seeds.csv'}")
    log.info(f"  Macros:     {exp_dir / 'latex' / 'paper_metrics_macros.tex'}")
    log.info(f"  Manifesto:  {exp_dir / 'manifest.json'}")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
