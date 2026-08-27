# -*- coding: utf-8 -*-
"""
tera_pipeline/export/tera_export.py
═══════════════════════════════════════════════════════════════════════════════
Exportação de resultados do TERA Pipeline.

Responsabilidades:
  · Agregar results_all_seeds.csv → summary_by_config.csv
  · Verificar cobertura completa de seeds (invariante: 4 seeds × 4 configs)
  · Escrever calibration_summary.csv consolidado
  · Calcular métricas derivadas (CostReduction, SkipRate, melhoria TTDef)
  · Escrever relatório de integridade (integrity_report.txt)

SAÍDAS em exp_dir/metrics/:
  summary_by_config.csv   — média ± std por config_id
  integrity_report.txt    — verificação de cobertura e consistência
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


REQUIRED_SEEDS   = [42, 43, 44, 45]
REQUIRED_CONFIGS = ["baseline_fixed", "baseline_hybrid",
                    "aftkd_fixed", "aftkd_hybrid"]

NUM_METRICS = [
    "F1", "Precision", "Recall", "ECE",
    "FailRate", "TTD", "TTDef",
    "Lat_ms", "Cost_ms_per_frame", "SkipPct",
    "FailRate_Progressive", "FailRate_Abrupt",
    "TTD_Progressive", "TTD_Abrupt",
    "TTDef_Progressive", "TTDef_Abrupt",
]


class ResultsExporter:
    """Agrega e exporta todos os resultados do pipeline."""

    def __init__(self, cfg: dict, exp_dir: Path):
        self.cfg     = cfg
        self.exp_dir = exp_dir
        self.out_dir = exp_dir / "metrics"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Executa exportação completa."""
        results_path = self.out_dir / "results_all_seeds.csv"
        if not results_path.exists():
            print(f"  ⚠️  results_all_seeds.csv não encontrado em {self.out_dir}")
            return

        df = pd.read_csv(results_path)

        # Normaliza nome da coluna de seed
        if "Seed" not in df.columns and "seed" in df.columns:
            df = df.rename(columns={"seed": "Seed"})

        self._export_summary(df)
        self._check_integrity(df)
        self._export_derived_metrics(df)
        print(f"  ✓ Export concluído → {self.out_dir}")

    # ── Summary ───────────────────────────────────────────────────────────────

    def _export_summary(self, df: pd.DataFrame) -> None:
        """Agrega por config_id: média, std, n_seeds."""
        cols = [c for c in NUM_METRICS if c in df.columns]
        agg  = df.groupby("config_id")[cols].agg(["mean", "std"]).round(4)
        agg.columns = ["_".join(c) for c in agg.columns]

        n_seeds   = df.groupby("config_id")["Seed"].nunique().rename("N_Seeds")
        seeds_col = df.groupby("config_id")["Seed"].apply(
            lambda x: str(sorted(x.unique().tolist()))
        ).rename("Seeds")

        summary = agg.join(n_seeds).join(seeds_col).reset_index()
        out = self.out_dir / "summary_by_config.csv"
        summary.to_csv(out, index=False)
        print(f"  ✓ summary_by_config.csv ({len(summary)} configs)")

    # ── Verificação de integridade ────────────────────────────────────────────

    def _check_integrity(self, df: pd.DataFrame) -> None:
        """Verifica cobertura completa e escreve relatório."""
        lines = ["TERA Pipeline — Relatório de Integridade\n", "=" * 50 + "\n"]
        all_ok = True

        # Cobertura de seeds
        lines.append("\n[Cobertura de Seeds]\n")
        for config_id in REQUIRED_CONFIGS:
            sub = df[df["config_id"] == config_id]
            seeds_found = sorted(sub["Seed"].unique().tolist())
            missing = [s for s in REQUIRED_SEEDS if s not in seeds_found]
            status = "✓ COMPLETO" if not missing else f"✗ FALTANDO {missing}"
            if missing:
                all_ok = False
            lines.append(f"  {config_id:25s}: {seeds_found}  {status}\n")

        # Configs ausentes
        lines.append("\n[Configs Ausentes]\n")
        found_configs = df["config_id"].unique().tolist()
        for c in REQUIRED_CONFIGS:
            if c not in found_configs:
                lines.append(f"  ✗ {c} não encontrado no CSV\n")
                all_ok = False

        # Métricas críticas
        lines.append("\n[Métricas Críticas]\n")
        summary_path = self.out_dir / "summary_by_config.csv"
        if summary_path.exists():
            summ = pd.read_csv(summary_path)
            for _, row in summ.iterrows():
                config = row["config_id"]
                fr  = row.get("FailRate_mean", float("nan"))
                n   = int(row.get("N_Seeds", 0))
                fr_ok = "✓" if (not math.isnan(fr) and fr < 0.5) else "⚠"
                lines.append(
                    f"  {config:25s}: FR={fr:.3f} {fr_ok}  n_seeds={n}\n"
                )

        lines.append(f"\n{'=' * 50}\n")
        lines.append(f"RESULTADO GERAL: {'✓ PIPELINE COMPLETO' if all_ok else '✗ INCOMPLETO — verificar acima'}\n")

        out = self.out_dir / "integrity_report.txt"
        out.write_text("".join(lines), encoding="utf-8")
        print(f"  ✓ integrity_report.txt → {'OK' if all_ok else 'INCOMPLETO'}")

        if not all_ok:
            print("  ⚠️  Pipeline incompleto! Veja integrity_report.txt")

    # ── Métricas derivadas ────────────────────────────────────────────────────

    def _export_derived_metrics(self, df: pd.DataFrame) -> None:
        """Calcula e salva métricas derivadas usadas no artigo."""
        summary_path = self.out_dir / "summary_by_config.csv"
        if not summary_path.exists():
            return

        summ = pd.read_csv(summary_path)

        def get(config_id: str, col: str) -> float:
            row = summ[summ["config_id"] == config_id]
            return float(row[col].iloc[0]) if (not row.empty and col in row.columns) else float("nan")

        cost_base = get("baseline_fixed",  "Cost_ms_per_frame_mean")
        cost_afkd = get("aftkd_hybrid",    "Cost_ms_per_frame_mean")
        skip_afkd = get("aftkd_hybrid",    "SkipPct_mean")
        ttdef_base = get("baseline_fixed", "TTDef_mean")
        ttdef_afkd = get("aftkd_hybrid",   "TTDef_mean")
        ece_base   = get("baseline_fixed", "ECE_mean")
        ece_fixed  = get("aftkd_fixed",    "ECE_mean")

        cost_reduction = (
            (cost_base - cost_afkd) / cost_base * 100
            if not math.isnan(cost_base) and cost_base > 0
            else float("nan")
        )
        ttdef_improvement = (
            (ttdef_base - ttdef_afkd) / ttdef_base * 100
            if not math.isnan(ttdef_base) and ttdef_base > 0
            else float("nan")
        )
        ece_reduction = (
            (ece_base - ece_fixed) / ece_base * 100
            if not math.isnan(ece_base) and ece_base > 0
            else float("nan")
        )

        derived = {
            "CostReduction_pct":     round(cost_reduction, 1),
            "SkipRate_pct":          round(skip_afkd, 1),
            "TTDef_improvement_pct": round(ttdef_improvement, 1),
            "ECE_reduction_pct":     round(ece_reduction, 1),
            "Cost_baseline_ms":      round(cost_base, 4),
            "Cost_aftkd_hybrid_ms":  round(cost_afkd, 4),
            "TTDef_baseline_s":      round(ttdef_base, 4),
            "TTDef_aftkd_hybrid_s":  round(ttdef_afkd, 4),
        }

        out = self.out_dir / "derived_metrics.json"
        out.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
        print(f"  ✓ derived_metrics.json")
        print(f"      CostReduction={cost_reduction:.1f}%  "
              f"SkipRate={skip_afkd:.1f}%  "
              f"TTDef improvement={ttdef_improvement:.1f}%")
