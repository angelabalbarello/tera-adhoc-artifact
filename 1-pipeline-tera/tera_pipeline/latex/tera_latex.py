# -*- coding: utf-8 -*-
"""
tera_pipeline/latex/tera_latex.py
═══════════════════════════════════════════════════════════════════════════════
Geração automática de macros LaTeX e tabelas a partir do CSV canônico.

REGRA DE ORO:
  paper_metrics_macros.tex é SOMENTE LEITURA para humanos.
  Todo valor numérico no paper deve ser referenciado via \macro.
  Nenhum número é editado manualmente.

FONTES:
  results_all_seeds.csv → agrega → macros + tabelas

SAÍDAS:
  paper_metrics_macros.tex        → \newcommand{\resXXX}{0.000}
  paper_table2_rows.tex           → linhas da Tabela 2 (fatorial 2×2)
  paper_table_stratified_rows.tex → linhas da Tabela estratificada
  paper_table_seeds_coverage.tex  → tabela de cobertura de seeds
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Mapeamento canônico: (prefixo_macro, config_id) ─────────────────────────
MACRO_MAPPING = [
    ("BaseFixo",  "baseline_fixed"),
    ("BaseHibr",  "baseline_hybrid"),
    ("AlvoFixo",  "aftkd_fixed"),
    ("AlvoHibr",  "aftkd_hybrid"),
]

# Métricas e seus sufixos de macro
METRIC_CMDS = [
    ("F1",                "FUm"),
    ("ECE",               "ECE"),
    ("FailRate",          "Fail"),
    ("TTD",               "TTD"),
    ("TTDef",             "TTDef"),
    ("Lat_ms",            "Lat"),
    ("Cost_ms_per_frame", "Cost"),
    ("SkipPct",           "Skip"),
    ("Precision",         "Prec"),
    ("Recall",            "Rec"),
]

# Métricas estratificadas
STRAT_METRIC_CMDS = [
    ("FailRate_Progressive", "ProgFail"),
    ("TTD_Progressive",      "ProgTTD"),
    ("TTDef_Progressive",    "ProgTTDef"),
    ("FailRate_Abrupt",      "AbruptFail"),
    ("TTD_Abrupt",           "AbruptTTD"),
    ("TTDef_Abrupt",         "AbruptTTDef"),
]

# Macros derivadas (calculadas a partir dos resultados)
DERIVED_MACROS = {
    "CostReduction": lambda s: _cost_reduction(s),
    "SkipRate":      lambda s: _skip_rate(s),
    "ECEReduction":  lambda s: _ece_reduction(s),
}


def _fmt(v, decimals: int = 3) -> str:
    """Formata número para macro LaTeX. NaN → '--'."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{v:.{decimals}f}"


def _latex_cmd(name: str, value: str) -> str:
    """Gera linha \\newcommand{\\name}{value}."""
    return "\\newcommand{\\" + name + "}{" + value + "}"


def _cost_reduction(summary: pd.DataFrame) -> float:
    """Redução de custo: (Cost_BaseFixo - Cost_AlvoHibr) / Cost_BaseFixo × 100."""
    row_base = summary[summary["config_id"] == "baseline_fixed"]
    row_afkd = summary[summary["config_id"] == "aftkd_hybrid"]
    if row_base.empty or row_afkd.empty:
        return float("nan")
    cost_base = float(row_base["Cost_ms_per_frame_mean"].iloc[0])
    cost_afkd = float(row_afkd["Cost_ms_per_frame_mean"].iloc[0])
    return round((cost_base - cost_afkd) / cost_base * 100, 1)


def _skip_rate(summary: pd.DataFrame) -> float:
    row = summary[summary["config_id"] == "aftkd_hybrid"]
    if row.empty:
        return float("nan")
    return round(float(row["SkipPct_mean"].iloc[0]), 1)


def _ece_reduction(summary: pd.DataFrame) -> float:
    row_base = summary[summary["config_id"] == "baseline_fixed"]
    row_afkd = summary[summary["config_id"] == "aftkd_fixed"]
    if row_base.empty or row_afkd.empty:
        return float("nan")
    ece_base = float(row_base["ECE_mean"].iloc[0])
    ece_afkd = float(row_afkd["ECE_mean"].iloc[0])
    return round((ece_base - ece_afkd) / ece_base * 100, 1)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega results_all_seeds.csv por config_id."""
    num_cols = [c for c in df.columns
                if c not in ("Seed", "config_id", "Modelo", "Gating",
                              "N_Progressive", "N_Abrupt")]
    agg = df.groupby("config_id")[num_cols].agg(["mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    n_seeds = df.groupby("config_id")["Seed"].nunique().rename("N_Seeds")
    seeds_list = df.groupby("config_id")["Seed"].apply(
        lambda x: sorted(x.unique().tolist())
    ).rename("Seeds_List")
    return agg.join(n_seeds).join(seeds_list).reset_index()


class LatexGenerator:
    """
    Gerador de assets LaTeX a partir do CSV canônico de resultados.
    Nunca editar os arquivos gerados manualmente.
    """

    def __init__(self, cfg: dict, exp_dir: Path):
        self.cfg      = cfg
        self.exp_dir  = exp_dir
        self.latex_cfg = cfg.get("latex", {})
        self.out_dir  = exp_dir / self.latex_cfg.get("output_dir", "latex")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Executa geração completa de assets LaTeX."""
        results_path = self.exp_dir / "metrics" / "results_all_seeds.csv"
        if not results_path.exists():
            raise FileNotFoundError(f"results_all_seeds.csv não encontrado: {results_path}")

        df      = pd.read_csv(results_path)
        summary = build_summary(df)

        self._validate_seed_coverage(summary)
        self.generate_macros(summary)
        self.generate_table2(summary)
        self.generate_table_stratified(df)
        self.generate_seed_coverage_table(summary)
        self._write_readme(summary)

    def _validate_seed_coverage(self, summary: pd.DataFrame) -> None:
        """Avisa se alguma config não tem 4 seeds."""
        for _, row in summary.iterrows():
            n = int(row.get("N_Seeds", 0))
            if n < 4:
                print(f"  ⚠️  {row['config_id']}: apenas {n}/4 seeds no CSV.")
                print("      Macros geradas são menos robustas estatisticamente.")

    def generate_macros(self, summary: pd.DataFrame) -> None:
        """
        Gera paper_metrics_macros.tex a partir do summary CSV.
        REGRA: este arquivo é gerado pelo pipeline, nunca editado manualmente.
        """
        lines = [
            "% ═══════════════════════════════════════════════════════════════",
            "% paper_metrics_macros.tex — AUTO-GERADO PELO TERA PIPELINE v1.0",
            "% NÃO EDITAR MANUALMENTE.",
            "% Fonte: results_all_seeds.csv",
            f"% Gerado por: tera_latex.py",
            "% ═══════════════════════════════════════════════════════════════",
            "",
        ]

        # ── Família 1: métricas principais por config ─────────────────────
        lines.append("% ── Tabela 2: resultados principais (fatorial 2×2) ──────────────")
        for prefix, config_id in MACRO_MAPPING:
            row = summary[summary["config_id"] == config_id]
            if row.empty:
                lines.append(f"% ⚠️  config '{config_id}' não encontrada no CSV")
                for _, cmd in METRIC_CMDS:
                    lines.append(_latex_cmd(f"res{prefix}{cmd}", "--"))
                    lines.append(_latex_cmd(f"std{prefix}{cmd}", "--"))
                continue

            r = row.iloc[0]
            n = int(r.get("N_Seeds", 0))
            lines.append(f"% {config_id}  (n={n} seeds: {r.get('Seeds_List', [])})")
            for metric, cmd in METRIC_CMDS:
                mn_col = f"{metric}_mean"
                sd_col = f"{metric}_std"
                if mn_col in r.index:
                    lines.append(_latex_cmd(f"res{prefix}{cmd}", _fmt(r[mn_col])))
                if sd_col in r.index:
                    lines.append(_latex_cmd(f"std{prefix}{cmd}", _fmt(r[sd_col])))
            lines.append("")

        # ── Família 2: métricas estratificadas ───────────────────────────
        lines.append("% ── Análise estratificada (progressivo vs abrupto) ─────────────")
        for prefix, config_id in MACRO_MAPPING:
            row = summary[summary["config_id"] == config_id]
            if row.empty:
                continue
            r = row.iloc[0]
            for metric, cmd in STRAT_METRIC_CMDS:
                mn_col = f"{metric}_mean"
                sd_col = f"{metric}_std"
                if mn_col in r.index:
                    lines.append(_latex_cmd(f"res{prefix}{cmd}", _fmt(r[mn_col])))
                if sd_col in r.index:
                    lines.append(_latex_cmd(f"std{prefix}{cmd}", _fmt(r[sd_col])))
        lines.append("")

        # ── Família 3: macros derivadas ───────────────────────────────────
        lines.append("% ── Macros derivadas ────────────────────────────────────────────")
        for macro_name, fn in DERIVED_MACROS.items():
            try:
                val = fn(summary)
                lines.append(_latex_cmd(macro_name, _fmt(val, decimals=1)))
            except Exception:
                lines.append(_latex_cmd(macro_name, "--"))
        lines.append("")

        # ── Cobertura de seeds ────────────────────────────────────────────
        lines.append("% ── Cobertura de seeds por configuração ─────────────────────────")
        for _, config_id in MACRO_MAPPING:
            row = summary[summary["config_id"] == config_id]
            n = int(row.iloc[0]["N_Seeds"]) if not row.empty else 0
            lines.append(f"% {config_id}: {n}/4 seeds")
        lines.append("")

        out = self.out_dir / self.latex_cfg.get("macros_file", "paper_metrics_macros.tex")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  ✓ Macros LaTeX → {out}")

    def generate_table2(self, summary: pd.DataFrame) -> None:
        """
        Gera linhas LaTeX da Tabela 2 (fatorial 2×2).
        Formato: Modelo & Gating & F1 & ECE & FR & TTD∂ & TTDb & TTDef & ms/q & %Sup \\
        """
        rows = []
        config_display = {
            "baseline_fixed":  ("LSTM-Baseline", "Não"),
            "baseline_hybrid": ("LSTM-Baseline", "Sim"),
            "aftkd_fixed":     ("LSTM-AF-TKD",   "Não"),
            "aftkd_hybrid":    ("LSTM-AF-TKD",   r"\textbf{Sim}"),
        }
        is_best = {
            "aftkd_hybrid": True,
        }

        for _, config_id in MACRO_MAPPING:
            row = summary[summary["config_id"] == config_id]
            if row.empty:
                continue
            r = row.iloc[0]
            modelo, gating = config_display.get(config_id, (config_id, "?"))
            bold = config_id in is_best

            def v(col):
                mn = r.get(f"{col}_mean", float("nan"))
                sd = r.get(f"{col}_std", float("nan"))
                val = f"${_fmt(mn)}\\pm{_fmt(sd)}$"
                return f"$\\mathbf{{{_fmt(mn)}\\pm{_fmt(sd)}}}$" if bold else val

            # TTD∂ só para as linhas sem gating
            if not config_display[config_id][1].startswith("Sim"):
                ttd_partial = v("TTD_Progressive")
            else:
                ttd_partial = v("TTD_Progressive")

            # TTDb (bruto, apenas detectados) — para Baseline Hybrid omite
            if config_id == "baseline_hybrid":
                ttd_b = "{---}"
            else:
                ttd_b = v("TTD")

            line = (
                f"{modelo} & {gating}"
                f" & {v('F1')} & {v('ECE')} & {v('FailRate')}"
                f" & {ttd_partial} & {ttd_b} & {v('TTDef')}"
                f" & {v('Cost_ms_per_frame')} & {v('SkipPct')} \\\\"
            )
            rows.append(line)

        out = self.out_dir / self.latex_cfg.get("table2_file", "paper_table2_rows.tex")
        out.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"  ✓ Tabela 2 LaTeX → {out}")

    def generate_table_stratified(self, df: pd.DataFrame) -> None:
        """Gera linhas da tabela estratificada (progressivo vs abrupto)."""
        summary = build_summary(df)
        rows = []
        config_display = {
            "baseline_fixed":  "LSTM-Baseline",
            "baseline_hybrid": "Baseline+Gating",
            "aftkd_fixed":     "AF-TKD Fixo",
            "aftkd_hybrid":    "AF-TKD+Gating",
        }
        for _, config_id in MACRO_MAPPING:
            row = summary[summary["config_id"] == config_id]
            if row.empty:
                continue
            r = row.iloc[0]
            name = config_display.get(config_id, config_id)

            def v(col):
                mn = r.get(f"{col}_mean", float("nan"))
                sd = r.get(f"{col}_std", float("nan"))
                return f"${_fmt(mn)}\\pm{_fmt(sd)}$"

            line = (
                f"{name}"
                f" & {v('FailRate_Progressive')} & {v('TTD_Progressive')}"
                f" & {v('FailRate_Abrupt')} & {v('TTD_Abrupt')}"
                f" & {v('FailRate')} & {v('TTDef')} \\\\"
            )
            rows.append(line)

        out = self.out_dir / self.latex_cfg.get(
            "table_stratified_file", "paper_table_stratified_rows.tex")
        out.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"  ✓ Tabela estratificada LaTeX → {out}")

    def generate_seed_coverage_table(self, summary: pd.DataFrame) -> None:
        """
        Gera tabela de cobertura de seeds.
        Exigência metodológica para FGCS: transparência sobre N_seeds por config.
        """
        all_seeds = [42, 43, 44, 45]
        config_names = {
            "baseline_fixed":  "Baseline Fixed",
            "baseline_hybrid": "Baseline Hybrid",
            "aftkd_fixed":     "AF-TKD Fixed",
            "aftkd_hybrid":    "AF-TKD Hybrid",
        }
        lines = [
            r"\begin{table}[!t]",
            r"\centering",
            r"\caption{Cobertura de sementes por configuração experimental.}",
            r"\label{tab:seed_coverage}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{5pt}",
            r"\renewcommand{\arraystretch}{1.15}",
            r"\begin{tabular}{l c c c c c}",
            r"\toprule",
            r"\textbf{Configuração} & \textbf{Seed 42} & \textbf{Seed 43} "
            r"& \textbf{Seed 44} & \textbf{Seed 45} & \textbf{Total} \\",
            r"\midrule",
        ]
        for _, config_id in MACRO_MAPPING:
            row_s = summary[summary["config_id"] == config_id]
            seeds_ran = set(row_s.iloc[0].get("Seeds_List", [])
                            if not row_s.empty else [])
            name = config_names.get(config_id, config_id)
            cells = []
            for s in all_seeds:
                cells.append(r"\ding{51}" if s in seeds_ran else r"\ding{55}")
            total = sum(1 for s in all_seeds if s in seeds_ran)
            line = f"{name} & " + " & ".join(cells) + f" & {total}/4 \\\\"
            lines.append(line)

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        out = self.out_dir / "paper_table_seed_coverage.tex"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  ✓ Tabela cobertura seeds → {out}")

    def _write_readme(self, summary: pd.DataFrame) -> None:
        """Escreve README no diretório latex/ explicando os arquivos."""
        lines = [
            "# latex/ — Assets LaTeX gerados automaticamente pelo TERA Pipeline",
            "",
            "## REGRA DE ORO",
            "Nenhum arquivo neste diretório deve ser editado manualmente.",
            "Para atualizar valores, re-execute o pipeline:",
            "  python run_experiment.py --stages export",
            "",
            "## Arquivos",
            "- `paper_metrics_macros.tex` → todas as macros \\resXXX e \\stdXXX",
            "- `paper_table2_rows.tex`    → linhas da Tabela 2 do paper",
            "- `paper_table_stratified_rows.tex` → linhas da tabela estratificada",
            "- `paper_table_seed_coverage.tex`   → tabela de cobertura de seeds",
            "",
            "## Como usar no paper",
            "```latex",
            "% No preâmbulo ou antes de \\begin{document}:",
            "\\input{../results/latest/latex/paper_metrics_macros.tex}",
            "```",
            "",
            "## Cobertura de seeds",
        ]
        for _, config_id in MACRO_MAPPING:
            row = summary[summary["config_id"] == config_id]
            n = int(row.iloc[0]["N_Seeds"]) if not row.empty else 0
            seeds = row.iloc[0].get("Seeds_List", []) if not row.empty else []
            lines.append(f"- `{config_id}`: {n}/4 seeds {seeds}")

        out = self.out_dir / "README.md"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
