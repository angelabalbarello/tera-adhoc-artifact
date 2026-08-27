# -*- coding: utf-8 -*-
"""
run_hybrid_completion.py
═══════════════════════════════════════════════════════════════════════════════
Completa a matriz experimental de seeds para as configurações híbridas:

  TABELA ALVO
  ┌─────────────┬────────────────┬─────────────────┬─────────────┬──────────────┐
  │ Seed        │ Baseline Fixed │ Baseline Hybrid │ AF-KD Fixed │ AF-KD Hybrid │
  ├─────────────┼────────────────┼─────────────────┼─────────────┼──────────────┤
  │ 42          │ ✓              │ ✓               │ ✓           │ ✓            │
  │ 43          │ ✓              │ → PREENCHE      │ ✓           │ → PREENCHE   │
  │ 44          │ ✓              │ → PREENCHE      │ ✓           │ → PREENCHE   │
  │ 45          │ ✓              │ → PREENCHE      │ ✓           │ → PREENCHE   │
  └─────────────┴────────────────┴─────────────────┴─────────────┴──────────────┘

MODO DE OPERAÇÃO
  - NÃO retreina modelos.
  - Carrega modelos de modelos_salvos/abl_A/
  - Carrega frame_probs de exp_abl_A_frame_probs_*_seed{N}.npy (se existirem)
  - Para cada seed em TARGET_SEEDS:
      1. Roda select_hybrid_policy no VAL → calibra (tau_d, tau_h)
      2. Avalia AF-KD Híbrida no TEST
      3. Avalia Baseline Híbrida no TEST (com os mesmos tau_d, tau_h)
  - Salva exp_abl_A_hybrid_completion.csv
  - Mescla com exp_abl_A_results_all_seeds.csv (remove duplicatas por Seed+Modelo+Politica)
  - Regenera exp_abl_A_paper_metrics_macros_merged.tex

PRÉ-REQUISITOS
  - run_v29_ablacao_ttdef_ajuste_gatting.py no mesmo diretório
  - modelos_salvos/abl_A/baseline_seed{N}.pt e student_seed{N}.pt para cada seed
  - dados_sinteticos/dataset_sintetico_seed{N}.npz e episodios_seed{N}.csv
  - (Opcional) exp_abl_A_frame_probs_base_seed{N}.npy / _kd_seed{N}.npy

USO
  python run_hybrid_completion.py
  # ou para seeds específicas:
  python run_hybrid_completion.py --seeds 43 44 45
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ─── Importa helpers do run_v29 ──────────────────────────────────────────────
# Estes imports devem funcionar se este script estiver na mesma pasta do run_v29.
try:
    from run_v29_ablacao_ttdef_ajuste_gatting import (
        MultiTaskLSTM,
        DatasetConfig,
        load_or_generate_dataset,
        make_splits,
        split_arrays,
        infer_class_column,
        infer_stream_fixed,
        infer_stream_hybrid,
        select_hybrid_policy,
        select_thr_ep,
        select_theta_ttd,
        summarize_eval,
        aggregate_mean_std,
        export_latex_macros,
        build_sensitivity_table,
        T, D, H, KIN, TTD_M, T_MAX_EF,
        TAU_H_GRID, TAU_DELTA_PERCENTILES,
        USE_STRATIFIED_SPLITS, FORCE_REGEN_SPLITS,
        EXP_NAME,
        MODEL_DIR,
    )
except ImportError as e:
    print(f"❌ Erro ao importar run_v29: {e}")
    print("   Certifique-se de que run_v29_ablacao_ttdef_ajuste_gatting.py está na mesma pasta.")
    sys.exit(1)

# ─── Configuração ─────────────────────────────────────────────────────────────
ALL_SEEDS      = [42, 43, 44, 45]
DEFAULT_SEEDS  = [43, 44, 45]   # seeds a completar (42 já está feita)
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_CSV        = f"{EXP_NAME}_hybrid_completion.csv"
MERGED_CSV     = f"{EXP_NAME}_results_all_seeds_merged.csv"
MERGED_MACROS  = f"{EXP_NAME}_paper_metrics_macros_merged.tex"

# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Completa seeds faltantes das configs híbridas")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help=f"Seeds a processar (default: {DEFAULT_SEEDS})")
    p.add_argument("--force", action="store_true",
                   help="Força reprocessamento mesmo que a seed já exista no CSV")
    p.add_argument("--no-merge", action="store_true",
                   help="Não mescla com o CSV existente; apenas salva completion")
    return p.parse_args()


# ─── Helpers de carregamento ──────────────────────────────────────────────────

def load_model(seed: int, role: str) -> torch.nn.Module:
    """Carrega baseline ou student de modelos_salvos/abl_A/."""
    ckpt = MODEL_DIR / f"{role}_seed{seed}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {ckpt}\n"
            f"Execute o run_v29 completo para a seed {seed} antes de rodar este script."
        )
    model = MultiTaskLSTM(D, H, bi=False).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    print(f"    ✓ {role}_seed{seed}.pt carregado")
    return model


def load_frame_probs_cache(seed: int, role: str) -> np.ndarray | None:
    """Tenta carregar frame_probs do cache .npy; retorna None se não existir."""
    npy = Path(f"{EXP_NAME}_frame_probs_{role}_seed{seed}.npy")
    if npy.exists():
        print(f"    ✓ Cache .npy carregado: {npy.name}")
        return np.load(npy)
    return None


def load_dataset_splits(seed: int):
    """
    Carrega dataset e retorna splits alinhados com o run_v29:
    (X_va, X_te, yep_va, yep_te, yfr_va, yfr_te, prog_va, prog_te)
    """
    cfg = DatasetConfig(seed=seed)
    # load_or_generate_dataset retorna (X, y_fr_cont, y_fr_obs, meta)
    X, y_fr_cont, _y_fr_obs, meta = load_or_generate_dataset(cfg)

    # y_ep: binário (Critico = 1)
    class_col = infer_class_column(meta)
    cat  = meta[class_col].astype(str).values
    y_ep = (cat == "Critico").astype(np.int64)

    # y_fr: binário frame-level, mascarado por episódio
    FRAME_THR = 0.35
    y_fr = ((y_fr_cont > FRAME_THR).astype(np.int64) * y_ep[:, None]).astype(np.int64)

    # progressive flag — igual ao run_v29
    progressive_flag = (
        meta["progressive"].fillna(0).astype(int).values.astype(bool)
        if "progressive" in meta.columns
        else np.zeros(len(meta), dtype=bool)
    )
    progressive_flag = progressive_flag & (y_ep == 1)

    # splits — usa EXATAMENTE os mesmos parâmetros do run_v29
    splits = make_splits(
        y_ep, seed,
        split_path=cfg.split_path,
        reuse=cfg.reuse_splits,
        meta=meta,
        use_stratified=USE_STRATIFIED_SPLITS,
        force_regen=FORCE_REGEN_SPLITS,
    )
    sliced = split_arrays(splits, X, y_ep, y_fr, progressive_flag)
    (X_va,  yep_va,  yfr_va,  prog_va)  = sliced["val"]
    (X_te,  yep_te,  yfr_te,  prog_te)  = sliced["test"]

    print(f"    va={X_va.shape[0]}  te={X_te.shape[0]} | "
          f"críticos te={yep_te.sum()} | "
          f"prog te={prog_te.sum()}")
    return X_va, X_te, yep_va, yep_te, yfr_va, yfr_te, prog_va, prog_te


# ─── Avaliação híbrida para uma seed ─────────────────────────────────────────

def run_hybrid_for_seed(seed: int) -> list[dict]:
    """
    Para uma seed:
      1. Carrega modelos e dados
      2. Calibra (tau_d, tau_h) via select_hybrid_policy no VAL (AF-KD)
      3. Avalia AF-KD Híbrida no TEST
      4. Avalia Baseline Híbrida no TEST com os mesmos limiares

    Retorna lista de dicts prontos para DataFrame.
    """
    print(f"\n{'─'*60}")
    print(f"  SEED {seed}")
    print(f"{'─'*60}")

    # 1. Modelos
    baseline = load_model(seed, "baseline")
    student  = load_model(seed, "student")

    # 2. Dataset + splits
    print("  Carregando dataset e splits...")
    X_va, X_te, yep_va, yep_te, yfr_va, yfr_te, prog_va, prog_te = load_dataset_splits(seed)
    print(f"    VAL: {len(X_va)} ep | TEST: {len(X_te)} ep | críticos: {yep_te.sum()}")

    progressive_te_mask = prog_te.astype(bool)
    abrupt_te_mask = (~progressive_te_mask) & (yep_te == 1)

    # 3. Calibração no VAL (AF-KD)
    print("  Calibrando limiares no VAL...")
    va_kd      = infer_stream_fixed(student, X_va, DEVICE)
    thr_kd     = select_thr_ep(va_kd.frame_probs, yep_va)
    theta_kd   = select_theta_ttd(yfr_va, va_kd.frame_probs, TTD_M)

    ref_kd_va  = summarize_eval(
        va_kd, yep_va, yfr_va, thr_kd, theta_kd, TTD_M,
        progressive_mask=prog_va.astype(bool),
        abrupt_mask=(~prog_va.astype(bool)) & (yep_va == 1),
    )

    hybrid_params = select_hybrid_policy(
        student, X_va, yep_va, yfr_va, DEVICE,
        thr_ep=thr_kd, theta_ttd=theta_kd,
        ref_fixed=ref_kd_va, m_detect=TTD_M,
    )

    if hybrid_params is None:
        print(f"  ⚠️  Seed {seed}: nenhum (τΔ, τH) satisfaz as restrições no VAL → seed ignorada")
        return []

    tau_d, tau_h = hybrid_params
    print(f"  ✓ Limiares calibrados: τΔ={tau_d:.4f}  τH={tau_h:.2f}")

    # 4. Frame_probs TEST — cache ou inferência
    probs_kd_te   = load_frame_probs_cache(seed, "kd")
    probs_base_te = load_frame_probs_cache(seed, "base")

    if probs_kd_te is None:
        print("  Sem cache KD; rodando infer_stream_fixed no TEST...")
        te_kd_fixed = infer_stream_fixed(student, X_te, DEVICE)
        probs_kd_te = te_kd_fixed.frame_probs
        np.save(f"{EXP_NAME}_frame_probs_kd_seed{seed}.npy", probs_kd_te)

    if probs_base_te is None:
        print("  Sem cache Baseline; rodando infer_stream_fixed no TEST...")
        te_base_fixed = infer_stream_fixed(baseline, X_te, DEVICE)
        probs_base_te = te_base_fixed.frame_probs
        np.save(f"{EXP_NAME}_frame_probs_base_seed{seed}.npy", probs_base_te)

    # Calibração do Baseline no VAL (para thr_base)
    va_base  = infer_stream_fixed(baseline, X_va, DEVICE)
    thr_base = select_thr_ep(va_base.frame_probs, yep_va)
    theta_base = theta_kd   # FIX-17b: theta simétrico

    # 5. Inferência híbrida AF-KD no TEST
    print("  Inferindo AF-KD Híbrida no TEST...")
    te_afkd_h = infer_stream_hybrid(student, X_te, DEVICE,
                                     tau_delta=tau_d, tau_h=tau_h)
    row_afkd_h = {
        "Seed": seed, "m": TTD_M,
        "Modelo": "LSTM-AF-KD",
        "Politica": f"Hibrida (tau={tau_d:.4f}, H={tau_h:.2f})",
        **summarize_eval(te_afkd_h, yep_te, yfr_te, thr_kd, theta_kd, TTD_M,
                         progressive_mask=progressive_te_mask,
                         abrupt_mask=abrupt_te_mask),
    }
    print(f"    AF-KD Híbrida  → FR={row_afkd_h.get('FailRate', float('nan')):.4f}"
          f"  Skip={row_afkd_h.get('SkipPct', float('nan')):.1f}%"
          f"  Cost={row_afkd_h.get('Cost_ms_per_frame', float('nan')):.4f} ms/q")

    # 6. Inferência híbrida Baseline no TEST (mesmos limiares tau)
    print("  Inferindo Baseline Híbrida no TEST...")
    te_base_h = infer_stream_hybrid(baseline, X_te, DEVICE,
                                     tau_delta=tau_d, tau_h=tau_h)
    row_base_h = {
        "Seed": seed, "m": TTD_M,
        "Modelo": "LSTM-Baseline",
        "Politica": f"Hibrida (tau={tau_d:.4f}, H={tau_h:.2f})",
        **summarize_eval(te_base_h, yep_te, yfr_te, thr_base, theta_base, TTD_M,
                         progressive_mask=progressive_te_mask,
                         abrupt_mask=abrupt_te_mask),
    }
    print(f"    Baseline Híbrida→ FR={row_base_h.get('FailRate', float('nan')):.4f}"
          f"  Skip={row_base_h.get('SkipPct', float('nan')):.1f}%"
          f"  Cost={row_base_h.get('Cost_ms_per_frame', float('nan')):.4f} ms/q")

    return [row_afkd_h, row_base_h]


# ─── Agregação e exportação de macros ────────────────────────────────────────

def rebuild_macros_from_merged(merged_csv: str, out_tex: str) -> None:
    """
    Reconstrói o arquivo de macros LaTeX a partir do CSV mesclado.
    Usa export_latex_macros do run_v29 com main_summary reconstruído.
    """
    df = pd.read_csv(merged_csv)
    main_df = df[df["m"] == TTD_M].copy()

    # Colapsa variantes da política híbrida
    if "Politica" in main_df.columns:
        main_df["Politica"] = main_df["Politica"].apply(
            lambda s: "Hibrida" if isinstance(s, str) and s.startswith("Hibrida") else s
        )

    num_cols = [c for c in [
        "F1", "ECE", "FailRate", "TTD", "TTDef", "TTD_Adapt",
        "TTD_Progressive", "TTD_Abrupt", "FailRate_Progressive", "FailRate_Abrupt",
        "Lat_ms", "Cost_ms_per_frame", "SkipPct",
    ] if c in main_df.columns]

    main_summary = aggregate_mean_std(main_df, ["Modelo", "Politica"], num_cols)

    # Tabela de sensibilidade — usa apenas linhas m≠TTD_M se existirem
    sens_df = df[df["m"] != TTD_M] if len(df[df["m"] != TTD_M]) > 0 else pd.DataFrame()
    sens_summary = build_sensitivity_table(sens_df) if not sens_df.empty else pd.DataFrame()

    export_latex_macros(main_summary, sens_summary, path=out_tex)
    print(f"\n  ✓ Macros LaTeX (merged) → {out_tex}")

    # Imprime resumo por configuração
    print("\n  === RESUMO MERGED (média entre seeds disponíveis) ===")
    for _, row in main_summary.iterrows():
        modelo  = row.get("Modelo", "?")
        pol     = row.get("Politica", "?")
        n_seeds = len(df[
            (df["Modelo"] == modelo) &
            (df["Politica"].str.startswith(pol if pol != "Hibrida" else "Hibrida"))
            & (df["m"] == TTD_M)
        ])
        fr  = row.get("FailRate_mean", float("nan"))
        f1  = row.get("F1_mean", float("nan"))
        skip = row.get("SkipPct_mean", float("nan"))
        cost = row.get("Cost_ms_per_frame_mean", float("nan"))
        print(f"    {modelo:20s} [{pol:12s}] n={n_seeds:1d} │ "
              f"FR={fr:.3f}  F1={f1:.3f}  Skip={skip:.1f}%  Cost={cost:.3f}ms/q")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    target_seeds = sorted(set(args.seeds))

    print("=" * 60)
    print(f"  run_hybrid_completion — EXP: {EXP_NAME}")
    print(f"  Seeds alvo: {target_seeds}")
    print(f"  Dispositivo: {DEVICE}")
    print("=" * 60)

    # Carrega resultados existentes (se houver)
    existing_csv = Path(f"{EXP_NAME}_results_all_seeds.csv")
    if existing_csv.exists():
        df_existing = pd.read_csv(existing_csv)
        print(f"\n  CSV existente: {existing_csv.name}  ({len(df_existing)} linhas)")
    else:
        df_existing = pd.DataFrame()
        print(f"\n  CSV existente não encontrado — será criado do zero.")

    # Verifica quais seeds já têm híbrida no CSV existente
    if not df_existing.empty and not args.force:
        for seed in list(target_seeds):
            has_hybr = (
                (df_existing.get("Seed", pd.Series(dtype=int)) == seed) &
                (df_existing.get("Politica", pd.Series(dtype=str))
                 .str.startswith("Hibrida", na=False))
            ).any()
            if has_hybr:
                print(f"  ℹ️  Seed {seed} já tem resultados híbridos no CSV."
                      f"  Use --force para reprocessar.")
                target_seeds.remove(seed)

    if not target_seeds:
        print("\n  Nada a processar. Use --force para forçar reprocessamento.")
    else:
        # Processa seeds faltantes
        new_rows = []
        skipped  = []
        for seed in target_seeds:
            rows = run_hybrid_for_seed(seed)
            if rows:
                new_rows.extend(rows)
            else:
                skipped.append(seed)

        if skipped:
            print(f"\n  ⚠️  Seeds ignoradas (sem (τΔ,τH) viável no VAL): {skipped}")

        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df_new.to_csv(OUT_CSV, index=False)
            print(f"\n  ✓ Novos resultados salvos: {OUT_CSV}  ({len(df_new)} linhas)")
        else:
            df_new = pd.DataFrame()
            print("\n  ⚠️  Nenhum novo resultado gerado.")

    # Mesclagem
    if not args.no_merge:
        print(f"\n{'─'*60}")
        print("  Mesclando com CSV existente...")

        parts = [p for p in [df_existing,
                              pd.read_csv(OUT_CSV) if Path(OUT_CSV).exists() else None]
                 if p is not None and not p.empty]

        if parts:
            df_merged = pd.concat(parts, ignore_index=True)

            # Remove duplicatas: mantém a linha mais recente (última) por Seed+Modelo+Politica+m
            df_merged["_pol_norm"] = df_merged["Politica"].apply(
                lambda s: "Hibrida" if isinstance(s, str) and s.startswith("Hibrida") else s
            )
            df_merged = df_merged.drop_duplicates(
                subset=["Seed", "Modelo", "_pol_norm", "m"], keep="last"
            ).drop(columns=["_pol_norm"])

            df_merged.to_csv(MERGED_CSV, index=False)
            print(f"  ✓ Merged CSV: {MERGED_CSV}  ({len(df_merged)} linhas)")

            # Conta seeds por config
            print("\n  === COBERTURA DE SEEDS NO MERGED ===")
            m_df = df_merged[df_merged["m"] == TTD_M].copy()
            m_df["pol_norm"] = m_df["Politica"].apply(
                lambda s: "Hibrida" if isinstance(s, str) and s.startswith("Hibrida") else s
            )
            for (modelo, pol), g in m_df.groupby(["Modelo", "pol_norm"]):
                seeds_found = sorted(g["Seed"].unique().tolist())
                status = "✓" if len(seeds_found) == 4 else f"⚠️  ({len(seeds_found)}/4)"
                print(f"    {modelo:20s} [{pol:8s}]  seeds: {seeds_found}  {status}")

            # Regenera macros
            rebuild_macros_from_merged(MERGED_CSV, MERGED_MACROS)
        else:
            print("  ⚠️  Nenhum dado para mesclar.")

    print(f"\n{'='*60}")
    print("  Arquivos gerados:")
    for f in [OUT_CSV, MERGED_CSV, MERGED_MACROS]:
        p = Path(f)
        if p.exists():
            print(f"    {p.name}  ({p.stat().st_size // 1024 + 1} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
