# -*- coding: utf-8 -*-
"""
generate_episode_frame_logs.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gera episode_frame_logs.csv — usado pelas figuras:
  · fig:temporal_trace   p(t), H(t), regiões de supressão MHEG, t0
  · fig:entropy_dist     distribuição de H(p) por fase temporal
  · fig:gating_heatmap   mapa de ativação MHEG (episódio × frame)

Uma linha por (seed, config, episode_id, frame):

  seed | config | episode_id | regime | recipe | is_critical
  | t0_frame | t0_time_s | t | time_s
  | p_t | H_t | is_suppressed
  | y_frame_clean | phase

Configs produzidas (4 colunas no trade-off):
  baseline_fixed  — LSTM-Baseline, política fixa (sem gating)
  afkd_fixed      — LSTM-AF-KD,    política fixa (sem gating)
  baseline_hybrid — LSTM-Baseline, política híbrida (com gating)
  afkd_hybrid     — LSTM-AF-KD,    política híbrida (com gating)

"phase" (para fig:entropy_dist):
  stable     — t < t0 − PRE_ONSET_WINDOW
  pre_onset  — t0 − PRE_ONSET_WINDOW ≤ t < t0
  post_onset — t ≥ t0
  (não-críticos têm phase = "normal" e t0_frame = -1)

SEEDS: por padrão apenas [42] (suficiente para todas as figuras de trace
e heatmap). Altere SEEDS_TO_RUN para incluir mais sementes se quiser a
distribuição de entropia completa (mais representativa).

PRÉ-REQUISITOS (no mesmo diretório):
  · run_v29_ablacao_ttdef_ajuste_gatting.py   (versão patched)
  · synthetic_driver_risk_v7.py
  · dados_sinteticos/dataset_sintetico_seed{N}.npz + episodios_seed{N}.csv
  · modelos_salvos/abl_A/baseline_seed{N}.pt + student_seed{N}.pt
  · (acelera) exp_abl_A_frame_probs_base_seed{N}.npy
  · (acelera) exp_abl_A_frame_probs_kd_seed{N}.npy
  · (tau params) exp_abl_A_results_all_seeds.csv

SAÍDA:
  · episode_frame_logs.csv  (~40 MB para 4 configs × seed 42)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import importlib
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Importar run_v29
# O arquivo deve estar no mesmo diretório com o nome exato abaixo.
RUN_V29_MODULE = "run_v29_ablacao_ttdef_ajuste_gatting"
try:
    rv = importlib.import_module(RUN_V29_MODULE)
except ModuleNotFoundError:
    sys.exit(
        f"ERRO: '{RUN_V29_MODULE}.py' não encontrado no diretório atual.\n"
        "Coloque este script na mesma pasta do run_v29."
    )

# Re-exportar símbolos necessários
MultiTaskLSTM            = rv.MultiTaskLSTM
DatasetConfig            = rv.DatasetConfig
load_or_generate_dataset = rv.load_or_generate_dataset
make_splits              = rv.make_splits
split_arrays             = rv.split_arrays
infer_stream_fixed       = rv.infer_stream_fixed
make_prefix_window       = rv.make_prefix_window
binary_entropy           = rv.binary_entropy
select_hybrid_policy     = rv.select_hybrid_policy
select_thr_ep            = rv.select_thr_ep
select_theta_ttd         = rv.select_theta_ttd
summarize_eval           = rv.summarize_eval
episode_probs_from_frames= rv.episode_probs_from_frames
StreamEval               = rv.StreamEval
sync_cuda                = rv.sync_cuda
warmup_model             = rv.warmup_model
infer_stream_hybrid      = rv.infer_stream_hybrid
KIN                      = rv.KIN
TTD_M                    = rv.TTD_M
T_FRAMES                 = rv.T          # 96 frames por episódio
D                        = rv.D
H_DIM                    = rv.H
K_AGG                    = rv.K_AGG
EXP_NAME                 = rv.EXP_NAME  # "exp_abl_A"
MODEL_DIR                = rv.MODEL_DIR
USE_STRATIFIED_SPLITS    = rv.USE_STRATIFIED_SPLITS
FORCE_REGEN_SPLITS       = rv.FORCE_REGEN_SPLITS
infer_class_column       = rv.infer_class_column

# CONFIGURAÇÃO — ajuste aqui se necessário
SEEDS_TO_RUN     = [42]        # adicione mais sementes se quiser entrop. dist. completa
FRAME_THR        = 0.35        # limiar de onset (igual ao run_v29)
PRE_ONSET_WINDOW = 10          # frames antes do onset = "pre_onset" (~1 s)
OUTPUT_CSV       = "episode_frame_logs.csv"

# HELPER — inferência híbrida com máscara de gating por frame

@torch.no_grad()
def infer_hybrid_with_mask(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    tau_delta: float,
    tau_h: float,
    window: int = T_FRAMES,
) -> tuple:
    """
    Igual a infer_stream_hybrid, mas retorna também a máscara de gating.

    Retorna:
        frame_probs  — (n, T) float32  probabilidades por frame
        gating_mask  — (n, T) int8     1=frame processado, 0=suprimido
        lat_ms       — float           latência média por frame ativo (ms)
        skip_pct     — float           % de frames suprimidos
        cost_ms      — float           custo amortizado por frame (ms)
    """
    model.eval()
    n, t_len, _ = X.shape
    frame_probs  = np.zeros((n, t_len), dtype=np.float32)
    gating_mask  = np.zeros((n, t_len), dtype=np.int8)
    lat_list     = []
    skipped      = 0
    updates      = 0

    x_ref = make_prefix_window(X[0], min(5, t_len - 1), window)
    warmup_model(model, x_ref, device)

    for i in range(n):
        last_p = None

        for t in range(t_len):
            # Frame 0 -> sempre atualiza
            if t == 0 or last_p is None:
                x_slice = make_prefix_window(X[i], t, window)
                xt = torch.tensor(x_slice, dtype=torch.float32,
                                   device=device).unsqueeze(0)
                sync_cuda()
                ts = time.perf_counter()
                fr_log, _ = model(xt)
                sync_cuda()
                te = time.perf_counter()
                last_p = float(torch.sigmoid(fr_log[0, -1, 0]).item())
                frame_probs[i, t]  = last_p
                gating_mask[i, t]  = 1
                lat_list.append((te - ts) * 1000.0)
                updates += 1
                continue

            # Decisão de gating
            try:
                dk = float(np.abs(X[i, t, KIN] - X[i, t - 1, KIN]).max())
            except Exception:
                dk = 0.0
            p_clip = float(np.clip(last_p, 1e-6, 1 - 1e-6))
            # FIX 2026-09-23 (auditoria): entropia do gate em BITS (log2), como
            # na Eq. 11 do artigo e no run_v29 que calibrou tau_h=0.95; a versao
            # anterior usava np.log (nats, max 0.693 < 0.95), de modo que o ramo
            # entropico nunca disparava e a mascara ficava puramente cinematica.
            H_val  = -(p_clip * np.log2(p_clip) + (1 - p_clip) * np.log2(1 - p_clip))
            update = (dk >= tau_delta) or (H_val >= tau_h)

            if not update:
                skipped += 1
                frame_probs[i, t] = last_p
                gating_mask[i, t] = 0
                continue

            x_slice = make_prefix_window(X[i], t, window)
            xt = torch.tensor(x_slice, dtype=torch.float32,
                               device=device).unsqueeze(0)
            sync_cuda()
            ts = time.perf_counter()
            fr_log, _ = model(xt)
            sync_cuda()
            te = time.perf_counter()
            last_p = float(torch.sigmoid(fr_log[0, -1, 0]).item())
            frame_probs[i, t] = last_p
            gating_mask[i, t] = 1
            lat_list.append((te - ts) * 1000.0)
            updates += 1

    avg_lat   = float(np.mean(lat_list)) if lat_list else 0.0
    skip_pct  = 100.0 * skipped / float(n * t_len)
    total_ms  = sum(lat_list)
    cost_ms   = total_ms / float(n * t_len) if n * t_len > 0 else 0.0
    return frame_probs, gating_mask, avg_lat, skip_pct, cost_ms


# HELPER — carregar tau_delta e tau_h do CSV de resultados

def load_tau_params(seed: int, exp_name: str) -> tuple:
    """
    Tenta ler tau_delta e tau_h da coluna Politica do CSV de resultados.
    Formato esperado: "Hibrida (tau=0.0190, H=0.95)"
    Retorna (tau_delta, tau_h) ou (None, None) se não encontrar.
    """
    csv_path = Path(f"{exp_name}_results_all_seeds.csv")
    if not csv_path.exists():
        return None, None
    try:
        df = pd.read_csv(csv_path)
        hibr = df[
            (df["Seed"].astype(int) == seed) &
            (df["Politica"].astype(str).str.contains("Hibrida", na=False))
        ]
        if hibr.empty:
            return None, None
        pol = str(hibr.iloc[0]["Politica"])
        m_d = re.search(r"tau=([\d.]+)", pol)
        m_h = re.search(r"H=([\d.]+)", pol)
        if m_d and m_h:
            return float(m_d.group(1)), float(m_h.group(1))
    except Exception as e:
        print(f"  [warn] Não foi possível ler tau params do CSV: {e}")
    return None, None


# HELPER — derivar fase temporal

def compute_phase(t: int, t0: int, pre_window: int) -> str:
    """Classifica o frame t em relação ao onset t0."""
    if t0 < 0:
        return "normal"          # episódio não-crítico
    if t >= t0:
        return "post_onset"
    if t >= t0 - pre_window:
        return "pre_onset"
    return "stable"


# LOOP PRINCIPAL

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"EXP_NAME: {EXP_NAME}")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"SEEDS: {SEEDS_TO_RUN}")

    all_rows: list = []
    dt = 10.0 / T_FRAMES  # segundos por frame

    for SEED in SEEDS_TO_RUN:
        print(f"\n{'='*60}")
        print(f"  SEED {SEED}")
        print(f"{'='*60}")

        # 1. Dataset + splits
        ds_cfg = DatasetConfig(seed=SEED)
        X, y_fr_cont, y_fr_obs, meta = load_or_generate_dataset(ds_cfg)

        class_col = infer_class_column(meta)
        cat       = meta[class_col].astype(str).values
        y_ep      = (cat == "Critico").astype(np.int64)

        prog_flag = (
            meta["progressive"].fillna(0).astype(int).values.astype(bool)
            if "progressive" in meta.columns
            else np.zeros(len(meta), dtype=bool)
        )
        prog_flag = prog_flag & (y_ep == 1)

        # y_fr binarizado (para métricas), y_fr_cont para ground truth limpo
        y_fr = ((y_fr_cont > FRAME_THR).astype(np.int64) * y_ep[:, None]).astype(np.int64)

        splits = make_splits(
            y_ep, SEED,
            split_path=ds_cfg.split_path,
            reuse=ds_cfg.reuse_splits,
            meta=meta,
            use_stratified=USE_STRATIFIED_SPLITS,
            force_regen=FORCE_REGEN_SPLITS,
        )
        sliced = split_arrays(splits, X, y_ep, y_fr, y_fr_cont, prog_flag)
        X_va, yep_va, yfr_va, _, prog_va = sliced["val"]
        X_te, yep_te, yfr_te, yfrc_te, prog_te = sliced["test"]

        meta_te = meta.iloc[splits["test_idx"]].reset_index(drop=True)
        recipe_ids = (
            meta_te["recipe_id"].astype(str).values
            if "recipe_id" in meta_te.columns
            else np.array(["unknown"] * len(X_te))
        )

        print(f"  Test set: {X_te.shape[0]} episódios | "
              f"{yep_te.sum()} críticos | "
              f"{prog_te.astype(int).sum()} progressivos")

        # 2. Carregar modelos
        ckpt_base = MODEL_DIR / f"baseline_seed{SEED}.pt"
        ckpt_stud = MODEL_DIR / f"student_seed{SEED}.pt"
        if not ckpt_base.exists() or not ckpt_stud.exists():
            print(f"  [ERRO] Checkpoints não encontrados em {MODEL_DIR}.")
            print("  Execute run_v29 primeiro para treinar os modelos.")
            continue

        baseline = MultiTaskLSTM(D, H_DIM, bi=False).to(device)
        baseline.load_state_dict(torch.load(ckpt_base, map_location=device))
        baseline.eval()

        student = MultiTaskLSTM(D, H_DIM, bi=False).to(device)
        student.load_state_dict(torch.load(ckpt_stud, map_location=device))
        student.eval()
        print(f"  Modelos carregados: {ckpt_base.name}, {ckpt_stud.name}")

        # 3. Frame probs para configs FIXAS
        # Tenta carregar cache .npy; se não existir, roda inferência.

        def _load_or_infer_fixed(model, name: str, label: str) -> np.ndarray:
            npy = Path(f"{EXP_NAME}_frame_probs_{name}_seed{SEED}.npy")
            if npy.exists():
                probs = np.load(npy)
                print(f"  [{label}] frame_probs carregados do cache: {npy.name}")
                return probs
            print(f"  [{label}] Inferindo frame_probs (sem cache)...")
            se = infer_stream_fixed(model, X_te, device)
            np.save(npy, se.frame_probs)
            print(f"  [{label}] frame_probs salvos: {npy.name}")
            return se.frame_probs

        fp_base = _load_or_infer_fixed(baseline, "base", "baseline_fixed")
        fp_kd   = _load_or_infer_fixed(student,  "kd",   "afkd_fixed")

        # 4. Política híbrida (tau_d, tau_h)
        tau_d, tau_h = load_tau_params(SEED, EXP_NAME)

        if tau_d is None:
            print("  [info] tau params não encontrados no CSV — "
                  "recalibrando no VAL (pode demorar alguns minutos)...")
            va_kd = infer_stream_fixed(student, X_va, device)
            thr_kd   = select_thr_ep(va_kd.frame_probs, yep_va)
            theta_kd = select_theta_ttd(yfr_va, va_kd.frame_probs, TTD_M)
            ref_kd   = summarize_eval(va_kd, yep_va, yfr_va, thr_kd, theta_kd)
            params   = select_hybrid_policy(
                student, X_va, yep_va, yfr_va, device,
                thr_ep=thr_kd, theta_ttd=theta_kd,
                ref_fixed=ref_kd,
            )
            if params is None:
                print("  [warn] Nenhum ponto de gating encontrado; "
                      "usando fallback tau_d=0.019, tau_h=0.95")
                tau_d, tau_h = 0.019, 0.95
            else:
                tau_d, tau_h = params
                print(f"  Gating calibrado: tau_d={tau_d:.4f}  tau_h={tau_h:.2f}")
        else:
            print(f"  Gating (do CSV): tau_d={tau_d:.4f}  tau_h={tau_h:.2f}")

        # 5. Frame probs + máscara de gating para configs HÍBRIDAS ─
        print("  Inferindo baseline_hybrid (com rastreamento de gating)...")
        fp_base_h, mask_base_h, _, skip_b, _ = infer_hybrid_with_mask(
            baseline, X_te, device, tau_d, tau_h)

        print("  Inferindo afkd_hybrid (com rastreamento de gating)...")
        fp_kd_h, mask_kd_h, _, skip_k, _ = infer_hybrid_with_mask(
            student, X_te, device, tau_d, tau_h)

        print(f"  Skip rates: baseline_hybrid={skip_b:.1f}%  "
              f"afkd_hybrid={skip_k:.1f}%")

        # 6. Construir linhas do CSV
        configs = [
            ("baseline_fixed",  fp_base,   None),
            ("afkd_fixed",      fp_kd,     None),
            ("baseline_hybrid", fp_base_h, mask_base_h),
            ("afkd_hybrid",     fp_kd_h,   mask_kd_h),
        ]

        n_te = X_te.shape[0]
        for ep_idx in range(n_te):
            is_crit = int(yep_te[ep_idx])
            is_prog = int(prog_te[ep_idx]) if is_crit else 0
            regime  = ("progressive" if is_prog
                        else "abrupt" if is_crit
                        else "normal")
            recipe  = recipe_ids[ep_idx]

            # t0: primeiro frame onde y_frame_clean > FRAME_THR
            yfc = yfrc_te[ep_idx]  # (T,) float32, ground truth limpo
            t0_frames_above = np.where(yfc > FRAME_THR)[0]
            t0_frame = int(t0_frames_above[0]) if (is_crit and len(t0_frames_above) > 0) else -1
            t0_time  = t0_frame * dt if t0_frame >= 0 else float("nan")

            for cfg_name, fp, mask in configs:
                for t in range(T_FRAMES):
                    p_t        = float(fp[ep_idx, t])
                    p_clip     = max(1e-7, min(1 - 1e-7, p_t))
                    H_t        = -(p_clip * math.log(p_clip)
                                   + (1 - p_clip) * math.log(1 - p_clip))
                    is_supp    = int(mask[ep_idx, t] == 0) if mask is not None else -1
                    phase      = compute_phase(t, t0_frame, PRE_ONSET_WINDOW)

                    all_rows.append({
                        "seed":          SEED,
                        "config":        cfg_name,
                        "episode_id":    ep_idx,
                        "regime":        regime,
                        "recipe":        recipe,
                        "is_critical":   is_crit,
                        "t0_frame":      t0_frame,
                        "t0_time_s":     round(t0_time, 4) if t0_frame >= 0 else "",
                        "t":             t,
                        "time_s":        round(t * dt, 4),
                        "p_t":           round(p_t, 6),
                        "H_t":           round(H_t, 6),
                        "is_suppressed": is_supp,
                        "y_frame_clean": round(float(yfc[t]), 6),
                        "phase":         phase,
                    })

        print(f"  Linhas acumuladas: {len(all_rows):,}")

    # 7. Salvar CSV
    if not all_rows:
        print("\n[ERRO] Nenhuma linha gerada. Verifique os erros acima.")
        return

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nok CSV salvo: {OUTPUT_CSV}")
    print(f"   Linhas: {len(df_out):,}")
    print(f"   Tamanho aprox.: {Path(OUTPUT_CSV).stat().st_size / 1e6:.1f} MB")
    print(f"\nColunas: {list(df_out.columns)}")
    print(f"\nDist. de configs:\n{df_out['config'].value_counts()}")
    print(f"\nDist. de regimes:\n{df_out['regime'].value_counts()}")
    print(f"\nAmostras por fase (is_critical==1):")
    crit = df_out[df_out["is_critical"] == 1]
    print(crit["phase"].value_counts())


if __name__ == "__main__":
    main()
