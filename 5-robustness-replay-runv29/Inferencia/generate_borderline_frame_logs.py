# -*- coding: utf-8 -*-
"""
generate_borderline_frame_logs.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gera borderline_frame_logs.csv — usado pela figura:
  · fig:prob_dist   distribuição de p_t por nível de risco (Normal /
                    Atenção / Alerta / Crítico) para Baseline e AF-TKD

Uma linha por (seed, config, episode_id, frame):

  seed | config | episode_id | risk_level | categoria | recipe
  | t | time_s | p_t | y_frame_clean | p_max_episode

RECEITAS avaliadas:
  Nível Normal  -> amostras das receitas N01–N03 do treino principal
                  (o modelo os viu como não-críticos — FPR esperada ≈ 0)
  Nível Atenção -> receitas borderline A01, A02, A03
  Nível Alerta  -> receitas borderline L01, L02, L03
  Nível Crítico -> receitas do treino C01–C10 (subconjunto do test set)

Todos os episódios borderline são gerados de novo (receitas NÃO vistas
no treino -> avalia generalização do modelo calibrado).
Os episódios Críticos e Normais vêm do test set da seed especificada.

SEEDS_TO_RUN: por padrão [42] (basta para a figura do artigo).
              Adicione mais sementes para distribuições mais robustas.

PRÉ-REQUISITOS (no mesmo diretório):
  · run_v29_ablacao_ttdef_ajuste_gatting.py   (versão patched)
  · synthetic_driver_risk_v7.py
  · dados_sinteticos/dataset_sintetico_seed{N}.npz + episodios_seed{N}.csv
  · modelos_salvos/abl_A/baseline_seed{N}.pt + student_seed{N}.pt
  · (acelera) exp_abl_A_frame_probs_base_seed{N}.npy
  · (acelera) exp_abl_A_frame_probs_kd_seed{N}.npy

SAÍDA:
  · borderline_frame_logs.csv   (~5 MB para seed 42)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Importar run_v29
RUN_V29_MODULE = "run_v29_ablacao_ttdef_ajuste_gatting"
try:
    rv = importlib.import_module(RUN_V29_MODULE)
except ModuleNotFoundError:
    sys.exit(
        f"ERRO: '{RUN_V29_MODULE}.py' não encontrado no diretório atual.\n"
        "Coloque este script na mesma pasta do run_v29."
    )

MultiTaskLSTM            = rv.MultiTaskLSTM
DatasetConfig            = rv.DatasetConfig
load_or_generate_dataset = rv.load_or_generate_dataset
make_splits              = rv.make_splits
split_arrays             = rv.split_arrays
infer_stream_fixed       = rv.infer_stream_fixed
make_prefix_window       = rv.make_prefix_window
episode_probs_from_frames= rv.episode_probs_from_frames
StreamEval               = rv.StreamEval
sync_cuda                = rv.sync_cuda
warmup_model             = rv.warmup_model
infer_class_column       = rv.infer_class_column
EXP_NAME                 = rv.EXP_NAME
MODEL_DIR                = rv.MODEL_DIR
USE_STRATIFIED_SPLITS    = rv.USE_STRATIFIED_SPLITS
FORCE_REGEN_SPLITS       = rv.FORCE_REGEN_SPLITS
T_FRAMES                 = rv.T
D                        = rv.D
H_DIM                    = rv.H
K_AGG                    = rv.K_AGG

# Importar gerador sintético
try:
    import synthetic_driver_risk_v7 as gen
except ModuleNotFoundError:
    sys.exit(
        "ERRO: 'synthetic_driver_risk_v7.py' não encontrado.\n"
        "Coloque este script na mesma pasta do gerador."
    )

# CONFIGURAÇÃO
SEEDS_TO_RUN    = [42]
FRAME_THR       = 0.35
PER_RECIPE_BL   = 80       # episódios por receita borderline (Atenção/Alerta)
PER_RECIPE_NORM = 40       # episódios Normal gerados de novo para complementar
OUTPUT_CSV      = "borderline_frame_logs.csv"

# Mapeamento categoria -> risk_level legível
RISK_LEVEL_MAP = {
    "Normal":   "Normal",
    "Atencao":  "Atenção",
    "Alerta":   "Alerta",
    "Critico":  "Crítico",
}


# HELPER — inferência por episódio -> frame_probs (n, T)

def infer_fixed_batch(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    window: int = T_FRAMES,
) -> np.ndarray:
    """
    Versão simplificada de infer_stream_fixed sem medição de latência.
    Adequado para episódios borderline onde não medimos custo.
    Retorna frame_probs (n, T).
    """
    model.eval()
    n, t_len, _ = X.shape
    frame_probs = np.zeros((n, t_len), dtype=np.float32)

    x_ref = make_prefix_window(X[0], min(5, t_len - 1), window)
    warmup_model(model, x_ref, device)

    with torch.no_grad():
        for i in range(n):
            for t in range(t_len):
                x_slice = make_prefix_window(X[i], t, window)
                xt = torch.tensor(x_slice, dtype=torch.float32,
                                   device=device).unsqueeze(0)
                fr_log, _ = model(xt)
                frame_probs[i, t] = float(
                    torch.sigmoid(fr_log[0, -1, 0]).item()
                )
    return frame_probs


# HELPER — gerar dataset borderline (Atenção + Alerta)

def generate_borderline_dataset(seed: int, per_recipe: int) -> tuple:
    """
    Gera episódios Atenção + Alerta a partir de RECIPES_BORDERLINE.
    Retorna (X, y_fr_clean, metas).
    """
    X_list, yf_list, meta_list = [], [], []
    for recipe_key, spec in gen.RECIPES_BORDERLINE.items():
        for _ in range(per_recipe):
            X_ep, _, yf_clean, _, meta = gen.generate_episode(
                recipe_key, gen.RECIPES_BORDERLINE,
                window=T_FRAMES, SENSOR_LEVEL=4,
            )
            X_list.append(X_ep)
            yf_list.append(yf_clean)
            meta_list.append(meta)
    X   = np.stack(X_list, axis=0).astype(np.float32)
    Yfc = np.stack(yf_list, axis=0).astype(np.float32)
    return X, Yfc, meta_list


def generate_normal_dataset(seed: int, per_recipe: int) -> tuple:
    """
    Gera episódios Normal para preencher o nível 'Normal' no gráfico.
    Usa RECIPES_TRAIN, filtrando só categorias Normal.
    """
    normal_recipes = {
        k: v for k, v in gen.RECIPES_TRAIN.items()
        if v.get("categoria") == "Normal"
        and k != "N01"   # exclui highway 100 km/h (overlap com padrão crítico)
    }
    X_list, yf_list, meta_list = [], [], []
    for recipe_key, spec in normal_recipes.items():
        for _ in range(per_recipe):
            X_ep, _, yf_clean, _, meta = gen.generate_episode(
                recipe_key, normal_recipes,
                window=T_FRAMES, SENSOR_LEVEL=4,
            )
            X_list.append(X_ep)
            yf_list.append(yf_clean)
            meta_list.append(meta)
    X   = np.stack(X_list, axis=0).astype(np.float32)
    Yfc = np.stack(yf_list, axis=0).astype(np.float32)
    return X, Yfc, meta_list


# HELPER — extrair episódios Críticos do test set

def load_critical_from_test(
    seed: int,
    fp_base: np.ndarray,
    fp_kd: np.ndarray,
    X_te: np.ndarray,
    yep_te: np.ndarray,
    yfrc_te: np.ndarray,
    meta_te: pd.DataFrame,
) -> dict:
    """
    Filtra os episódios críticos do test set e retorna
    {config_name: (X_crit, fp_crit, yfc_crit, meta_crit)}.
    """
    crit_mask = yep_te.astype(bool)
    return {
        "baseline_fixed": (
            X_te[crit_mask],
            fp_base[crit_mask],
            yfrc_te[crit_mask],
            meta_te[crit_mask].reset_index(drop=True),
        ),
        "afkd_fixed": (
            X_te[crit_mask],
            fp_kd[crit_mask],
            yfrc_te[crit_mask],
            meta_te[crit_mask].reset_index(drop=True),
        ),
    }


# HELPER — construir linhas do CSV para um grupo de episódios

def build_rows(
    seed: int,
    config: str,
    frame_probs: np.ndarray,
    yf_clean: np.ndarray,
    metas: list,
    risk_level: str,
) -> list:
    """
    Para cada episódio e cada frame gera uma linha do CSV.
    frame_probs: (n, T), yf_clean: (n, T), metas: lista de dicts.
    """
    rows = []
    dt = 10.0 / T_FRAMES
    n  = frame_probs.shape[0]

    for i in range(n):
        fp_ep  = frame_probs[i]         # (T,)
        yfc_ep = yf_clean[i]            # (T,)
        m      = metas[i] if i < len(metas) else {}

        recipe   = str(m.get("recipe_id", m.get("id", "unknown")))
        categoria= str(m.get("categoria", ""))
        p_max    = float(fp_ep.max())

        for t in range(T_FRAMES):
            rows.append({
                "seed":          seed,
                "config":        config,
                "episode_id":    i,
                "risk_level":    risk_level,
                "categoria":     categoria,
                "recipe":        recipe,
                "t":             t,
                "time_s":        round(t * dt, 4),
                "p_t":           round(float(fp_ep[t]), 6),
                "y_frame_clean": round(float(yfc_ep[t]), 6),
                "p_max_episode": round(p_max, 6),
            })
    return rows


# LOOP PRINCIPAL

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"EXP_NAME: {EXP_NAME}")
    print(f"MODEL_DIR: {MODEL_DIR}")

    all_rows: list = []

    for SEED in SEEDS_TO_RUN:
        print(f"\n{'='*60}")
        print(f"  SEED {SEED}")
        print(f"{'='*60}")

        # 1. Carregar modelos
        ckpt_base = MODEL_DIR / f"baseline_seed{SEED}.pt"
        ckpt_stud = MODEL_DIR / f"student_seed{SEED}.pt"
        if not ckpt_base.exists() or not ckpt_stud.exists():
            print(f"  [ERRO] Checkpoints não encontrados em {MODEL_DIR}.")
            print("  Execute run_v29 primeiro.")
            continue

        baseline = MultiTaskLSTM(D, H_DIM, bi=False).to(device)
        baseline.load_state_dict(torch.load(ckpt_base, map_location=device))
        baseline.eval()

        student = MultiTaskLSTM(D, H_DIM, bi=False).to(device)
        student.load_state_dict(torch.load(ckpt_stud, map_location=device))
        student.eval()
        print(f"  Modelos carregados.")

        # 2. Dataset principal + splits (para Crítico e Normal)
        ds_cfg = DatasetConfig(seed=SEED)
        X, y_fr_cont, _, meta = load_or_generate_dataset(ds_cfg)

        class_col = infer_class_column(meta)
        cat       = meta[class_col].astype(str).values
        y_ep      = (cat == "Critico").astype(np.int64)
        y_fr      = ((y_fr_cont > FRAME_THR).astype(np.int64) * y_ep[:, None]).astype(np.int64)

        splits = make_splits(
            y_ep, SEED,
            split_path=ds_cfg.split_path,
            reuse=ds_cfg.reuse_splits,
            meta=meta,
            use_stratified=USE_STRATIFIED_SPLITS,
            force_regen=FORCE_REGEN_SPLITS,
        )
        sliced  = split_arrays(splits, X, y_ep, y_fr_cont)
        X_te    = sliced["test"][0]
        yep_te  = sliced["test"][1]
        yfrc_te = sliced["test"][2]
        meta_te = meta.iloc[splits["test_idx"]].reset_index(drop=True)

        # 3. Frame probs do test set (Crítico + Normal)
        def _load_or_infer(model, name: str, label: str) -> np.ndarray:
            npy = Path(f"{EXP_NAME}_frame_probs_{name}_seed{SEED}.npy")
            if npy.exists():
                probs = np.load(npy)
                print(f"  [{label}] cache: {npy.name}")
                return probs
            print(f"  [{label}] inferindo...")
            se = infer_stream_fixed(model, X_te, device)
            np.save(npy, se.frame_probs)
            return se.frame_probs

        fp_base_te = _load_or_infer(baseline, "base", "baseline_fixed")
        fp_kd_te   = _load_or_infer(student,  "kd",   "afkd_fixed")

        # 4. Críticos do test set
        crit_mask   = yep_te.astype(bool)
        norm_mask   = (~crit_mask)

        for cfg_name, fp_te in [("baseline_fixed", fp_base_te),
                                  ("afkd_fixed",    fp_kd_te)]:
            # Crítico
            if crit_mask.sum() > 0:
                meta_crit = meta_te[crit_mask].reset_index(drop=True)
                metas_crit = meta_crit.to_dict("records")
                rows = build_rows(
                    SEED, cfg_name,
                    fp_te[crit_mask],
                    yfrc_te[crit_mask],
                    metas_crit,
                    risk_level="Crítico",
                )
                all_rows.extend(rows)
                print(f"  [{cfg_name}] Crítico: {crit_mask.sum()} eps -> "
                      f"{len(rows):,} linhas")

            # Normal do test set
            if norm_mask.sum() > 0:
                meta_norm = meta_te[norm_mask].reset_index(drop=True)
                # Filtrar somente categoria Normal (excluir Alerta/Atenção se existirem)
                is_normal = meta_norm[class_col].astype(str).eq("Normal")
                if is_normal.sum() > 0:
                    meta_n2 = meta_norm[is_normal].reset_index(drop=True)
                    fp_norm = fp_te[norm_mask][is_normal.values]
                    yfc_norm = yfrc_te[norm_mask][is_normal.values]
                    metas_norm = meta_n2.to_dict("records")
                    rows = build_rows(
                        SEED, cfg_name,
                        fp_norm, yfc_norm, metas_norm,
                        risk_level="Normal",
                    )
                    all_rows.extend(rows)
                    print(f"  [{cfg_name}] Normal (test): {is_normal.sum()} eps -> "
                          f"{len(rows):,} linhas")

        # 5. Gerar episódios Normal extras
        # Complementa os Normais do test set com episódios gerados
        print("  Gerando episódios Normal extras...")
        X_norm, yfc_norm_gen, meta_norm_gen = generate_normal_dataset(
            SEED, PER_RECIPE_NORM)

        for cfg_name, model in [("baseline_fixed", baseline),
                                  ("afkd_fixed",    student)]:
            fp_norm_gen = infer_fixed_batch(model, X_norm, device)
            rows = build_rows(
                SEED, cfg_name,
                fp_norm_gen, yfc_norm_gen,
                meta_norm_gen, risk_level="Normal",
            )
            all_rows.extend(rows)
            print(f"  [{cfg_name}] Normal (gerado): {X_norm.shape[0]} eps -> "
                  f"{len(rows):,} linhas")

        # 6. Gerar episódios Atenção + Alerta
        print("  Gerando episódios Atenção + Alerta (RECIPES_BORDERLINE)...")
        X_bl, yfc_bl, meta_bl = generate_borderline_dataset(SEED, PER_RECIPE_BL)

        # Mapa de recipe_id -> risk_level
        recipe_to_level = {}
        for k, spec in gen.RECIPES_BORDERLINE.items():
            cat_bl = spec.get("categoria", "")
            recipe_to_level[k] = RISK_LEVEL_MAP.get(cat_bl, cat_bl)

        for cfg_name, model in [("baseline_fixed", baseline),
                                  ("afkd_fixed",    student)]:
            fp_bl = infer_fixed_batch(model, X_bl, device)

            # Separa por nível
            for level_key in ["Atenção", "Alerta"]:
                cat_code = "Atencao" if level_key == "Atenção" else "Alerta"
                idx_level = [
                    i for i, m in enumerate(meta_bl)
                    if m.get("categoria") == cat_code
                ]
                if not idx_level:
                    continue
                idx_arr  = np.array(idx_level)
                meta_sub = [meta_bl[i] for i in idx_level]
                rows = build_rows(
                    SEED, cfg_name,
                    fp_bl[idx_arr],
                    yfc_bl[idx_arr],
                    meta_sub,
                    risk_level=level_key,
                )
                all_rows.extend(rows)
                print(f"  [{cfg_name}] {level_key}: {len(idx_level)} eps -> "
                      f"{len(rows):,} linhas")

        print(f"  Total de linhas acumuladas: {len(all_rows):,}")

    # 7. Salvar CSV
    if not all_rows:
        print("\n[ERRO] Nenhuma linha gerada.")
        return

    df_out = pd.DataFrame(all_rows)

    # Estatística resumida por risk_level e config
    print("\n=== RESUMO — p_max_episode por nível e config ===")
    for cfg in df_out["config"].unique():
        print(f"\n  {cfg}:")
        sub = df_out[df_out["config"] == cfg]
        ep_stats = (
            sub.groupby(["risk_level", "episode_id"])["p_t"]
            .max()
            .reset_index()
            .groupby("risk_level")["p_t"]
            .agg(["mean", "std", "count"])
        )
        print(ep_stats.round(4))

    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nok CSV salvo: {OUTPUT_CSV}")
    print(f"   Linhas: {len(df_out):,}")
    print(f"   Tamanho aprox.: {Path(OUTPUT_CSV).stat().st_size / 1e6:.1f} MB")
    print(f"\nNíveis de risco presentes:\n{df_out['risk_level'].value_counts()}")
    print(f"\nConfigs presentes:\n{df_out['config'].value_counts()}")


if __name__ == "__main__":
    main()
