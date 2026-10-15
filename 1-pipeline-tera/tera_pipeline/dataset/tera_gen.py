# -*- coding: utf-8 -*-
"""
tera_pipeline/dataset/tera_gen.py
Gerenciamento de dataset sintético do TERA Pipeline.

Responsabilidades:
  · Gerar/reutilizar dataset_sintetico_seed{N}.npz via synthetic_driver_risk_v7
  · Verificar hash do gerador para detectar mudanças silenciosas
  · Criar splits determinísticos (70/15/15, estratificado por recipe)
  · Salvar splits como .npy em exp_dir/data/ para uso pelos módulos downstream
  · Exportar dataset_metadata_seed{N}.json para rastreabilidade

SAÍDAS em exp_dir/data/:
  X_tr_seed{N}.npy, X_va_seed{N}.npy, X_te_seed{N}.npy
  y_fr_tr_seed{N}.npy, y_fr_va_seed{N}.npy, y_fr_te_seed{N}.npy
  y_ep_tr_seed{N}.npy, y_ep_va_seed{N}.npy, y_ep_te_seed{N}.npy
  progressive_tr_seed{N}.npy, progressive_va_seed{N}.npy, progressive_te_seed{N}.npy
  dataset_metadata_seed{N}.json
"""

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Constantes
GENERATOR_MODULE = "synthetic_driver_risk_v7"
FRAME_THR        = 0.35     # limiar binário frame-level (alinhado com run_v29)


def _hash_generator(module_name: str) -> str:
    """Hash SHA-256 do módulo gerador para detectar alterações."""
    candidates = [
        Path(f"{module_name}.py"),
        Path("tera_pipeline") / "dataset" / f"{module_name}.py",
    ]
    for p in candidates:
        if p.exists():
            sha = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            return sha
    return "not_found"


def _load_generator():
    """Importa o módulo gerador de dataset."""
    try:
        return importlib.import_module(GENERATOR_MODULE)
    except ModuleNotFoundError:
        raise ImportError(
            f"Módulo '{GENERATOR_MODULE}.py' não encontrado.\n"
            f"Certifique-se de que '{GENERATOR_MODULE}.py' está na raiz do projeto."
        )


# DatasetManager

class DatasetManager:
    """
    Gerencia geração e splits do dataset sintético para todas as seeds.
    """

    def __init__(self, cfg: dict, exp_dir: Path, seeds: List[int]):
        self.cfg      = cfg
        self.exp_dir  = exp_dir
        self.seeds    = seeds
        self.ds_cfg   = cfg.get("dataset", {})
        self.sp_cfg   = cfg.get("splits", {})
        self.data_dir = Path(self.ds_cfg.get("data_dir", "dados_sinteticos"))
        self.out_dir  = exp_dir / "data"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Parâmetros do gerador
        self.window     = self.ds_cfg.get("window", 96)
        self.per_recipe = self.ds_cfg.get("per_recipe", 80)
        self.input_dim  = self.ds_cfg.get("input_dim", 31)
        self.frame_thr  = self.ds_cfg.get("frame_thr", FRAME_THR)

        # Parâmetros dos splits
        self.train_r  = self.sp_cfg.get("train_ratio", 0.70)
        self.val_r    = self.sp_cfg.get("val_ratio",   0.15)
        self.force    = self.sp_cfg.get("force_regen", False)

    # Geração

    def _npz_path(self, seed: int) -> Path:
        return self.data_dir / f"dataset_sintetico_seed{seed}.npz"

    def _csv_path(self, seed: int) -> Path:
        return self.data_dir / f"episodios_seed{seed}.csv"

    def _splits_exist(self, seed: int) -> bool:
        return (self.out_dir / f"X_te_seed{seed}.npy").exists()

    def generate_seed(self, seed: int, reuse: bool = True) -> None:
        """Gera dataset para uma seed, reutilizando se existir."""
        npz = self._npz_path(seed)
        csv = self._csv_path(seed)

        if reuse and npz.exists() and csv.exists() and not self.force:
            print(f"    [cache] dataset seed={seed} — reutilizando {npz.name}")
        else:
            print(f"    Gerando dataset seed={seed}...")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            gen = _load_generator()
            gen.quick_generate(
                out_npz=str(npz),
                out_csv=str(csv),
                seed=seed,
                window=self.window,
                per_recipe=self.per_recipe,
            )
            print(f"    Dataset gerado: {npz.name}")

    # Splits

    def _make_splits(
        self, seed: int
    ) -> Dict[str, np.ndarray]:
        """
        Cria splits estratificados 70/15/15 por recipe_id (críticos)
        e por class (não-críticos). Alinhado com o protocolo do run_v29.
        """
        npz  = np.load(self._npz_path(seed))
        meta = pd.read_csv(self._csv_path(seed), sep=";")

        X       = npz["X"].astype(np.float32)
        y_fr_cont = npz.get("y_frame_clean", npz.get("y_frame")).astype(np.float32)

        # y_episode: binário por recipe/class
        class_col = self._infer_class_col(meta)
        cat  = meta[class_col].astype(str).values
        y_ep = (cat == "Critico").astype(np.int64)

        # y_frame: binário mascarado por episódio
        y_fr = (
            (y_fr_cont > self.frame_thr).astype(np.int64) * y_ep[:, None]
        ).astype(np.int64)

        # Flag progressivo
        prog = (
            meta["progressive"].fillna(0).astype(int).values.astype(bool)
            if "progressive" in meta.columns
            else np.zeros(len(meta), dtype=bool)
        )
        prog = prog & (y_ep == 1)

        # Chave de estratificação: recipe_id para críticos, class para normais
        strat_key = np.where(
            cat == "Critico",
            meta["recipe_id"].astype(str).values,
            cat,
        )

        idx = np.arange(len(X))
        tr_idx, tmp_idx = train_test_split(
            idx, test_size=1 - self.train_r,
            random_state=seed, stratify=strat_key
        )
        val_frac = self.val_r / (self.val_r + (1 - self.train_r - self.val_r))
        va_idx, te_idx = train_test_split(
            tmp_idx, test_size=0.5,
            random_state=seed, stratify=strat_key[tmp_idx]
        )

        return {
            "tr": (tr_idx, X[tr_idx], y_fr[tr_idx], y_ep[tr_idx], prog[tr_idx]),
            "va": (va_idx, X[va_idx], y_fr[va_idx], y_ep[va_idx], prog[va_idx]),
            "te": (te_idx, X[te_idx], y_fr[te_idx], y_ep[te_idx], prog[te_idx]),
        }

    def _infer_class_col(self, meta: pd.DataFrame) -> str:
        for col in ("class_name", "categoria", "class"):
            if col in meta.columns:
                return col
        raise ValueError("Coluna de classe não encontrada no metadata.")

    def save_splits(self, seed: int) -> None:
        """Salva splits em .npy no exp_dir/data/."""
        splits = self._make_splits(seed)
        for split_name, (idx, X, y_fr, y_ep, prog) in splits.items():
            s = split_name  # tr / va / te
            np.save(self.out_dir / f"X_{s}_seed{seed}.npy",              X)
            np.save(self.out_dir / f"y_fr_{s}_seed{seed}.npy",           y_fr)
            np.save(self.out_dir / f"y_ep_{s}_seed{seed}.npy",           y_ep)
            np.save(self.out_dir / f"progressive_{s}_seed{seed}.npy",    prog)
            np.save(self.out_dir / f"indices_{s}_seed{seed}.npy",        idx)

    def save_metadata(self, seed: int) -> None:
        """Salva metadados do dataset para rastreabilidade."""
        meta = {
            "seed":           seed,
            "generator":      GENERATOR_MODULE,
            "generator_hash": _hash_generator(GENERATOR_MODULE),
            "window":         self.window,
            "per_recipe":     self.per_recipe,
            "input_dim":      self.input_dim,
            "frame_thr":      self.frame_thr,
            "train_ratio":    self.train_r,
            "val_ratio":      self.val_r,
        }
        # Conta episódios
        npz_path = self._npz_path(seed)
        if npz_path.exists():
            npz = np.load(npz_path)
            meta["n_episodes"] = int(npz["X"].shape[0])

        out = self.out_dir / f"dataset_metadata_seed{seed}.json"
        out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Orquestrador

    def run(self, reuse: bool = True) -> None:
        """Executa geração + splits para todas as seeds."""
        print(f"  Seeds: {self.seeds}")
        for seed in self.seeds:
            print(f"\n  [Dataset] seed={seed}")
            self.generate_seed(seed, reuse=reuse)
            if not self._splits_exist(seed) or self.force:
                print(f"    Criando splits...")
                self.save_splits(seed)
            else:
                print(f"    [cache] splits seed={seed} — OK")
            self.save_metadata(seed)

        self._verify_consistency()

    def _verify_consistency(self) -> None:
        """Verifica que todos os splits existem e têm dimensões consistentes."""
        print("\n  Verificando consistência dos datasets...")
        for seed in self.seeds:
            for split in ["tr", "va", "te"]:
                p = self.out_dir / f"X_{split}_seed{seed}.npy"
                if not p.exists():
                    raise FileNotFoundError(f"Split faltando: {p}")
                X = np.load(p)
                assert X.ndim == 3 and X.shape[2] == self.input_dim, \
                    f"Dimensão inesperada em {p}: {X.shape}"
        print("  Todos os datasets OK")
