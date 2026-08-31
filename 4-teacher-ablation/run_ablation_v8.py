# -*- coding: utf-8 -*-
"""
run_ablation_v8.py
ABLAÇÃO COMPLETA v8: TEACHER STUDY  ─  BiLSTM / Transformer / None

MUDANÇAS v7 -> v8 (CORREÇÕES DE PROTOCOLO — ALINHAMENTO COM run_v29):

  CORREÇÃO 1 — Gerador sintético: v6 -> v7
    • DEFAULT_GENERATOR_MODULE = "synthetic_driver_risk_v7"
    • quick_generate() chamado sem recipe_set / SENSOR_LEVEL (interface v7)
    • y_frame_obs carregado de forma defensiva (fallback para y_frame_clean se
      ausente no .npz — campo não garantido no v7)
    • y_episode derivado de y_frame_clean via limiar frame_thr (compatível v7)
    • EXP_NAME atualizado para "exp_ablacao_v8" — evita colisão com resultados
      do experimento anterior gerado com v6

  CORREÇÃO 2 — TTDef alinhado ao run_v29
    • Adicionado campo TTDef = TTD_bruto + FR × T_EP (T_EP = 10.0 s) em
      summarize_eval, idêntico à fórmula usada nas macros do artigo
      (\\resAlvoHibrTTDef, \\resBaseFixoTTDef, etc.)
    • Campo "TTD" (penalizado via (window-t0)*dt) preservado para
      retrocompatibilidade com análises de robustez internas
    • TTDef adicionado ao CSV de exportação, ao sumário e à tabela LaTeX
    • Constante T_EP = 10.0 declarada explicitamente na seção de configuração

  Protocolo original mantido integralmente:
    • 5 modelos × 4 seeds (42–45)
    • θ_TTD calibrado por modelo (FailRate ≤ FAIL_BUDGET=0.05)
    • Temperature scaling pós-treino
    • Análises 1–5 de robustez inalteradas

REGRA DE OURO DO PROTOCOLO (paper):
  ok TTD_efetivo = penalizado: USAR no paper (métrica justa)
  aviso: TTD_bruto  = sem pena:   somente análise auxiliar (exclui misses -> otimista)

EXPERIMENTOS (5 modelos × 4 seeds = 20 treinos):
  ┌──────────────────────────┬─────────────┬────────────────────────────────┐
  │ Experimento              │ Teacher     │ Pergunta respondida            │
  ├──────────────────────────┼─────────────┼────────────────────────────────┤
  │ LSTM-Baseline            │ —           │ baseline sem KD                │
  │ LSTM-AF-KD               │ BiLSTM      │ teacher recorrente -> student   │
  │ LSTM-TransKD ★ NOVO      │ Transformer │ teacher atencional -> student   │
  │ Transformer-Baseline     │ —           │ baseline sem KD                │
  │ Transformer-AF-KD        │ BiLSTM      │ teacher recorrente -> atencional│
  └──────────────────────────┴─────────────┴────────────────────────────────┘

  Ablação científica central:
    LSTM-AF-KD (BiLSTM->LSTM)  vs  LSTM-TransKD (Transformer->LSTM)
    -> Student (LSTM causal) idêntico; apenas o teacher muda.
    -> Isola o efeito do teacher atencional (full-sequence) vs. recorrente.

PROTOCOLO IDÊNTICO GARANTIDO:
  ok Mesmo dataset sintético  (synthetic_driver_risk_v7.py)  [v8: atualizado de v6]
  ok Mesmas seeds             [42, 43, 44, 45]
  ok Mesmos splits train/val/test  (salvos em JSON e reutilizados)
  ok Mesma estratificação     por recipe_id (críticos)
  ok Mesmo m=3                para TTD (consecutivos)
  ok Mesmo FailRate_max=0.05
  ok Mesmo K_AGG=6            (agregação causal)
  ok Mesma calibração de θ    via validação (θ global simétrico)
  ok Mesmo formato de saída   CSV + LaTeX
  ok TTDef = TTD_bruto + FR × T_EP  [v8: alinhado com run_v29]

COMO RODAR:
  Pré-requisito: o gerador sintético deve estar no mesmo diretório:
    synthetic_driver_risk_v7.py

  Instalação de dependências (se necessário):
    pip install torch numpy pandas scikit-learn matplotlib

  Execução completa (todas as seeds, CPU ou GPU):
    python run_ablation_teacher_study.py

  Modo rápido (1 seed para validar o setup):
    python run_ablation_teacher_study.py --seed 42

  Reutilizar dataset já gerado (padrão = True):
    REUSE_PREGENERATED_DATASET = False  (obrigatório ao trocar gerador v4->v5)

  Forçar regerar splits:
    FORCE_REGEN_SPLITS = True   (edite a constante no código)

SAÍDAS (em ./resultados_unified/):
  ├─ results_all_seeds_unified.csv    <- valores por seed × modelo
  ├─ summary_unified.csv              <- média ± std por (Modelo, Método, Teacher)
  └─ table_comparison_latex.tex       <- tabela LaTeX com coluna Teacher

  Frame probs (em ./dados_sinteticos/frame_probs/):
  ├─ lstm_base_seedXX.npy
  ├─ lstm_afkd_seedXX.npy
  ├─ lstm_transkd_seedXX.npy    <- NOVO: Transformer->LSTM
  ├─ trans_base_seedXX.npy
  ├─ trans_afkd_seedXX.npy
  └─ yfr_test_seedXX.npy
  (usados por analise_fairness_v4.py e analise_robustez_afkd_fgcs_v2.ipynb)

MÉTRICAS POR MODELO:
  F1, FailRate (≤0.05), TTD total, TTD_Progressive, TTD_Abrupt, ECE, Lat_ms

TEMPO ESTIMADO:
  CPU: ~90–120 min (20 treinos × ~5min cada)
  GPU: ~20–30 min

"""

import argparse
import json
import math
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# 0) CONFIGURAÇÃO GLOBAL — PROTOCOLO UNIFICADO
#    Altere apenas aqui; o resto do código não deve precisar de edições.

SEEDS            = [42, 43, 44, 45]

# Identificador do experimento
# Prefixo aplicado em TODOS os arquivos de saída para distinguir dos resultados
# do experimento principal (run_v22_otimizado.py -> EXP_NAME = "exp_main").
# Permite comparação direta: exp_ablacao_* vs exp_main_*
EXP_NAME         = "exp_ablacao_v8"  # [v8] separado dos resultados gerados com v6
T                = 96       # janela temporal (frames por episódio)
D                = 31       # dimensão de entrada (features)
H                = 64       # hidden size LSTM
K_AGG            = 6        # janela de agregação causal
TTD_M            = 3        # mínimo de frames consecutivos para detecção
FAIL_BUDGET      = 0.05     # FailRate máximo tolerado
LAT_WARMUP_STEPS = 100      # passos de aquecimento para medição de latência
T_EP             = 10.0     # [v8] duração do episódio (s) — penalidade TTDef

# Grade de θ global (calibrada na validação)
THETA_GLOBAL_GRID        = list(np.concatenate([
    np.array([0.10, 0.12, 0.15, 0.18, 0.20], dtype=float),
    np.linspace(0.25, 0.90, 14),
]))
THETA_GLOBAL_FAILRATE_MAX = FAIL_BUDGET

# Grade de m para análise de sensibilidade (Análise 4)
M_SENS_GRID = [1, 2, 3, 4, 5]

# Grade de θ para análise de sensibilidade de θ (Análise 3) — 18 pontos entre 0.05 e 0.60
THETA_SENS_GRID = [round(0.05 + i * (0.60 - 0.05) / 17, 3) for i in range(18)]

# Calibração v3 fiel ao run_v22
KIN = [12, 13, 14]
TTD_BUDGET_RATIO = 1.20
TAU_DELTA_PERCENTILES = list(range(50, 96, 3))
TAU_H_GRID = [0.40, 0.55, 0.70, 0.85, 0.95]
THR_EP_GRID = np.concatenate([
    np.linspace(0.05, 0.40, 15),
    np.linspace(0.40, 0.95, 12),
])
THETA_TTD_ADAPT_GRID = np.array([0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30], dtype=float)
TTD_ADAPT_TARGET_THETA = 0.25
TTD_ADAPT_MAX_MISS = 0.05
DERIV_DELTA_GRID = np.array([0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020], dtype=float)
DERIV_PROB_GRID = np.array([0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30], dtype=float)
DERIV_SMOOTH_GRID = [3, 5]
DERIV_TARGET_DELTA = 0.006
DERIV_TARGET_PROB = 0.18
DERIV_MAX_MISS = 0.10
RUN_HYBRID_V3 = True

# Configuração de dataset
DATA_ROOT                  = Path(".")
DATA_DIR                   = DATA_ROOT / "dados_sinteticos"
REUSE_PREGENERATED_DATASET = False  # IMPORTANTE: False obrigatório ao trocar gerador (v4->v5)
REUSE_SPLITS               = True   # True -> reutiliza splits_seedXX.json
USE_STRATIFIED_SPLITS      = True   # estratificação por recipe_id
FORCE_REGEN_SPLITS         = True   # True -> força novo split mesmo que JSON exista
DEFAULT_GENERATOR_MODULE   = "synthetic_driver_risk_v7"  # [v8] atualizado de v6 -> v7
# v6: RECIPES_TRAIN = Normal + Crítico apenas (Atenção e Alerta movidos para RECIPES_BORDERLINE)
# Label principal = y_episode_severity (0.7·mean + 0.3·p95), suavização
# temporal 3 frames, decaimento pós-plateau, +2 receitas progressivas (60%)

# Arquitetura do Transformer Student (causal — usa máscara)
TRANSFORMER_CONFIG = {
    "input_dim":       D,
    "d_model":         96,
    "nhead":           4,
    "num_layers":      3,
    "dim_feedforward": 192,
    "dropout":         0.10,
    "max_len":         128,
    "frame_hidden":    96,
    "episode_hidden":  64,
}

# Arquitetura do TransformerTeacher (não-causal — full-sequence, sem máscara)
# Mesmos hiperparâmetros do Student para comparação justa; separado para
# poder ajustar independentemente se necessário.
TRANSFORMER_TEACHER_CONFIG = {
    "input_dim":       D,
    "d_model":         96,
    "nhead":           4,
    "num_layers":      3,
    "dim_feedforward": 192,
    "dropout":         0.10,
    "max_len":         128,
    "frame_hidden":    96,
    "episode_hidden":  64,
}

# Arquitetura LSTM
LSTM_CONFIG = {
    "input_dim":   D,
    "hidden_size": H,
    "num_layers":  2,
    "dropout":     0.10,
}


# 1) DATACLASSES E HELPERS

@dataclass
class DatasetConfig:
    seed: int
    window: int = T
    per_recipe: int = 80
    recipe_set: str = "base"
    sensor_level: int = 4
    frame_thr: float = 0.35
    generator_module: str = DEFAULT_GENERATOR_MODULE
    # Path mutable default correto com field(default_factory=...)
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    reuse_pregenerated: bool = REUSE_PREGENERATED_DATASET
    reuse_splits: bool = REUSE_SPLITS

    @property
    def npz_path(self) -> Path:
        return Path(self.data_dir) / f"dataset_sintetico_seed{self.seed}.npz"

    @property
    def csv_path(self) -> Path:
        return Path(self.data_dir) / f"episodios_seed{self.seed}.csv"

    @property
    def config_path(self) -> Path:
        return Path(self.data_dir) / f"dataset_config_seed{self.seed}.json"

    @property
    def split_path(self) -> Path:
        return Path(self.data_dir) / f"splits_seed{self.seed}.json"


@dataclass
class InferResult:
    frame_probs:   np.ndarray
    episode_probs: np.ndarray
    episode_pred:  np.ndarray
    latencies_ms:  List[float]
    total_time:    float


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    raise TypeError(f"Cannot serialize {type(obj)}")


def _cfg_to_dict(cfg: DatasetConfig) -> Dict[str, Any]:
    """Converte DatasetConfig para dict serializável (Path -> str)."""
    d = asdict(cfg)
    d["data_dir"] = str(d["data_dir"])
    return d


def _cfg_dicts_match(saved: Dict, current: Dict) -> bool:
    """Compara dois dicts de config tolerando Path vs str."""
    for k in current:
        v_c = str(current[k]) if isinstance(current[k], Path) else current[k]
        v_s = str(saved.get(k, "")) if isinstance(saved.get(k, ""), Path) else saved.get(k, "")
        if v_c != v_s:
            return False
    return True


def infer_class_column(meta_df: pd.DataFrame) -> str:
    for col in ("class_name", "categoria", "class"):
        if col in meta_df.columns:
            return col
    raise KeyError("Nenhuma coluna de classe encontrada no CSV de metadados (esperado: class_name/categoria/class).")


# 2) MODELOS — FAMÍLIA LSTM

class TeacherModel(nn.Module):
    """
    Professor BiLSTM — vê a sequência em ambas as direções (oráculo recorrente).
    Compartilhado entre LSTM-AF-KD e Transformer-AF-KD.
    """
    def __init__(self, input_dim: int = D, hidden_size: int = H,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.frame_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, 1),
        )
        self.episode_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.lstm(x)
        h = self.norm(h)
        frame_logits = self.frame_head(h).squeeze(-1)
        ep_logits = self.episode_head(h[:, -1, :]).squeeze(-1)
        return frame_logits, ep_logits


class LSTMStudent(nn.Module):
    """
    Aluno LSTM causal — processa frames estritamente da esquerda para a direita.
    Usado como student em: LSTM-Baseline, LSTM-AF-KD e LSTM-TransKD.
    """
    def __init__(self, input_dim: int = D, hidden_size: int = H,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.frame_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, 1),
        )
        self.episode_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.lstm(x)
        h = self.norm(h)
        frame_logits = self.frame_head(h).squeeze(-1)
        ep_logits = self.episode_head(h[:, -1, :]).squeeze(-1)
        return frame_logits, ep_logits


# 3) MODELOS — FAMÍLIA TRANSFORMER

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


def build_causal_mask(T_len: int, device: torch.device) -> torch.Tensor:
    """Máscara causal: posição t não pode ver t+1, t+2, ..."""
    return torch.triu(torch.ones(T_len, T_len, dtype=torch.bool, device=device), diagonal=1)


def _make_encoder(cfg: Dict[str, Any]) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=cfg["d_model"], nhead=cfg["nhead"],
        dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"],
        activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=cfg["num_layers"])


class TransformerStudent(nn.Module):
    """
    Aluno Transformer causal — atenção mascarada, processa frames da esq. p/ dir.
    Usado como student em: Transformer-Baseline e Transformer-AF-KD.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.in_proj = nn.Linear(cfg["input_dim"], cfg["d_model"])
        self.pos_enc = PositionalEncoding(cfg["d_model"], max_len=cfg["max_len"])
        self.encoder = _make_encoder(cfg)
        self.norm = nn.LayerNorm(cfg["d_model"])
        self.frame_head = nn.Sequential(
            nn.Linear(cfg["d_model"], cfg["frame_hidden"]), nn.GELU(),
            nn.Dropout(cfg["dropout"]), nn.Linear(cfg["frame_hidden"], 1),
        )
        self.ep_head = nn.Sequential(
            nn.Linear(cfg["d_model"], cfg["episode_hidden"]), nn.GELU(),
            nn.Dropout(cfg["dropout"]), nn.Linear(cfg["episode_hidden"], 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.pos_enc(self.in_proj(x))
        mask = build_causal_mask(x.size(1), x.device)
        h = self.norm(self.encoder(h, mask=mask))
        return self.frame_head(h).squeeze(-1), self.ep_head(h[:, -1, :]).squeeze(-1)


class TransformerTeacher(nn.Module):
    """
    Professor Transformer — atenção FULL-SEQUENCE (sem máscara causal).

    Diferença crítica vs TransformerStudent:
      • Sem build_causal_mask -> acesso a frames futuros durante treino.
      • Funciona como oráculo atencional, análogo ao papel do BiLSTM teacher.
      • Usado exclusivamente para destilação no experimento LSTM-TransKD.

    Hipótese de ablação:
      Se LSTM-TransKD > LSTM-AF-KD, o ganho vem do poder representacional
      do teacher atencional, não da arquitetura do student (ambos são LSTM causal).
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.in_proj = nn.Linear(cfg["input_dim"], cfg["d_model"])
        self.pos_enc = PositionalEncoding(cfg["d_model"], max_len=cfg["max_len"])
        self.encoder = _make_encoder(cfg)
        self.norm = nn.LayerNorm(cfg["d_model"])
        self.frame_head = nn.Sequential(
            nn.Linear(cfg["d_model"], cfg["frame_hidden"]), nn.GELU(),
            nn.Dropout(cfg["dropout"]), nn.Linear(cfg["frame_hidden"], 1),
        )
        self.ep_head = nn.Sequential(
            nn.Linear(cfg["d_model"], cfg["episode_hidden"]), nn.GELU(),
            nn.Dropout(cfg["dropout"]), nn.Linear(cfg["episode_hidden"], 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.pos_enc(self.in_proj(x))
        # SEM máscara causal -> full-sequence attention (modo oráculo)
        h = self.norm(self.encoder(h))
        return self.frame_head(h).squeeze(-1), self.ep_head(h[:, -1, :]).squeeze(-1)


# 4) DATASET — GERAÇÃO E CARREGAMENTO

def load_or_generate_dataset(cfg: DatasetConfig) -> Tuple[np.ndarray, ...]:
    """
    Carrega dataset sintético v7 se já existir (.npz + JSON de config),
    ou gera um novo chamando quick_generate() do gerador v7.

    Label principal em y_episode = 0.7·mean(y_frame) + 0.3·p95(y_frame).

    [v8] Interface v7: quick_generate não aceita recipe_set nem SENSOR_LEVEL.
    [v8] y_frame_obs carregado de forma defensiva: fallback para y_frame_clean
         se a chave não estiver no .npz (gerador v7 pode não incluí-la).
    [v8] y_episode derivado de y_frame_clean quando ausente no .npz.

    Reprodutibilidade garantida via:
      • config salvo em JSON e comparado antes de reusar
      • seed fixada antes de qualquer operação aleatória
    """
    set_seed(cfg.seed)
    ensure_dir(cfg.data_dir)

    # Tentar reutilizar
    if cfg.reuse_pregenerated and cfg.npz_path.exists() and cfg.config_path.exists():
        with open(cfg.config_path, "r") as f:
            saved_cfg = json.load(f)
        current_cfg = _cfg_to_dict(cfg)
        if _cfg_dicts_match(saved_cfg, current_cfg):
            print(f"  Reusando dataset pré-gerado: {cfg.npz_path}")
            data = np.load(cfg.npz_path)
            meta_df = pd.read_csv(cfg.csv_path, sep=";")
            X = data["X"]
            y_frame_clean = data["y_frame_clean"]
            # [v8] defensive: y_frame_obs pode não existir no v7
            y_frame_obs = data["y_frame_obs"] if "y_frame_obs" in data.files else y_frame_clean
            # [v8] defensive: y_episode pode não existir; deriva de y_frame_clean
            if "y_episode" in data.files:
                y_episode = data["y_episode"]
            else:
                y_episode = (
                    0.7 * y_frame_clean.mean(axis=1)
                    + 0.3 * np.percentile(y_frame_clean, 95, axis=1)
                )
            return X, y_frame_clean, y_frame_obs, y_episode, meta_df

    # Gerar novo
    print(f"  ⚙ Gerando novo dataset (seed={cfg.seed}, gerador={cfg.generator_module})...")
    sys.path.insert(0, str(Path(cfg.data_dir).parent))
    try:
        import importlib
        sd = importlib.import_module(cfg.generator_module)
    except ImportError:
        raise ImportError(
            f"Gerador não encontrado: {cfg.generator_module}\n"
            f"Certifique-se de que o arquivo está em: {Path(cfg.data_dir).parent}"
        )

    # [v8] Interface v7: apenas out_npz, out_csv, window, per_recipe, seed
    # recipe_set e SENSOR_LEVEL foram removidos (interface v6 apenas)
    sd.quick_generate(
        out_npz=str(cfg.npz_path),
        out_csv=str(cfg.csv_path),
        window=cfg.window,
        per_recipe=cfg.per_recipe,
        seed=cfg.seed,
    )

    with open(cfg.config_path, "w") as f:
        json.dump(_cfg_to_dict(cfg), f, indent=2, default=_json_default)

    data = np.load(cfg.npz_path)
    meta_df = pd.read_csv(cfg.csv_path, sep=";")
    X = data["X"]
    y_frame_clean = data["y_frame_clean"]
    # [v8] defensive loading para campos opcionais do v7
    y_frame_obs = data["y_frame_obs"] if "y_frame_obs" in data.files else y_frame_clean
    if "y_episode" in data.files:
        y_episode = data["y_episode"]
    else:
        y_episode = (
            0.7 * y_frame_clean.mean(axis=1)
            + 0.3 * np.percentile(y_frame_clean, 95, axis=1)
        )
    print(f"  Dataset gerado: shape={data['X'].shape}, episódios={len(meta_df)}")
    return X, y_frame_clean, y_frame_obs, y_episode, meta_df


def create_stratified_splits(
    X: np.ndarray,
    y_frame_main: np.ndarray,
    y_episode: np.ndarray,
    meta_df: pd.DataFrame,
    cfg: DatasetConfig,
    test_size: float = 0.15,   # ALINHADO ao run_v22: 70/15/15
    val_size: float  = 0.15,   # ALINHADO ao run_v22: 70/15/15
) -> Dict[str, np.ndarray]:
    """
    Splits estratificados alinhados ao protocolo do run_v22:
      • split sobre TODO o dataset (não apenas críticos)
      • críticos estratificados por recipe_id
      • não-críticos estratificados por classe semântica

    Também aceita reuso de JSON no formato novo (train_idx/val_idx/test_idx)
    ou antigo (idx_train/idx_val/idx_test).
    """
    split_file = cfg.split_path

    if cfg.reuse_splits and split_file.exists() and not FORCE_REGEN_SPLITS:
        print(f"  Reusando splits salvos: {split_file}")
        with open(split_file, "r") as f:
            payload = json.load(f)

        if all(k in payload for k in ("train_idx", "val_idx", "test_idx")):
            return {
                "train_idx": np.asarray(payload["train_idx"], dtype=np.int64),
                "val_idx":   np.asarray(payload["val_idx"], dtype=np.int64),
                "test_idx":  np.asarray(payload["test_idx"], dtype=np.int64),
            }

        if all(k in payload for k in ("idx_train", "idx_val", "idx_test")):
            return {
                "train_idx": np.asarray(payload["idx_train"], dtype=np.int64),
                "val_idx":   np.asarray(payload["idx_val"], dtype=np.int64),
                "test_idx":  np.asarray(payload["idx_test"], dtype=np.int64),
            }

        raise ValueError(
            f"Formato de split inválido em {split_file}. "
            f"Chaves encontradas: {sorted(payload.keys())}"
        )

    print(f"  ⚙ Criando novos splits estratificados (seed={cfg.seed})...")
    set_seed(cfg.seed)

    idx = np.arange(len(y_episode), dtype=np.int64)
    class_col = infer_class_column(meta_df)

    class_values = meta_df[class_col].astype(str).str.strip().values
    recipe_values = meta_df["recipe_id"].astype(str).str.strip().values if "recipe_id" in meta_df.columns else class_values

    strat_key = np.where(class_values == "Critico", recipe_values, class_values)

    idx_train, idx_tmp = train_test_split(
        idx,
        test_size=(test_size + val_size),
        random_state=cfg.seed,
        stratify=strat_key if USE_STRATIFIED_SPLITS else None,
    )

    tmp_ratio = test_size / (test_size + val_size)
    idx_val, idx_test = train_test_split(
        idx_tmp,
        test_size=tmp_ratio,
        random_state=cfg.seed,
        stratify=strat_key[idx_tmp] if USE_STRATIFIED_SPLITS else None,
    )

    splits_dict = {
        "train_idx": np.sort(idx_train).tolist(),
        "val_idx":   np.sort(idx_val).tolist(),
        "test_idx":  np.sort(idx_test).tolist(),
    }
    with open(split_file, "w") as f:
        json.dump(splits_dict, f, indent=2)

    print(f"  Splits criados: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    return {
        "train_idx": np.asarray(splits_dict["train_idx"], dtype=np.int64),
        "val_idx":   np.asarray(splits_dict["val_idx"], dtype=np.int64),
        "test_idx":  np.asarray(splits_dict["test_idx"], dtype=np.int64),
    }


# 5) TREINO — BASELINE E AF-KD  (funciona para qualquer arquitetura student)

def _pos_weight(y: np.ndarray) -> torch.Tensor:
    pos = float(np.sum(y > 0.5))
    neg = float(y.size - pos)
    return torch.tensor(max(1.0, neg / max(pos, 1.0)), dtype=torch.float32)


def _loss_objects(yfr, yep, device):
    pw_fr = _pos_weight(yfr).to(device)
    pw_ep = _pos_weight(yep).to(device)
    return (
        nn.BCEWithLogitsLoss(pos_weight=pw_fr, reduction="none"),
        nn.BCEWithLogitsLoss(pos_weight=pw_ep, reduction="mean"),
    )


def _squeeze(t: torch.Tensor) -> torch.Tensor:
    return t.squeeze(-1) if t.dim() > 1 else t


def train_baseline(
    model: nn.Module,
    X: np.ndarray,
    yfr_main: np.ndarray,
    yep: np.ndarray,
    device: torch.device,
    epochs: int = 30,
    lr: float = 5e-4,
    yfr_aux: Optional[np.ndarray] = None,
    aux_weight: float = 0.10,
) -> nn.Module:
    """
    Treino supervisionado padrão (sem KD).
    Funciona para BiLSTM teacher, LSTMStudent, TransformerStudent e
    TransformerTeacher — todos compartilham a mesma interface (x) -> (fr_logits, ep_logits).
    """
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    bce_fr, bce_ep = _loss_objects(yfr_main, yep, device)
    Xt     = torch.tensor(X,        device=device, dtype=torch.float32)
    yfr_t  = torch.tensor(yfr_main, device=device, dtype=torch.float32)
    yep_t  = torch.tensor(yep,      device=device, dtype=torch.float32).unsqueeze(1)

    bce_aux, yfr_aux_t = None, None
    if yfr_aux is not None:
        bce_aux, _  = _loss_objects(yfr_aux, yep, device)
        yfr_aux_t   = torch.tensor(yfr_aux, device=device, dtype=torch.float32)

    for e in range(epochs):
        opt.zero_grad()
        fr_log, ep_log = model(Xt)
        fr_log, ep_log = _squeeze(fr_log), _squeeze(ep_log)

        loss = (
            bce_fr(fr_log, yfr_t).mean()
            + (bce_aux(fr_log, yfr_aux_t).mean() * aux_weight if bce_aux else 0.0)
            + bce_ep(ep_log.unsqueeze(1), yep_t) * 0.10
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if (e + 1) % 10 == 0:
            print(f"    baseline epoch {e+1:3d}/{epochs}  loss={loss.item():.4f}")

    return model


def train_afkd(
    student: nn.Module,
    teacher: nn.Module,
    X: np.ndarray,
    yfr_main: np.ndarray,
    yep: np.ndarray,
    device: torch.device,
    epochs: int = 50,
    lr: float = 5e-4,
    beta_max: float = 0.35,
    temp: float = 2.5,
    yfr_soft: Optional[np.ndarray] = None,
    yfr_aux: Optional[np.ndarray] = None,
    early_onset_thr: float = 0.12,
    aux_now_weight: float = 0.05,
) -> nn.Module:
    """
    AF-KD forecast-first — idêntico para qualquer par (teacher, student).
    O teacher pode ser BiLSTM (TeacherModel) ou Transformer (TransformerTeacher);
    o student pode ser LSTMStudent ou TransformerStudent.
    Todos os hiperparâmetros fixados aqui são os mesmos para todos os experimentos.
    """
    student.to(device).train()
    teacher.to(device).eval()
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)

    bce_fr, bce_ep = _loss_objects(yfr_main, yep, device)
    Xt      = torch.tensor(X,        device=device, dtype=torch.float32)
    yfr_t   = torch.tensor(yfr_main, device=device, dtype=torch.float32)
    yep_t   = torch.tensor(yep,      device=device, dtype=torch.float32).unsqueeze(1)

    yfr_soft_t = torch.tensor(
        yfr_main.astype(np.float32) if yfr_soft is None else yfr_soft.astype(np.float32),
        device=device, dtype=torch.float32,
    )

    bce_aux, yfr_aux_t = None, None
    if yfr_aux is not None:
        bce_aux, _  = _loss_objects(yfr_aux, yep, device)
        yfr_aux_t   = torch.tensor(yfr_aux, device=device, dtype=torch.float32)

    # Hiperparâmetros AF-KD (FIX-17b)
    t_len          = X.shape[1]
    idx            = torch.arange(t_len, device=device, dtype=torch.float32)
    K              = 45
    p_min_start    = 0.50
    p_min_end      = 0.86
    delta_start    = 0.12
    delta_end      = 3.20
    warmup_epochs  = 10   # v4: 8->10  (60 épocas totais, proporções mantidas)
    rampup_epochs  = 17   # v4: 14->17 (refine = 60-10-17 = 33)

    for e in range(epochs):
        # Scheduler de fases
        if e + 1 <= warmup_epochs:
            stage = "WARMUP"
            frac  = (e + 1) / max(1, warmup_epochs)
            alpha = 1.00
            beta  = 0.00
            delta = delta_start + (0.25 - delta_start) * frac
        elif e + 1 <= warmup_epochs + rampup_epochs:
            stage = "RAMPUP"
            frac  = (e + 1 - warmup_epochs) / max(1, rampup_epochs)
            alpha = 0.80 - (0.80 - 0.58) * frac
            beta  = 0.20 * frac
            delta = 0.25 + (1.29 - 0.25) * frac
        else:
            stage = "REFINE"
            frac  = (e + 1 - warmup_epochs - rampup_epochs) / max(
                1, epochs - warmup_epochs - rampup_epochs
            )
            alpha = 0.58 - (0.58 - 0.40) * frac
            beta  = 0.20
            delta = 1.29 + (delta_end - 1.29) * frac

        opt.zero_grad()

        with torch.no_grad():
            t_fr_log, _ = teacher(Xt)
            t_fr_log    = _squeeze(t_fr_log)

        s_fr_log, s_ep_log = student(Xt)
        s_fr_log, s_ep_log = _squeeze(s_fr_log), _squeeze(s_ep_log)

        # Perdas componentes
        l_hard = bce_fr(s_fr_log, yfr_t).mean()
        l_aux  = (
            bce_aux(s_fr_log, yfr_aux_t).mean() * aux_now_weight
            if bce_aux else torch.tensor(0.0, device=device)
        )
        l_soft = F.binary_cross_entropy_with_logits(
            s_fr_log / temp, torch.sigmoid(t_fr_log / temp)
        )
        l_ep = bce_ep(s_ep_log.unsqueeze(1), yep_t)

        # Early-onset loss (temporal)
        onset_idx   = (yfr_soft_t >= early_onset_thr).float().argmax(dim=1)
        onset_valid = (yfr_soft_t.max(dim=1).values >= early_onset_thr).float()
        rel  = idx.unsqueeze(0) - onset_idx.unsqueeze(1).float()
        win  = ((rel >= 0.0) & (rel < float(K))).float() * onset_valid.unsqueeze(1)
        p    = torch.sigmoid(s_fr_log)
        prog = torch.clamp(rel / max(1.0, float(K - 1)), 0.0, 1.0)

        l_track = F.mse_loss(p, yfr_soft_t)
        l_floor = (F.relu((p_min_start + (p_min_end - p_min_start) * prog) - p) * win
                   ).sum() / (win.sum() + 1e-6)
        l_mono  = F.relu(
            (yfr_soft_t[:, 1:] - yfr_soft_t[:, :-1] > 0).float() * 0.01
            - (p[:, 1:] - p[:, :-1])
        ).mean()
        l_push  = (F.relu(torch.clamp(yfr_soft_t + 0.15, max=0.82) - p) * win
                   ).sum() / (win.sum() + 1e-6)
        l_early = l_track + l_floor + 0.5 * l_mono + l_push

        loss = alpha * l_hard + l_aux + beta * l_soft + 0.10 * l_ep + delta * l_early
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        opt.step()

        if (e + 1) % 5 == 0:
            print(
                f"    afkd epoch {e+1:3d}/{epochs} [{stage:6s}]  "
                f"loss={loss.item():.4f} "
                f"(hard={l_hard.item():.3f} soft={l_soft.item():.3f} "
                f"early={l_early.item():.3f}) "
                f"[α={alpha:.2f} β={beta:.2f} δ={delta:.2f}]"
            )

    return student


# 5b) TEMPERATURE SCALING — calibração pós-treino (v4)

class TScaledModel(nn.Module):
    """
    Wrapper leve que divide os frame-logits por um escalar T aprendido.
    Transparente para o restante do pipeline: mantém a mesma interface
    (x) -> (fr_logits / T, ep_logits) sem alterar os pesos do modelo base.
    """
    def __init__(self, base: nn.Module, temperature: float):
        super().__init__()
        self.base = base
        self.T = float(temperature)

    def forward(self, x: torch.Tensor):
        fr_log, ep_log = self.base(x)
        return fr_log / self.T, ep_log


def temperature_scale(
    model: nn.Module,
    X_val: np.ndarray,
    yfr_val: np.ndarray,
    device: torch.device,
    n_steps: int = 20,
    lr: float = 0.01,
    t_init: float = 1.5,
    t_min: float = 0.5,
    t_max: float = 5.0,
) -> TScaledModel:
    """
    Aprende temperatura T que minimiza NLL dos frame-logits no conjunto de
    validação (20 passos Adam). Retorna TScaledModel pronto para inferência.

    Protocolo v4:
      • T inicializado em 1.5 (ligeiramente suavizado)
      • clamp em [0.5, 5.0] para evitar divergência
      • logits congelados (model.eval() + torch.no_grad()) durante a busca
    """
    model.to(device).eval()
    T = nn.Parameter(torch.tensor(t_init, device=device, dtype=torch.float32))
    opt = torch.optim.Adam([T], lr=lr)

    Xv = torch.tensor(X_val,  device=device, dtype=torch.float32)
    yv = torch.tensor(yfr_val, device=device, dtype=torch.float32)

    # Logits fixos — só T é otimizado
    with torch.no_grad():
        fr_log, _ = model(Xv)
        fr_log = fr_log.squeeze(-1) if fr_log.dim() > 1 else fr_log

    for step in range(n_steps):
        opt.zero_grad()
        T_clamped = T.clamp(t_min, t_max)
        loss = F.binary_cross_entropy_with_logits(fr_log / T_clamped, yv)
        loss.backward()
        opt.step()

    T_final = float(T.clamp(t_min, t_max).detach().cpu())
    print(f"    temperature_scale: T={T_final:.4f}  ({n_steps} passos, NLL final={loss.item():.4f})")
    return TScaledModel(model, T_final)

def infer_stream_fixed(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 1,
    warmup_steps: int = LAT_WARMUP_STEPS,
) -> InferResult:
    """
    Inferência sample-a-sample com medição de latência.
    Protocolo idêntico para todos os modelos.
    """
    model.to(device).eval()
    N, T_len, _ = X.shape
    frame_probs   = np.zeros((N, T_len), dtype=np.float32)
    episode_probs = np.zeros(N, dtype=np.float32)
    latencies: List[float] = []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            Xt = torch.tensor(X[i:i+batch_size], device=device, dtype=torch.float32)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            fr_log, ep_log = model(Xt)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            if i >= warmup_steps:
                latencies.append((t1 - t0) * 1000.0)

            fr_p = torch.sigmoid(_squeeze(fr_log)).cpu().numpy()
            ep_p = torch.sigmoid(_squeeze(ep_log)).cpu().numpy()
            frame_probs[i:i+batch_size]   = fr_p
            episode_probs[i:i+batch_size] = ep_p

    return InferResult(
        frame_probs=frame_probs,
        episode_probs=episode_probs,
        episode_pred=(episode_probs > 0.5).astype(int),
        latencies_ms=latencies,
        total_time=sum(latencies) / 1000.0 if latencies else 0.0,
    )


def aggregate_probs_k6(probs: np.ndarray, k: int = K_AGG) -> np.ndarray:
    """Agregação causal por janela deslizante de tamanho k (K_AGG=6)."""
    out = np.zeros_like(probs)
    for i in range(probs.shape[0]):
        for t in range(probs.shape[1]):
            out[i, t] = probs[i, max(0, t - k + 1):t + 1].mean()
    return out


def episode_probs_from_frames(frame_probs: np.ndarray, k: int = K_AGG) -> np.ndarray:
    return frame_probs[:, -k:].mean(axis=1)

def fail_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    pos = (y_true == 1)
    p = int(pos.sum())
    if p == 0:
        return 0.0
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return float(fn / p)

def binary_entropy(p: float) -> float:
    p = float(np.clip(p, 1e-7, 1 - 1e-7))
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

def smooth_series(x: np.ndarray, w: int = 3) -> np.ndarray:
    w = max(1, int(w))
    if w <= 1:
        return np.asarray(x, dtype=np.float32)
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(np.asarray(x, dtype=np.float32), kernel, mode="same")

def first_derivative_detection(probs: np.ndarray, onset: int, delta: float, prob_floor: float, m: int = TTD_M, smooth_w: int = 3) -> Optional[int]:
    s = smooth_series(probs, smooth_w)
    if onset >= len(s):
        return None
    d = np.diff(s, prepend=s[0])
    count = 0
    start = None
    for t in range(max(1, onset), len(s)):
        cond = (s[t] >= prob_floor) and (d[t] >= delta)
        if cond:
            count += 1
            if start is None:
                start = t
            if count >= m:
                return start
        else:
            count = 0
            start = None
    return None

def compute_ttd_derivative(y_true_fr: np.ndarray, y_pred_probs: np.ndarray, window: int = T, delta: float = 0.006, prob_floor: float = 0.18, m: int = TTD_M, smooth_w: int = 3) -> float:
    dt = 10.0 / window
    ttds = []
    for yt, yp in zip(y_true_fr, y_pred_probs):
        idx = np.where(yt > 0)[0]
        if len(idx) == 0:
            continue
        t0 = int(idx[0])
        det = first_derivative_detection(yp, t0, delta=delta, prob_floor=prob_floor, m=m, smooth_w=smooth_w)
        delay = (det - t0) if det is not None else (window - t0)
        ttds.append(delay * dt)
    return float(np.mean(ttds)) if ttds else float("nan")

def compute_ttd_subset(y_true_fr: np.ndarray, y_pred_probs: np.ndarray, episode_mask: np.ndarray, window: int = T, thr: float = 0.25, m: int = TTD_M) -> float:
    mask = np.asarray(episode_mask).astype(bool)
    if mask.sum() == 0:
        return float("nan")
    dt = 10.0 / window
    ttds = []
    for yt, yp in zip(y_true_fr[mask], y_pred_probs[mask]):
        idx = np.where(yt > 0)[0]
        if len(idx) == 0:
            continue
        t0 = int(idx[0])
        det = _first_stable_m(yp, t0, thr=thr, m=m)
        delay = (det - t0) if det is not None else (window - t0)
        ttds.append(delay * dt)
    return float(np.mean(ttds)) if ttds else float("nan")

def compute_ttd_derivative_subset(y_true_fr: np.ndarray, y_pred_probs: np.ndarray, episode_mask: np.ndarray, window: int = T, delta: float = 0.006, prob_floor: float = 0.18, m: int = TTD_M, smooth_w: int = 3) -> float:
    mask = np.asarray(episode_mask).astype(bool)
    if mask.sum() == 0:
        return float("nan")
    return compute_ttd_derivative(y_true_fr[mask], y_pred_probs[mask], window=window, delta=delta, prob_floor=prob_floor, m=m, smooth_w=smooth_w)

def fail_rate_positive_subset(y_true_ep: np.ndarray, y_pred_ep: np.ndarray, episode_mask: np.ndarray) -> float:
    mask = np.asarray(episode_mask).astype(bool)
    if mask.sum() == 0:
        return float("nan")
    y_true_s = np.asarray(y_true_ep[mask]).astype(int)
    y_pred_s = np.asarray(y_pred_ep[mask]).astype(int)
    pos = (y_true_s == 1)
    p = int(pos.sum())
    if p == 0:
        return float("nan")
    fn = int(((y_true_s == 1) & (y_pred_s == 0)).sum())
    return float(fn / p)

def _ttd_threshold_stats(y_true_fr: np.ndarray, frame_probs: np.ndarray, theta: float, m_detect: int = TTD_M) -> Tuple[float, float]:
    dt = 10.0 / T
    ttds=[]; misses=0; positives=0
    for yt, yp in zip(y_true_fr, frame_probs):
        idx=np.where(yt>0)[0]
        if len(idx)==0: continue
        positives += 1
        t0=int(idx[0])
        det=_first_stable_m(yp,t0,float(theta),m_detect)
        if det is None:
            misses += 1
            ttds.append((T-t0)*dt)
        else:
            ttds.append((det-t0)*dt)
    miss_rate=(misses/positives) if positives>0 else 0.0
    return (float(np.mean(ttds)) if ttds else float("nan"), float(miss_rate))

def select_theta_ttd(y_true_fr: np.ndarray, frame_probs: np.ndarray, m_detect: int = TTD_M) -> float:
    MISS_CAP=0.75
    positive_ttd=[]; zero_ttd=[]; fallback=[]
    for theta in THETA_GLOBAL_GRID:
        ttd_val, miss_rate = _ttd_threshold_stats(y_true_fr, frame_probs, float(theta), m_detect)
        ttd_key = ttd_val if not math.isnan(ttd_val) else 1e9
        prox = abs(float(theta)-0.30)
        fallback.append((miss_rate, ttd_key, prox, float(theta)))
        if miss_rate <= MISS_CAP:
            item=(ttd_key, miss_rate, prox, float(theta))
            if ttd_key > 1e-8:
                positive_ttd.append(item)
            else:
                zero_ttd.append(item)
    if positive_ttd:
        positive_ttd.sort(); return float(positive_ttd[0][3])
    if zero_ttd:
        zero_ttd.sort(); return float(zero_ttd[0][3])
    fallback.sort(); return float(fallback[0][3]) if fallback else 0.5

def select_theta_per_model(y_true_fr: np.ndarray, frame_probs: np.ndarray, m_detect: int = TTD_M) -> float:
    """Calibra θ_TTD por modelo: minimiza TTD com FailRate ≤ FAIL_BUDGET.

    Diferente de select_theta_ttd (MISS_CAP=0.75, usa probs do LSTM-Baseline),
    esta função usa as probabilidades do PRÓPRIO modelo e impõe o mesmo
    orçamento de segurança do select_thr_ep (FAIL_BUDGET=0.05).

    Prioridade de seleção:
      (1) Entre candidatos com FailRate ≤ FAIL_BUDGET -> min TTD;
          se TTD empata (ambos 0 ou nan) -> θ mais próximo de 0.30.
      (2) Fallback (nenhum θ atinge budget): menor FailRate possível,
          desempate por menor TTD, depois por proximidade a 0.30.
    """
    feasible: list = []   # (ttd_key, miss_rate, prox, theta)
    fallback:  list = []  # (miss_rate, ttd_key, prox, theta)
    for theta in THETA_GLOBAL_GRID:
        ttd_val, miss_rate = _ttd_threshold_stats(y_true_fr, frame_probs, float(theta), m_detect)
        ttd_key = ttd_val if not math.isnan(ttd_val) else 1e9
        prox = abs(float(theta) - 0.30)
        fallback.append((miss_rate, ttd_key, prox, float(theta)))
        if miss_rate <= FAIL_BUDGET:
            feasible.append((ttd_key, miss_rate, prox, float(theta)))
    if feasible:
        feasible.sort()          # (1) min TTD, (2) min FR, (3) prox a 0.30
        return float(feasible[0][3])
    fallback.sort()              # fallback: (1) min FR, (2) min TTD, (3) prox
    return float(fallback[0][3]) if fallback else 0.5


def select_theta_ttd_adapt(y_true_fr: np.ndarray, frame_probs: np.ndarray, m_detect: int = TTD_M) -> float:
    feasible=[]; fallback=[]
    for theta in THETA_TTD_ADAPT_GRID:
        ttd_val, miss_rate = _ttd_threshold_stats(y_true_fr, frame_probs, float(theta), m_detect)
        ttd_key = ttd_val if not math.isnan(ttd_val) else 1e9
        item=(miss_rate, ttd_key, abs(float(theta)-TTD_ADAPT_TARGET_THETA), float(theta))
        fallback.append(item)
        if miss_rate <= TTD_ADAPT_MAX_MISS:
            feasible.append((ttd_key, abs(float(theta)-TTD_ADAPT_TARGET_THETA), float(theta)))
    if feasible:
        feasible.sort(); return float(feasible[0][2])
    fallback.sort(); return float(fallback[0][3]) if fallback else 0.25

def _ttd_derivative_stats(y_true_fr: np.ndarray, frame_probs: np.ndarray, delta: float, prob_floor: float, m_detect: int = TTD_M, smooth_w: int = 3) -> Tuple[float, float]:
    ttd_val = compute_ttd_derivative(y_true_fr, frame_probs, delta=delta, prob_floor=prob_floor, m=m_detect, smooth_w=smooth_w)
    misses=0; positives=0
    for yt, yp in zip(y_true_fr, frame_probs):
        idx=np.where(yt>0)[0]
        if len(idx)==0: continue
        positives += 1
        if first_derivative_detection(yp, int(idx[0]), delta=delta, prob_floor=prob_floor, m=m_detect, smooth_w=smooth_w) is None:
            misses += 1
    miss_rate=(misses/positives) if positives>0 else 0.0
    return float(ttd_val), float(miss_rate)

def select_derivative_ttd_params(y_true_fr: np.ndarray, frame_probs: np.ndarray, m_detect: int = TTD_M) -> Tuple[float, float, int]:
    feasible=[]; fallback=[]
    for smooth_w in DERIV_SMOOTH_GRID:
        for delta in DERIV_DELTA_GRID:
            for prob_floor in DERIV_PROB_GRID:
                ttd_val, miss_rate = _ttd_derivative_stats(y_true_fr, frame_probs, float(delta), float(prob_floor), m_detect, int(smooth_w))
                ttd_key = ttd_val if not math.isnan(ttd_val) else 1e9
                prox = abs(float(delta)-DERIV_TARGET_DELTA)+abs(float(prob_floor)-DERIV_TARGET_PROB)
                item=(miss_rate, ttd_key, prox, int(smooth_w), float(delta), float(prob_floor))
                fallback.append(item)
                if miss_rate <= DERIV_MAX_MISS:
                    feasible.append((ttd_key, prox, int(smooth_w), float(delta), float(prob_floor)))
    if feasible:
        feasible.sort(); _,_,sw,d,p = feasible[0]; return float(d), float(p), int(sw)
    fallback.sort(); _,_,_,sw,d,p = fallback[0]; return float(d), float(p), int(sw)

def select_thr_ep(frame_probs: np.ndarray, y_true_ep: np.ndarray) -> float:
    ep_probs = episode_probs_from_frames(frame_probs, k=K_AGG)
    feasible=[]; fallback=[]
    for thr in THR_EP_GRID:
        pred=(ep_probs>=float(thr)).astype(np.int64)
        f1=f1_score(y_true_ep,pred,zero_division=0)
        fr=fail_rate(y_true_ep,pred)
        fallback.append((fr,-f1,float(thr)))
        if fr <= FAIL_BUDGET:
            feasible.append((-f1,fr,float(thr)))
    if feasible:
        feasible.sort(); return float(feasible[0][2])
    fallback.sort(); return float(fallback[0][2]) if fallback else 0.5

@dataclass
class StreamEval:
    frame_probs: np.ndarray
    episode_probs: np.ndarray
    lat_ms: float
    skip_pct: float
    cost_ms_per_frame: float

def infer_stream_fixed_eval(model: nn.Module, X: np.ndarray, device: torch.device) -> StreamEval:
    res = infer_stream_fixed(model, X, device)
    lat = float(np.mean(res.latencies_ms)) if res.latencies_ms else 0.0
    return StreamEval(frame_probs=res.frame_probs, episode_probs=episode_probs_from_frames(res.frame_probs), lat_ms=lat, skip_pct=0.0, cost_ms_per_frame=lat)

def infer_stream_hybrid_from_raw(frame_probs_raw: np.ndarray, X: np.ndarray, avg_lat_ms: float, tau_delta: float, tau_h: float) -> StreamEval:
    n,t_len,_ = X.shape
    out = np.zeros_like(frame_probs_raw, dtype=np.float32)
    skipped=0
    for i in range(n):
        last_p=0.0
        for t in range(t_len):
            dk = np.abs(X[i,t,KIN]-X[i,t-1,KIN]).max() if t>0 else 1.0
            if (dk < tau_delta) and (binary_entropy(last_p) < tau_h):
                out[i,t]=last_p; skipped += 1
            else:
                last_p=float(frame_probs_raw[i,t]); out[i,t]=last_p
    skip_pct=(skipped/float(n*t_len))*100.0
    return StreamEval(frame_probs=out, episode_probs=episode_probs_from_frames(out), lat_ms=float(avg_lat_ms), skip_pct=skip_pct, cost_ms_per_frame=float(avg_lat_ms)*(1.0-skip_pct/100.0))

def _compute_ttd_bruto(
    frame_probs: np.ndarray,
    y_fr: np.ndarray,
    theta: float,
    m: int,
    window: int = T,
) -> float:
    """
    TTD médio calculado APENAS sobre episódios detectados (sem penalidade de miss).

    aviso: ATENÇÃO — uso restrito:
       • Usar SOMENTE como métrica auxiliar / análise de distribuição
       • NÃO usar como métrica principal no paper (exclui misses -> otimista)
       • A métrica justa para o paper é TTD_efetivo (= campo 'TTD' em summarize_eval)

    Equivalente a: penalize=False em _ttd_vec()
    """
    dt   = 10.0 / window
    ttds = []
    for yp, yt in zip(frame_probs, y_fr):
        onset = np.where(yt > 0)[0]
        if len(onset) == 0:
            continue
        t0  = int(onset[0])
        det = _first_stable_m(yp, t0, theta, m)
        if det is not None:                    # <- só episódios detectados
            ttds.append((det - t0) * dt)
    return float(np.mean(ttds)) if ttds else float("nan")


def summarize_eval(se: StreamEval, y_ep: np.ndarray, y_fr: np.ndarray, thr_ep: float, theta_ttd: float, m_detect: int = TTD_M, progressive_mask: Optional[np.ndarray] = None, abrupt_mask: Optional[np.ndarray] = None, theta_ttd_adapt: Optional[float] = None, deriv_delta: Optional[float] = None, deriv_prob_floor: Optional[float] = None, deriv_smooth_w: Optional[int] = None) -> dict:
    ep_hat=(se.episode_probs>=thr_ep).astype(np.int64)
    f1=f1_score(y_ep,ep_hat,zero_division=0)
    fr=fail_rate(y_ep,ep_hat)
    meta_tmp = pd.DataFrame({"progressive": np.zeros(len(y_fr), dtype=bool)})
    # TTD efetivo (= métrica penalizada): misses recebem (window-t0)*dt
    ttd=evaluate_streaming_policy(se.frame_probs,y_fr,meta_tmp,theta=theta_ttd,m=m_detect)["TTD"]
    # TTD bruto (análise auxiliar): somente episódios detectados — NÃO usar no paper sozinho
    ttd_bruto=_compute_ttd_bruto(se.frame_probs,y_fr,theta=theta_ttd,m=m_detect)
    # [v8] TTDef alinhado com run_v29: TTD_bruto + FR × T_EP
    # Fórmula idêntica à das macros LaTeX do artigo (resBaseFixoTTDef, etc.)
    ttdef = round(ttd_bruto + fr * T_EP, 4) if not math.isnan(ttd_bruto) else float("nan")
    ece=expected_calibration_error(y_ep.astype(float),se.episode_probs)
    out={
        "F1":               round(f1,4),
        "ECE":              round(ece,4),
        "FailRate":         round(fr,4),
        # TTD = efetivo penalizado -> retrocompatibilidade com análises internas
        "TTD":              round(ttd,4)       if not math.isnan(ttd)       else ttd,
        # TTDef = alinhado run_v29: TTD_bruto + FR × T_EP -> métrica do paper
        "TTDef":            ttdef,
        # TTD_bruto = sem penalidade -> uso auxiliar somente
        "TTD_bruto":        round(ttd_bruto,4) if not math.isnan(ttd_bruto) else ttd_bruto,
        "ThetaTTD":         round(theta_ttd,4),
        "ThrEP":            round(thr_ep,4),
        "Lat_ms":           round(se.lat_ms,4),
        "SkipPct":          round(se.skip_pct,2),
        "Cost_ms_per_frame":round(se.cost_ms_per_frame,4),
    }
    if theta_ttd_adapt is not None:
        ttd_adapt=evaluate_streaming_policy(se.frame_probs,y_fr,meta_tmp,theta=theta_ttd_adapt,m=m_detect)["TTD"]
        out.update({"ThetaTTD_Adapt":round(theta_ttd_adapt,4),"TTD_Adapt":round(ttd_adapt,4) if not math.isnan(ttd_adapt) else ttd_adapt})
    if deriv_delta is not None and deriv_prob_floor is not None:
        ttd_deriv=compute_ttd_derivative(y_fr,se.frame_probs,delta=deriv_delta,prob_floor=deriv_prob_floor,m=m_detect,smooth_w=(deriv_smooth_w or 3))
        out.update({"DerivDelta":round(deriv_delta,4),"DerivProbFloor":round(deriv_prob_floor,4),"DerivSmoothW":int(deriv_smooth_w or 3),"TTD_Deriv":round(ttd_deriv,4) if not math.isnan(ttd_deriv) else ttd_deriv})
    if progressive_mask is not None:
        prog_mask=np.asarray(progressive_mask).astype(bool)
        ttd_prog=compute_ttd_subset(y_fr,se.frame_probs,prog_mask,thr=theta_ttd,m=m_detect)
        fr_prog=fail_rate_positive_subset(y_ep,ep_hat,prog_mask)
        out.update({"N_Progressive":int(prog_mask.sum()),"TTD_Progressive":round(ttd_prog,4) if not math.isnan(ttd_prog) else ttd_prog,"FailRate_Progressive":round(fr_prog,4) if not math.isnan(fr_prog) else fr_prog})
        if theta_ttd_adapt is not None:
            ttd_prog_adapt=compute_ttd_subset(y_fr,se.frame_probs,prog_mask,thr=theta_ttd_adapt,m=m_detect)
            out["TTD_Progressive_Adapt"]=round(ttd_prog_adapt,4) if not math.isnan(ttd_prog_adapt) else ttd_prog_adapt
        if deriv_delta is not None and deriv_prob_floor is not None:
            ttd_prog_deriv=compute_ttd_derivative_subset(y_fr,se.frame_probs,prog_mask,delta=deriv_delta,prob_floor=deriv_prob_floor,m=m_detect,smooth_w=(deriv_smooth_w or 3))
            out["TTD_Progressive_Deriv"]=round(ttd_prog_deriv,4) if not math.isnan(ttd_prog_deriv) else ttd_prog_deriv
    if abrupt_mask is not None:
        abr_mask=np.asarray(abrupt_mask).astype(bool)
        ttd_abr=compute_ttd_subset(y_fr,se.frame_probs,abr_mask,thr=theta_ttd,m=m_detect)
        fr_abr=fail_rate_positive_subset(y_ep,ep_hat,abr_mask)
        out.update({"N_Abrupt":int(abr_mask.sum()),"TTD_Abrupt":round(ttd_abr,4) if not math.isnan(ttd_abr) else ttd_abr,"FailRate_Abrupt":round(fr_abr,4) if not math.isnan(fr_abr) else fr_abr})
        if theta_ttd_adapt is not None:
            ttd_abr_adapt=compute_ttd_subset(y_fr,se.frame_probs,abr_mask,thr=theta_ttd_adapt,m=m_detect)
            out["TTD_Abrupt_Adapt"]=round(ttd_abr_adapt,4) if not math.isnan(ttd_abr_adapt) else ttd_abr_adapt
        if deriv_delta is not None and deriv_prob_floor is not None:
            ttd_abr_deriv=compute_ttd_derivative_subset(y_fr,se.frame_probs,abr_mask,delta=deriv_delta,prob_floor=deriv_prob_floor,m=m_detect,smooth_w=(deriv_smooth_w or 3))
            out["TTD_Abrupt_Deriv"]=round(ttd_abr_deriv,4) if not math.isnan(ttd_abr_deriv) else ttd_abr_deriv
    return out

def select_hybrid_policy_from_raw(frame_probs_raw: np.ndarray, Xv: np.ndarray, yepv: np.ndarray, yfrv: np.ndarray, avg_lat_ms: float, thr_ep: float, theta_ttd: float, ref_fixed: dict, progressive_mask: Optional[np.ndarray] = None, abrupt_mask: Optional[np.ndarray] = None, m_detect: int = TTD_M) -> Optional[Tuple[float, float]]:
    deltas = np.abs(np.diff(Xv[:,:,KIN], axis=1)).max(axis=2).reshape(-1).astype(np.float32)
    min_skip,max_skip=40.0,85.0
    f1_floor=0.90*ref_fixed["F1"] if ref_fixed["F1"]>0 else 0.0
    fail_limit=max(FAIL_BUDGET, ref_fixed["FailRate"]+0.02)
    best=None; best_params=None
    for p in TAU_DELTA_PERCENTILES:
        tau_delta=float(np.percentile(deltas,p))
        for tau_h in TAU_H_GRID:
            se=infer_stream_hybrid_from_raw(frame_probs_raw,Xv,avg_lat_ms,tau_delta=tau_delta,tau_h=float(tau_h))
            r=summarize_eval(se,yepv,yfrv,thr_ep,theta_ttd,m_detect,progressive_mask=progressive_mask,abrupt_mask=abrupt_mask)
            if r["FailRate"]>fail_limit: continue
            if not (math.isnan(r["TTD"]) or math.isnan(ref_fixed["TTD"])):
                if r["TTD"] > ref_fixed["TTD"] * TTD_BUDGET_RATIO: continue
            if r["F1"] < f1_floor: continue
            if not (min_skip <= r["SkipPct"] <= max_skip): continue
            cand=(r["Cost_ms_per_frame"], -r["F1"], r["TTD"], -r["SkipPct"], tau_delta, float(tau_h))
            if best is None or cand < best:
                best=cand; best_params=(tau_delta,float(tau_h))
    return best_params


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    pf = probs.flatten()
    tf = targets.flatten()
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1],
                      np.linspace(0, 1, n_bins + 1)[1:]):
        mask = (pf >= lo) & (pf < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(pf) * abs(tf[mask].mean() - pf[mask].mean())
    return float(ece)

# Compatibilidade de nomenclatura com run_v22 e helpers da v4
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    return compute_ece(y_prob, y_true, n_bins=n_bins)


def _first_stable_m(probs: np.ndarray, onset: int, thr: float, m: int) -> Optional[int]:
    """Primeiro frame de m consecutivos ≥ thr a partir de onset."""
    count = 0
    for t in range(onset, len(probs)):
        if probs[t] >= thr:
            count += 1
            if count >= m:
                return t - m + 1
        else:
            count = 0
    return None


def evaluate_streaming_policy(
    frame_probs: np.ndarray,
    yfr: np.ndarray,
    meta_rows: pd.DataFrame,
    theta: float,
    m: int = TTD_M,
    window: int = T,
) -> Dict[str, float]:
    """
    Avalia política de detecção binária com θ e m consecutivos.
    Retorna: F1, FailRate, TTD (penalizado = efetivo), TTD_Progressive, TTD_Abrupt.

    PROTOCOLO JUSTO (alinhado ao paper):
      • TTD penalizado: episódio não detectado recebe (window - t0) * dt
        -> nenhum episódio é descartado do cálculo
      • FailRate = misses / positivos  (FN / P)
        -> divide SOMENTE pelos episódios que tinham onset (positivos críticos)
        -> NÃO divide pelo total de episódios (que inclui negativos)
    """
    dt       = 10.0 / window   # segundos por frame
    ttds, ttds_prog, ttds_abr = [], [], []
    misses   = 0
    positives = 0   # <- contador de episódios com onset (críticos)

    for i in range(len(frame_probs)):
        onset_frames = np.where(yfr[i] > 0)[0]
        if len(onset_frames) == 0:
            continue                  # episódio negativo — não conta para FailRate
        positives += 1                # <- incrementa antes de qualquer filtro
        t0  = int(onset_frames[0])
        det = _first_stable_m(frame_probs[i], t0, theta, m)

        penalty = (window - t0) * dt  # penalidade para miss

        if det is not None:
            ttd = (det - t0) * dt
        else:
            misses += 1
            ttd = penalty             # <- miss NÃO descarta o episódio

        ttds.append(ttd)              # <- todos os episódios positivos incluídos
        if "progressive" in meta_rows.columns:
            is_prog = bool(meta_rows.iloc[i]["progressive"])
            (ttds_prog if is_prog else ttds_abr).append(ttd)

    y_true = (yfr.max(axis=1) > 0).astype(int)
    y_pred = ((frame_probs >= theta).sum(axis=1) >= m).astype(int)

    return {
        "F1":              float(f1_score(y_true, y_pred, zero_division=0)),
        # CORREÇÃO: misses / positives  (FN/P) — não mais misses / len(frame_probs)
        "FailRate":        float(misses / max(positives, 1)),
        "TTD":             float(np.mean(ttds))      if ttds      else float("nan"),
        "TTD_Progressive": float(np.mean(ttds_prog)) if ttds_prog else float("nan"),
        "TTD_Abrupt":      float(np.mean(ttds_abr))  if ttds_abr  else float("nan"),
    }


def select_theta_global_on_validation(
    probs_val: np.ndarray,
    yfr_val: np.ndarray,
    meta_val: pd.DataFrame,
    m: int = TTD_M,
    failrate_max: float = THETA_GLOBAL_FAILRATE_MAX,
    theta_grid: Optional[List[float]] = None,
) -> Tuple[float, List[Dict]]:
    """
    Seleciona θ global via validação (protocolo ético).
    Critério: minimizar TTD_Progressive com FailRate ≤ failrate_max.
    Fallback: melhor θ disponível se nenhum atende ao constraint.
    """
    if theta_grid is None:
        theta_grid = THETA_GLOBAL_GRID

    rows: List[Dict] = []
    for theta in theta_grid:
        m_ = evaluate_streaming_policy(probs_val, yfr_val, meta_val, theta=theta, m=m)
        rows.append({
            "theta":           float(theta),
            "FailRate":        float(m_.get("FailRate", float("nan"))),
            "TTD":             float(m_.get("TTD", float("nan"))),
            "TTD_Progressive": float(m_.get("TTD_Progressive", float("nan"))),
            "TTD_Abrupt":      float(m_.get("TTD_Abrupt", float("nan"))),
        })

    candidates = [r for r in rows if r["FailRate"] <= failrate_max]
    pool = candidates if candidates else rows

    def _key(r):
        nan_to_big = lambda v: v if v == v else 1e9  # noqa: E731
        return (nan_to_big(r["TTD_Progressive"]), nan_to_big(r["TTD"]),
                nan_to_big(r["FailRate"]), r["theta"])

    best = sorted(pool, key=_key)[0]
    return float(best["theta"]), rows


# 7) ORQUESTRAÇÃO — EXECUÇÃO COMPLETA PARA UMA SEED

def run_single_seed_unified(
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Protocolo completo para uma seed:

      Passo 1  – Carrega/gera dataset (gerador v4)
      Passo 2  – Splits estratificados (train/val/test)
      Passo 3  – Treina professor BiLSTM
      Passo 4  – Treina LSTM Baseline
      Passo 5  – Treina LSTM-AF-KD  (BiLSTM -> LSTM)
      Passo 6  – Treina Transformer Baseline
      Passo 7  – Treina Transformer-AF-KD  (BiLSTM -> Transformer)
      Passo 8  – Treina TransformerTeacher  (oráculo full-sequence)
      Passo 9  – Treina LSTM-TransKD  (Transformer -> LSTM)  ★ ABLAÇÃO
      Passo 10 – Inferência na validação -> calibração de θ global
      Passo 11 – Inferência no teste
      Passo 12 – Avalia todos com θ global simétrico
      Passo 13 – Salva frame_probs (para fairness / robustez)
    """
    print(f"\n{'='*80}")
    print(f"SEED {seed}  —  5 MODELOS × PROTOCOLO UNIFICADO")
    print(f"{'='*80}\n")

    set_seed(seed)
    cfg = DatasetConfig(seed=seed)

    # 1. Dataset
    print("1. Carregando/gerando dataset...")
    X, y_frame_clean, y_frame_obs, y_episode, meta_df = load_or_generate_dataset(cfg)

    # IMPORTANTE: o gerador fornece sinais contínuos; o protocolo TTD usa onset
    # detectável com frame_thr (ex.: 0.35), não y>0. Se usar y>0, LOW_RISK_BASE
    # desloca o onset para t=0 e colapsa TTD para zero.
    y_frame_bin = (y_frame_obs >= cfg.frame_thr).astype(np.float32)
    y_episode_bin = (y_frame_bin.max(axis=1) > 0).astype(np.float32)

    # v7: FILTRO BINÁRIO — mantém apenas Normal e Crítico
    # Atenção e Alerta são excluídos do treinamento e avaliação para consistência
    # com o experimento principal (run_v22_v6.py). O FPR é calculado apenas sobre
    # episódios Normal (condução genuinamente segura).
    _class_col_abl = infer_class_column(meta_df)
    _cat_abl = meta_df[_class_col_abl].astype(str).values
    _binary_mask = (_cat_abl == "Normal") | (_cat_abl == "Critico")
    X            = X[_binary_mask]
    y_frame_clean = y_frame_clean[_binary_mask]
    y_frame_obs  = y_frame_obs[_binary_mask]
    y_episode    = y_episode[_binary_mask]
    y_frame_bin  = y_frame_bin[_binary_mask]
    y_episode_bin = y_episode_bin[_binary_mask]
    meta_df      = meta_df[_binary_mask].reset_index(drop=True)
    n_normal  = int((_cat_abl[_binary_mask] == "Normal").sum())
    n_critico = int((_cat_abl[_binary_mask] == "Critico").sum())
    print(f"  [v7-filtro] Normal={n_normal}  Crítico={n_critico}  "
          f"Total={len(meta_df)}  (Atenção+Alerta excluídos)")

    # 2. Splits
    print("2. Criando splits estratificados...")
    splits   = create_stratified_splits(X, y_frame_bin, y_episode_bin, meta_df, cfg)
    idx_tr   = splits["train_idx"]
    idx_val  = splits["val_idx"]
    idx_te   = splits["test_idx"]

    X_tr   = X[idx_tr];    yfr_tr  = y_frame_bin[idx_tr]; yep_tr = y_episode_bin[idx_tr]
    X_val  = X[idx_val];   yfr_val = y_frame_bin[idx_val]
    X_te   = X[idx_te];    yfr_te  = y_frame_bin[idx_te]
    meta_val = meta_df.iloc[idx_val].reset_index(drop=True)
    meta_te  = meta_df.iloc[idx_te ].reset_index(drop=True)
    yep_tr_arr = y_episode_bin[idx_tr]

    yfr_clean_tr = y_frame_clean[idx_tr]   # para yfr_soft no AF-KD

    print(f"  Train={len(idx_tr)}, Val={len(idx_val)}, Test={len(idx_te)}")
    print(f"  Frame bin threshold (frame_thr) = {cfg.frame_thr:.2f}")
    print(f"  Episódios positivos (bin): train={int(yep_tr_arr.sum())}, val={int(y_episode_bin[idx_val].sum())}, test={int(y_episode_bin[idx_te].sum())}")

    # 3. Professor BiLSTM
    print("\n3. Treinando professor BiLSTM (shared teacher)...")
    bilstm_teacher = TeacherModel(input_dim=D, hidden_size=H, num_layers=2, dropout=0.1)
    bilstm_teacher = train_baseline(
        bilstm_teacher, X_tr, yfr_tr, yep_tr_arr, device, epochs=60, lr=5e-4,  # ALINHADO run_v22
    )
    # teacher não recebe temperature scaling (não entra em inferência final)

    # 4. LSTM Baseline
    print("\n4. Treinando LSTM Baseline...")
    lstm_base = LSTMStudent(input_dim=D, hidden_size=H, num_layers=2, dropout=0.1)
    lstm_base = train_baseline(
        lstm_base, X_tr, yfr_tr, yep_tr_arr, device, epochs=60, lr=5e-4,  # ALINHADO run_v22
    )
    print("  -> Temperature scaling LSTM-Baseline (v4)...")
    lstm_base = temperature_scale(lstm_base, X_val, yfr_val, device)

    # 5. LSTM-AF-KD  (BiLSTM -> LSTM)
    print("\n5. Treinando LSTM-AF-KD  [BiLSTM -> LSTM]...")
    lstm_afkd = LSTMStudent(input_dim=D, hidden_size=H, num_layers=2, dropout=0.1)
    lstm_afkd = train_afkd(
        lstm_afkd, bilstm_teacher, X_tr, yfr_tr, yep_tr_arr, device,
        epochs=80, lr=5e-4, beta_max=0.35, temp=2.5, yfr_soft=yfr_clean_tr,  # ALINHADO run_v22
    )
    print("  -> Temperature scaling LSTM-AF-KD (v4)...")
    lstm_afkd = temperature_scale(lstm_afkd, X_val, yfr_val, device)

    # 6. Transformer Baseline
    print("\n6. Treinando Transformer Baseline...")
    trans_base = TransformerStudent(TRANSFORMER_CONFIG)
    trans_base = train_baseline(
        trans_base, X_tr, yfr_tr, yep_tr_arr, device, epochs=60, lr=5e-4,  # ALINHADO run_v22
    )
    print("  -> Temperature scaling Trans-Baseline (v4)...")
    trans_base = temperature_scale(trans_base, X_val, yfr_val, device)

    # 7. Transformer-AF-KD  (BiLSTM -> Transformer)
    print("\n7. Treinando Transformer-AF-KD  [BiLSTM -> Transformer]...")
    trans_afkd = TransformerStudent(TRANSFORMER_CONFIG)
    trans_afkd = train_afkd(
        trans_afkd, bilstm_teacher, X_tr, yfr_tr, yep_tr_arr, device,
        epochs=80, lr=5e-4, beta_max=0.35, temp=2.5, yfr_soft=yfr_clean_tr,  # ALINHADO run_v22
    )
    print("  -> Temperature scaling Trans-AF-KD (v4)...")
    trans_afkd = temperature_scale(trans_afkd, X_val, yfr_val, device)

    # 8. TransformerTeacher (oráculo full-sequence)
    print("\n8. Treinando TransformerTeacher (full-sequence, sem máscara causal)...")
    trans_teacher = TransformerTeacher(TRANSFORMER_TEACHER_CONFIG)
    trans_teacher = train_baseline(
        trans_teacher, X_tr, yfr_tr, yep_tr_arr, device, epochs=60, lr=5e-4,  # ALINHADO run_v22
    )
    print("  TransformerTeacher pronto (oráculo atencional — sem temperature scaling)")

    # 9. LSTM-TransKD  (Transformer -> LSTM)  ★ ABLAÇÃO CENTRAL
    print("\n9. Treinando LSTM-TransKD  [Transformer -> LSTM]  ★ ablação...")
    lstm_transkd = LSTMStudent(input_dim=D, hidden_size=H, num_layers=2, dropout=0.1)
    lstm_transkd = train_afkd(
        lstm_transkd, trans_teacher, X_tr, yfr_tr, yep_tr_arr, device,
        epochs=80, lr=5e-4, beta_max=0.35, temp=2.5, yfr_soft=yfr_clean_tr,  # ALINHADO run_v22
    )
    print("  -> Temperature scaling LSTM-TransKD (v4)...")
    lstm_transkd = temperature_scale(lstm_transkd, X_val, yfr_val, device)

    # 10. Inferência na validação -> calibração v3
    print("\n10. Inferência na validação para calibração v3 (thr_ep, θ_global, θ_adapt, derivativo, híbrido)...")

    progressive_val = np.asarray(meta_val["progressive"]).astype(bool) if "progressive" in meta_val.columns else None
    abrupt_val = ((~progressive_val) & (y_episode_bin[idx_val] == 1)) if progressive_val is not None else None
    progressive_te = np.asarray(meta_te["progressive"]).astype(bool) if "progressive" in meta_te.columns else None
    abrupt_te = ((~progressive_te) & (y_episode_bin[idx_te] == 1)) if progressive_te is not None else None

    va_models = {
        "LSTM-Baseline": infer_stream_fixed_eval(lstm_base, X_val, device),
        "LSTM-AF-KD": infer_stream_fixed_eval(lstm_afkd, X_val, device),
        "LSTM-TransKD": infer_stream_fixed_eval(lstm_transkd, X_val, device),
        "Transformer-Baseline": infer_stream_fixed_eval(trans_base, X_val, device),
        "Transformer-AF-KD": infer_stream_fixed_eval(trans_afkd, X_val, device),
    }

    thr_ep_map: Dict[str, float] = {}
    theta_adapt_map: Dict[str, float] = {}
    deriv_map: Dict[str, Tuple[float, float, int]] = {}
    hybrid_params_map: Dict[str, Optional[Tuple[float, float]]] = {}
    ref_fixed_map: Dict[str, Dict[str, Any]] = {}

    theta_global = select_theta_ttd(yfr_val, va_models["LSTM-Baseline"].frame_probs, m_detect=TTD_M)
    theta_scan = []
    print(f"  θ_global (LSTM-Baseline, MISS_CAP=0.75) = {theta_global:.3f}  [usado nas Análises 1/3]")

    # θ por modelo: calibrado individualmente com FailRate ≤ FAIL_BUDGET
    theta_model_map: Dict[str, float] = {}
    for mdl_name, se_v in va_models.items():
        theta_model_map[mdl_name] = select_theta_per_model(yfr_val, se_v.frame_probs, m_detect=TTD_M)
    print(f"  θ por modelo (FailRate ≤ {FAIL_BUDGET:.2f}, usado na avaliação principal):")
    for mdl_name, th in theta_model_map.items():
        print(f"       {mdl_name:24s} θ={th:.3f}")

    for mdl_name, se_val in va_models.items():
        thr_ep_map[mdl_name] = select_thr_ep(se_val.frame_probs, y_episode_bin[idx_val])
        theta_adapt_map[mdl_name] = select_theta_ttd_adapt(yfr_val, se_val.frame_probs, m_detect=TTD_M)
        deriv_map[mdl_name] = select_derivative_ttd_params(yfr_val, se_val.frame_probs, m_detect=TTD_M)
        dd, dp, ds = deriv_map[mdl_name]
        ref_fixed_map[mdl_name] = summarize_eval(
            se_val, y_episode_bin[idx_val], yfr_val,
            thr_ep=thr_ep_map[mdl_name], theta_ttd=theta_model_map[mdl_name], m_detect=TTD_M,
            progressive_mask=progressive_val, abrupt_mask=abrupt_val,
            theta_ttd_adapt=theta_adapt_map[mdl_name],
            deriv_delta=dd, deriv_prob_floor=dp, deriv_smooth_w=ds,
        )
        hybrid_params_map[mdl_name] = select_hybrid_policy_from_raw(
            se_val.frame_probs, X_val, y_episode_bin[idx_val], yfr_val, se_val.lat_ms,
            thr_ep=thr_ep_map[mdl_name], theta_ttd=theta_model_map[mdl_name], ref_fixed=ref_fixed_map[mdl_name],
            progressive_mask=progressive_val, abrupt_mask=abrupt_val, m_detect=TTD_M,
        ) if RUN_HYBRID_V3 else None
        print(f"    {mdl_name:24s} thr_ep={thr_ep_map[mdl_name]:.3f} θ_model={theta_model_map[mdl_name]:.3f} θ_adapt={theta_adapt_map[mdl_name]:.3f} deriv=({dd:.3f},{dp:.3f},w={ds}) hybrid={hybrid_params_map[mdl_name]}")

    # 11. Inferência no teste
    print("\n11. Inferência no teste (5 modelos)...")
    te_models = {
        "LSTM-Baseline": infer_stream_fixed_eval(lstm_base, X_te, device),
        "LSTM-AF-KD": infer_stream_fixed_eval(lstm_afkd, X_te, device),
        "LSTM-TransKD": infer_stream_fixed_eval(lstm_transkd, X_te, device),
        "Transformer-Baseline": infer_stream_fixed_eval(trans_base, X_te, device),
        "Transformer-AF-KD": infer_stream_fixed_eval(trans_afkd, X_te, device),
    }

    # 12. Avaliação com protocolo v3 — θ por modelo
    print(f"\n12. Avaliando modelos com θ por modelo (FailRate ≤ {FAIL_BUDGET:.2f} por arquitetura)...")
    teacher_map = {
        "LSTM-Baseline": "None",
        "LSTM-AF-KD": "BiLSTM",
        "LSTM-TransKD": "Transformer",
        "Transformer-Baseline": "None",
        "Transformer-AF-KD": "BiLSTM",
    }
    results: Dict[str, Dict] = {}
    hybrid_results: Dict[str, Dict] = {}
    calibration_rows: List[Dict[str, Any]] = []

    for name, se_test in te_models.items():
        teacher_lbl = teacher_map[name]
        parts = name.split("-")
        modelo = parts[0]
        metodo = "-".join(parts[1:]) if len(parts) > 1 else "Baseline"
        dd, dp, ds = deriv_map[name]
        theta_mdl = theta_model_map[name]       # θ calibrado por modelo
        summ = summarize_eval(
            se_test, y_episode_bin[idx_te], yfr_te,
            thr_ep=thr_ep_map[name], theta_ttd=theta_mdl, m_detect=TTD_M,
            progressive_mask=progressive_te, abrupt_mask=abrupt_te,
            theta_ttd_adapt=theta_adapt_map[name],
            deriv_delta=dd, deriv_prob_floor=dp, deriv_smooth_w=ds,
        )
        results[name] = {"Seed": seed, "Modelo": modelo, "Metodo": metodo, "Teacher": teacher_lbl, "Politica": "Fixa", **summ}
        print(
            f"  {name:24s} FIXA  θ={theta_mdl:.3f} -> "
            f"F1={summ['F1']:.4f} "
            f"FR={summ['FailRate']:.4f} "
            f"    TTDef={summ.get('TTDef', float('nan')):.4f}s "
            f"TTD_efet={summ['TTD']:.4f}s "
            f"TTD_bruto={summ.get('TTD_bruto', float('nan')):.4f}s "
            f"TTDad={summ.get('TTD_Adapt', float('nan')):.4f}s "
            f"ECE={summ['ECE']:.4f} "
            f"Lat={summ['Lat_ms']:.2f}ms"
        )

        calibration_rows.append({
            "Seed": seed, "Modelo": name,
            "ThrEP": thr_ep_map[name],
            "ThetaTTD": theta_mdl,              # θ por modelo (novo)
            "ThetaGlobal": theta_global,        # θ do LSTM-Baseline (referência)
            "ThetaAdapt": theta_adapt_map[name],
            "DerivDelta": dd, "DerivProbFloor": dp, "DerivSmoothW": ds,
            "HybridTauDelta": hybrid_params_map[name][0] if hybrid_params_map[name] is not None else np.nan,
            "HybridTauH": hybrid_params_map[name][1] if hybrid_params_map[name] is not None else np.nan,
        })

        if hybrid_params_map[name] is not None:
            tau_d, tau_h = hybrid_params_map[name]
            se_h = infer_stream_hybrid_from_raw(se_test.frame_probs, X_te, se_test.lat_ms, tau_d, tau_h)
            summ_h = summarize_eval(
                se_h, y_episode_bin[idx_te], yfr_te,
                thr_ep=thr_ep_map[name], theta_ttd=theta_mdl, m_detect=TTD_M,
                progressive_mask=progressive_te, abrupt_mask=abrupt_te,
                theta_ttd_adapt=theta_adapt_map[name],
                deriv_delta=dd, deriv_prob_floor=dp, deriv_smooth_w=ds,
            )
            hybrid_results[name] = {"Seed": seed, "Modelo": modelo, "Metodo": metodo, "Teacher": teacher_lbl, "Politica": "Hibrida", **summ_h}
            print(f"  {name:24s} HÍBR  θ={theta_mdl:.3f} -> F1={summ_h['F1']:.4f} FR={summ_h['FailRate']:.4f} TTD={summ_h['TTD']:.4f}s Skip={summ_h['SkipPct']:.2f}% Cost={summ_h['Cost_ms_per_frame']:.4f}ms/fr")

    # 13. Salvar frame_probs raw
    probs_dir = ensure_dir(DATA_DIR / "frame_probs")
    np.save(probs_dir / f"{EXP_NAME}_lstm_base_seed{seed}.npy", te_models["LSTM-Baseline"].frame_probs)
    np.save(probs_dir / f"{EXP_NAME}_lstm_afkd_seed{seed}.npy", te_models["LSTM-AF-KD"].frame_probs)
    np.save(probs_dir / f"{EXP_NAME}_lstm_transkd_seed{seed}.npy", te_models["LSTM-TransKD"].frame_probs)
    np.save(probs_dir / f"{EXP_NAME}_trans_base_seed{seed}.npy", te_models["Transformer-Baseline"].frame_probs)
    np.save(probs_dir / f"{EXP_NAME}_trans_afkd_seed{seed}.npy", te_models["Transformer-AF-KD"].frame_probs)
    np.save(probs_dir / f"{EXP_NAME}_yfr_test_seed{seed}.npy", yfr_te)
    print(f"\n  Frame probs salvos em {probs_dir}  (6 arquivos × seed {seed})")

    return {"seed": seed, "theta_global": theta_global, "theta_model_map": theta_model_map, "theta_scan": theta_scan, "results": results, "hybrid_results": hybrid_results, "calibration": calibration_rows}


# 8) AGREGAÇÃO E EXPORTAÇÃO

# Ordem canônica de apresentação na tabela (para LaTeX)
_TABLE_ORDER = [
    ("LSTM",        "Baseline",  "None"),
    ("LSTM",        "AF-KD",     "BiLSTM"),
    ("LSTM",        "TransKD",   "Transformer"),
    ("Transformer", "Baseline",  "None"),
    ("Transformer", "AF-KD",     "BiLSTM"),
]


def aggregate_and_export_results(
    all_seeds_results: List[Dict[str, Any]],
    output_dir: Path,
) -> pd.DataFrame:
    """Agrega resultados fixos + híbridos e exporta CSV/LaTeX prontos para paper."""
    ensure_dir(output_dir)

    rows_fixed = [m for sd in all_seeds_results for m in sd.get("results", {}).values()]
    rows_hybrid = [m for sd in all_seeds_results for m in sd.get("hybrid_results", {}).values()]
    rows_all = rows_fixed + rows_hybrid
    df = pd.DataFrame(rows_all)
    if df.empty:
        print("aviso: Nenhum resultado para agregar.")
        return pd.DataFrame()

    csv_path = output_dir / "results_all_seeds_unified.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nCSV completo: {csv_path}")

    calib_rows = [row for sd in all_seeds_results for row in sd.get("calibration", [])]
    if calib_rows:
        calib_path = output_dir / "calibration_summary_v8.csv"
        pd.DataFrame(calib_rows).to_csv(calib_path, index=False, float_format="%.6f")
        print(f"Calibração v8 (ThetaTTD=por modelo, ThetaGlobal=ref LSTM-Baseline): {calib_path}")

    grp_cols = ["Modelo", "Metodo", "Teacher", "Politica"] if "Politica" in df.columns else ["Modelo", "Metodo", "Teacher"]
    metric_cols = [c for c in [
        "F1", "FailRate",
        "TTDef",        # [v8] alinhado run_v29: TTD_bruto + FR × T_EP -> comparável às macros do artigo
        "TTD",          # efetivo penalizado — métrica interna
        "TTD_bruto",    # bruto (sem penalidade) — análise auxiliar
        "TTD_Adapt", "TTD_Deriv",
        "ECE", "Lat_ms", "SkipPct", "Cost_ms_per_frame",
        "ThrEP", "ThetaTTD", "ThetaTTD_Adapt",
    ] if c in df.columns]
    summary_rows = []
    for keys, grp in df.groupby(grp_cols):
        rec = {}
        if isinstance(keys, tuple):
            for k, v in zip(grp_cols, keys):
                rec[k] = v
        else:
            rec[grp_cols[0]] = keys
        for mc in metric_cols:
            rec[f"{mc}_mean"] = grp[mc].mean()
            rec[f"{mc}_std"] = grp[mc].std()
        rec["n_seeds"] = len(grp)
        summary_rows.append(rec)
    df_sum = pd.DataFrame(summary_rows)
    sum_path = output_dir / "summary_unified.csv"
    df_sum.to_csv(sum_path, index=False, float_format="%.6f")
    print(f"Sumário: {sum_path}")

    fixed_df = df_sum[df_sum["Politica"] == "Fixa"].copy() if "Politica" in df_sum.columns else df_sum.copy()
    idx_fixed = fixed_df.set_index(["Modelo", "Metodo", "Teacher"])
    lines = [
        "% Teacher Ablation Study v8 — gerador v7, protocolo alinhado ao run_v29",
        "% θ_TTD calibrado POR MODELO (FailRate ≤ FAIL_BUDGET=0.05 por arquitetura)",
        "% TTDef     = TTD_bruto + FR × T_EP  -> ALINHADO run_v29, comparável às macros do artigo",
        "% TTD       = penalizado via (window-t0)*dt — análise interna",
        "% TTD_bruto = somente episódios detectados (sem penalidade) — análise auxiliar",
        "\\begin{tabular}{lllcccccc}",
        "\\toprule",
        "Student & Método & Teacher & F1 & FailRate & TTDef & TTD$_{efetivo}$ & TTD$_{bruto}$ & ECE \\\\",
        "\\midrule",
    ]
    for (modelo, metodo, teacher_lbl) in _TABLE_ORDER:
        key = (modelo, metodo, teacher_lbl)
        if key not in idx_fixed.index:
            continue
        r = idx_fixed.loc[key]
        ttdef_mean = r.get("TTDef_mean", float("nan"))
        ttdef_std  = r.get("TTDef_std",  float("nan"))
        ttd_bruto_mean = r.get("TTD_bruto_mean", float("nan"))
        ttd_bruto_std  = r.get("TTD_bruto_std",  float("nan"))
        lines.append(
            f"{modelo} & {metodo} & {teacher_lbl} & "
            f"{r['F1_mean']:.4f}\\,{{\\tiny$\\pm${r['F1_std']:.4f}}} & "
            f"{r['FailRate_mean']:.4f}\\,{{\\tiny$\\pm${r['FailRate_std']:.4f}}} & "
            f"{ttdef_mean:.3f}\\,{{\\tiny$\\pm${ttdef_std:.3f}}} & "
            f"{r['TTD_mean']:.3f}\\,{{\\tiny$\\pm${r['TTD_std']:.3f}}} & "
            f"{ttd_bruto_mean:.3f}\\,{{\\tiny$\\pm${ttd_bruto_std:.3f}}} & "
            f"{r['ECE_mean']:.4f}\\,{{\\tiny$\\pm${r['ECE_std']:.4f}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    latex_path = output_dir / "table_comparison_latex.tex"
    latex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX: {latex_path}")

    if "Politica" in df_sum.columns and (df_sum["Politica"] == "Hibrida").any():
        idx_h = df_sum[df_sum["Politica"] == "Hibrida"].set_index(["Modelo", "Metodo", "Teacher"])
        lines_h = [
            "% Teacher Ablation Study v4 — política híbrida",
            "\\begin{tabular}{lllcccc}",
            "\\toprule",
            "Student & Método & Teacher & F1 & FailRate & Skip\\% & Cost (ms/fr) \\\\",
            "\\midrule",
        ]
        for (modelo, metodo, teacher_lbl) in _TABLE_ORDER:
            key = (modelo, metodo, teacher_lbl)
            if key not in idx_h.index:
                continue
            r = idx_h.loc[key]
            lines_h.append(
                f"{modelo} & {metodo} & {teacher_lbl} & "
                f"{r['F1_mean']:.4f}\\,{{\\tiny$\\pm${r['F1_std']:.4f}}} & "
                f"{r['FailRate_mean']:.4f}\\,{{\\tiny$\\pm${r['FailRate_std']:.4f}}} & "
                f"{r.get('SkipPct_mean', float('nan')):.2f}\\,{{\\tiny$\\pm${r.get('SkipPct_std', float('nan')):.2f}}} & "
                f"{r.get('Cost_ms_per_frame_mean', float('nan')):.4f}\\,{{\\tiny$\\pm${r.get('Cost_ms_per_frame_std', float('nan')):.4f}}} \\\\")
        lines_h += ["\\bottomrule", "\\end{tabular}"]
        hybrid_path = output_dir / "table_hybrid_latex.tex"
        hybrid_path.write_text("\n".join(lines_h), encoding="utf-8")
        print(f"LaTeX híbrido: {hybrid_path}")

    print("\n" + "=" * 80)
    print("RESUMO FINAL (média ± std)")
    print("=" * 80)
    print(df_sum.to_string(index=False))
    print("=" * 80)
    return df_sum


# 9) ANÁLISES DE ROBUSTEZ E FAIRNESS  (embutido — sem dependência externa)
#
# Origem: analise_fairness_v4.py + analise_robustez_afkd_fgcs_v2.ipynb
# Estendido para: 5 modelos (plots multi-curva) + Análise 4 (sensibilidade m)
#
# Análise 1 — θ Simétrico Global  (referência comparativa)
#   Todos os 5 modelos avaliados com o MESMO θ_global calibrado no LSTM-Baseline.
#   Serve como análise de robustez comparativa — mostra comportamento sob limiar
#   externo, não representa o ponto operacional ótimo de cada modelo.
#   (O ponto operacional ótimo — θ por modelo — é exportado nas tabelas principais.)
#
# Análise 2 — Curva Cobertura × TTD  (independe de θ)
#   Fração de episódios críticos detectados com TTD ≤ t*.
#   Revela modelos que "travam" abaixo de 100%.
#
# Análise 3 — Sensibilidade de θ  (θ ∈ THETA_SENS_GRID)
#   FailRate, TTD bruto e TTD efetivo para todos os 5 modelos.
#   Dominância deve persistir em toda a grade.
#
# Análise 4 — Sensibilidade de m  (m ∈ M_SENS_GRID = [1,2,3,4,5])  ★ NOVO
#   Mesma θ global; varia o número de frames consecutivos exigidos.
#   Isola se AF-KD é robusto a diferentes valores de m.
#
# Análise 5 — Ablação de Teacher (multi-seed agregada)
#   Plots de barra + intervalo de confiança: F1, FailRate, TTD por modelo.
#   Resume as 4 seeds em um único gráfico comparativo.

# Paleta de cores e estilos (5 modelos)
# Cores e linestyles distintos, amigáveis a daltônicos (Tableau + ajustes)
_PALETTE: Dict[str, Tuple[str, str, str]] = {
    # nome                    cor        linestyle  marcador
    "LSTM-Baseline":        ("#e15759",  ":",       "o"),
    "LSTM-AF-KD":           ("#4e79a7",  "-",       "s"),
    "LSTM-TransKD":         ("#59a14f",  "--",      "^"),   # ablação central
    "Transformer-Baseline": ("#f28e2b",  "-.",      "D"),
    "Transformer-AF-KD":    ("#b07aa1",  "-",       "P"),
}

# Nomes abreviados para legendas compactas
_SHORT: Dict[str, str] = {
    "LSTM-Baseline":        "LSTM-Base",
    "LSTM-AF-KD":           "LSTM-AFKD ★",
    "LSTM-TransKD":         "LSTM-TransKD ★",
    "Transformer-Baseline": "Trans-Base",
    "Transformer-AF-KD":    "Trans-AFKD",
}

# Ordem canônica de apresentação (ablação central no centro)
_MODEL_ORDER = [
    "LSTM-Baseline",
    "LSTM-AF-KD",
    "LSTM-TransKD",
    "Transformer-Baseline",
    "Transformer-AF-KD",
]


# Primitivas TTD

def _ttd_vec(
    yfr: np.ndarray,
    probs: np.ndarray,
    thr: float,
    m: int,
    window: int,
    penalize: bool = True,
) -> np.ndarray:
    """
    Vetor de TTDs (em frames) para episódios críticos.
    penalize=True  -> miss recebe (window - t0)  [métrica justa]
    penalize=False -> miss é excluído            [só para TTD bruto]
    """
    out = []
    for yt, yp in zip(yfr, probs):
        onset = np.where(yt > 0)[0]
        if len(onset) == 0:
            continue
        t0  = int(onset[0])
        det = _first_stable_m(yp, t0, thr, m)
        if det is not None:
            out.append(float(det - t0))
        elif penalize:
            out.append(float(window - t0))
    return np.array(out, dtype=float)


def _failrate_thr(yfr: np.ndarray, probs: np.ndarray,
                  thr: float, m: int, window: int) -> Tuple[float, int]:
    miss = total = 0
    for yt, yp in zip(yfr, probs):
        if not np.any(yt > 0):
            continue
        total += 1
        t0 = int(np.where(yt > 0)[0][0])
        if _first_stable_m(yp, t0, thr, m) is None:
            miss += 1
    return miss / max(total, 1), total


def _coverage_curve(
    yfr: np.ndarray, probs: np.ndarray,
    thr: float, m: int, window: int, dt: float,
    n_pts: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna (t_grid_s, coverage) — independe de qualquer θ além do dado."""
    ttds_s = _ttd_vec(yfr, probs, thr, m, window, penalize=True) * dt
    t_grid = np.linspace(0.0, window * dt * 1.02, n_pts)
    cov    = np.array([(ttds_s <= t).mean() for t in t_grid])
    return t_grid, cov


def _save_fig(fig: "plt.Figure", stem: Path) -> None:
    for ext in (".pdf", ".png"):
        fig.savefig(str(stem) + ext, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Salvo: {stem.name}.pdf / .png")


def _print_summary_multi(
    df_sym: pd.DataFrame,
    theta_global: float,
    seed: int,
) -> None:
    """
    Imprime tabela de texto comparando todos os modelos no θ_global.
    Destaca a ablação central LSTM-AF-KD vs LSTM-TransKD.
    Origem: analise_fairness_v4._print_summary(), estendido para 5 modelos.
    """
    # Encontrar a linha θ mais próxima do θ_global
    thetas = df_sym["theta"].unique()
    closest = float(thetas[np.argmin(np.abs(thetas - theta_global))])
    tg = df_sym[np.isclose(df_sym["theta"], closest, atol=1e-4)].copy()
    tg = tg.set_index("modelo")

    print(f"\n  ── Comparativo em θ≈{closest:.3f} (seed={seed}) ──────────────────")
    hdr = f"  {'Modelo':<24} {'FailRate':>10} {'TTD_bruto(s)':>14} {'TTD_efet(s)':>14}"
    print(hdr)
    print(f"  {'-'*64}")
    for mdl in _MODEL_ORDER:
        if mdl not in tg.index:
            continue
        row = tg.loc[mdl]
        mark = " ★" if "TransKD" in mdl or "AF-KD" in mdl else "  "
        print(
            f"  {_SHORT[mdl]+mark:<24}"
            f"  {row['FailRate']:>10.4f}"
            f"  {row['TTD_bruto_s']:>14.4f}"
            f"  {row['TTD_efetivo_s']:>14.4f}"
        )

    # Ablação central: LSTM-Base vs LSTM-AF-KD vs LSTM-TransKD
    print(f"\n  ── Ablação de Teacher (LSTM student fixo) ─────────────────────")
    refs = {"LSTM-Baseline": "Base", "LSTM-AF-KD": "BiLSTM->LSTM",
            "LSTM-TransKD": "Trans->LSTM"}
    ref_vals = {}
    for mdl, label in refs.items():
        if mdl in tg.index:
            ref_vals[label] = tg.loc[mdl]

    if "Base" in ref_vals:
        for label, vals in ref_vals.items():
            if label == "Base":
                continue
            base = ref_vals["Base"]
            dfr  = float(base["FailRate"])      - float(vals["FailRate"])
            dttd = float(base["TTD_efetivo_s"]) - float(vals["TTD_efetivo_s"])
            pct  = dttd / max(abs(float(base["TTD_efetivo_s"])), 1e-9) * 100
            dom  = "domina" if dfr >= 0 and dttd >= 0 else "pior"
            print(
                f"  {label:<18} -> ΔFailRate={dfr:+.4f}  ΔTTDef={dttd:+.4f}s ({pct:.1f}%)"
                f"  [{dom}]"
            )
    print()


# Tabela θ-simétrica multi-modelo

def _sym_table_multi(
    yfr: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    theta_grid: List[float],
    m: int,
    window: int,
    dt: float,
) -> pd.DataFrame:
    """DataFrame com (theta, modelo, FailRate, TTD_bruto_s, TTD_efetivo_s) para N modelos."""
    rows = []
    for thr in theta_grid:
        for label, probs in model_probs.items():
            fr, _ = _failrate_thr(yfr, probs, float(thr), m, window)
            tv_pen  = _ttd_vec(yfr, probs, float(thr), m, window, penalize=True)
            tv_only = _ttd_vec(yfr, probs, float(thr), m, window, penalize=False)
            rows.append({
                "theta":         round(float(thr), 4),
                "modelo":        label,
                "FailRate":      round(fr, 5),
                "TTD_bruto_s":   round(float(np.mean(tv_only)) * dt, 5) if len(tv_only) > 0 else float("nan"),
                "TTD_efetivo_s": round(float(np.mean(tv_pen))  * dt, 5) if len(tv_pen)  > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


# Análise 1 — θ Simétrico Global (5 modelos)

def _plot_a1_theta_symmetric(
    df: pd.DataFrame,
    seed: int,
    theta_global: float,
    out_dir: Path,
) -> None:
    """3 painéis: FailRate, TTD bruto, TTD efetivo × θ — para todos os 5 modelos."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"Análise 1 — θ Simétrico Global (seed {seed})\n"
        "Mesmo limiar aplicado a todos os 5 modelos",
        fontsize=12, fontweight="bold",
    )
    panels = [
        ("FailRate",      "FailRate (↓ melhor)"),
        ("TTD_bruto_s",   "TTD bruto — só detectados (s)\naviso: exclui perdidos"),
        ("TTD_efetivo_s", "TTD efetivo com penalidade (s)\nmétrica justa"),
    ]
    for ax, (col, title) in zip(axes, panels):
        for mdl in _MODEL_ORDER:
            grp = df[df["modelo"] == mdl]
            if grp.empty:
                continue
            c, ls, mk = _PALETTE[mdl]
            lw = 2.5 if "TransKD" in mdl or "AF-KD" in mdl else 1.4
            ax.plot(grp["theta"], grp[col], color=c, linestyle=ls,
                    linewidth=lw, marker=mk, markersize=4,
                    label=_SHORT[mdl])
        ax.axvline(theta_global, color="black", linestyle=":", linewidth=1.2,
                   alpha=0.7, label=f"θ_global={theta_global:.2f}")
        if col == "FailRate":
            ax.axhline(FAIL_BUDGET, color="red", linestyle="--",
                       linewidth=1.0, alpha=0.6, label=f"budget={FAIL_BUDGET}")
            ax.set_ylim(-0.02, 1.02)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("θ (limiar simétrico)", fontsize=9)
        ax.set_ylabel(col.replace("_", " "), fontsize=9)
        ax.legend(fontsize=7.5, ncol=2)
        ax.grid(True, alpha=0.35)
    plt.tight_layout()
    _save_fig(fig, out_dir / f"a1_theta_simetrico_seed{seed}")


# Análise 2 — Curva Cobertura × TTD (5 modelos)

def _plot_a2_coverage_ttd(
    yfr: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    theta_global: float,
    seed: int,
    m: int,
    window: int,
    dt: float,
    out_dir: Path,
) -> None:
    """
    Painel esq : curvas de cobertura acumulada para todos os 5 modelos.
    Painel centro: histogramas de TTD efetivo.
    Painel dir : ablação explícita LSTM-AF-KD vs LSTM-TransKD (cobertura diferencial).
    Salva também CSV com dados da curva de cobertura.
    """
    penalty_s = window * dt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5.5))
    fig.suptitle(
        f"Análise 2 — Curva Cobertura × TTD (seed {seed})\n"
        f"θ={theta_global:.3f} | m={m} | independe de qualquer θ adicional",
        fontsize=12, fontweight="bold",
    )

    # Painel esquerdo: curvas de cobertura
    csv_rows = []
    for mdl in _MODEL_ORDER:
        probs = model_probs[mdl]
        c, ls, mk = _PALETTE[mdl]
        lw = 2.5 if "TransKD" in mdl or "AF-KD" in mdl else 1.4

        tg, cov = _coverage_curve(yfr, probs, theta_global, m, window, dt)
        ax1.plot(tg, cov, color=c, linestyle=ls, linewidth=lw, label=_SHORT[mdl])

        ttd_only = _ttd_vec(yfr, probs, theta_global, m, window, penalize=False) * dt
        if len(ttd_only) > 0:
            ax1.axvline(ttd_only.mean(), color=c, linestyle=":", linewidth=1.0, alpha=0.55)

        for t_, c_ in zip(tg, cov):
            csv_rows.append({"seed": seed, "modelo": mdl, "t_s": round(t_, 4), "coverage": round(c_, 5)})

    ax1.axvline(penalty_s, color="black", linestyle="--", linewidth=1.1, alpha=0.45,
                label=f"penalidade≈{penalty_s:.1f}s")
    ax1.set_xlabel("Orçamento de TTD (s)", fontsize=11)
    ax1.set_ylabel("Fração de episódios críticos cobertos", fontsize=11)
    ax1.set_xlim(0, penalty_s * 1.06)
    ax1.set_ylim(0, 1.06)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.35)
    ax1.set_title(
        "Cobertura acumulada — Baseline trava abaixo de 100%\n"
        "★ LSTM-AF-KD vs LSTM-TransKD: isolamento do teacher",
        fontsize=9,
    )

    # Painel central: histogramas de TTD efetivo
    bins = np.linspace(0, penalty_s * 1.08, 30)
    for mdl in _MODEL_ORDER:
        probs = model_probs[mdl]
        c, ls, mk = _PALETTE[mdl]
        ttds = _ttd_vec(yfr, probs, theta_global, m, window, penalize=True) * dt
        ax2.hist(ttds, bins=bins, alpha=0.45, color=c, label=_SHORT[mdl], density=True)

    ax2.axvline(penalty_s, color="black", linestyle="--", linewidth=1.5,
                label=f"penalidade≈{penalty_s:.1f}s")
    ax2.set_xlabel("TTD efetivo (s) — inclui penalidade", fontsize=11)
    ax2.set_ylabel("Densidade", fontsize=11)
    ax2.set_title(
        "Distribuição do TTD — modelos fracos empilham\nmassa na penalidade máxima",
        fontsize=9,
    )
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.35)

    # Painel direito: ablação explícita LSTM-AF-KD vs LSTM-TransKD
    # Diferença de cobertura em relação ao LSTM-Baseline
    tg_ref, cov_base = _coverage_curve(
        yfr, model_probs["LSTM-Baseline"], theta_global, m, window, dt
    )
    for mdl in ["LSTM-AF-KD", "LSTM-TransKD"]:
        probs = model_probs[mdl]
        c, ls, mk = _PALETTE[mdl]
        _, cov_mdl = _coverage_curve(yfr, probs, theta_global, m, window, dt)
        ax3.plot(tg_ref, cov_mdl - cov_base, color=c, linestyle=ls, linewidth=2.2,
                 label=f"{_SHORT[mdl]} − Base")

    ax3.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.6,
                label="Baseline (referência)")
    ax3.set_xlabel("Orçamento de TTD (s)", fontsize=11)
    ax3.set_ylabel("Δ Cobertura vs LSTM-Baseline", fontsize=11)
    ax3.set_title(
        "★ Ablação: Ganho de cobertura vs Baseline\n"
        "LSTM-AF-KD [BiLSTM->LSTM] vs LSTM-TransKD [Trans->LSTM]",
        fontsize=9,
    )
    ax3.set_xlim(0, penalty_s * 1.06)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.35)

    plt.tight_layout()
    _save_fig(fig, out_dir / f"a2_coverage_ttd_seed{seed}")

    # Salvar CSV com dados da curva de cobertura
    pd.DataFrame(csv_rows).to_csv(
        out_dir / f"a2_coverage_data_seed{seed}.csv", index=False, float_format="%.5f"
    )
    print(f"    Salvo: a2_coverage_data_seed{seed}.csv")


# Análise 3 — Sensibilidade de θ (5 modelos, 4 painéis)

def _plot_a3_theta_sensitivity(
    df: pd.DataFrame,
    seed: int,
    theta_global: float,
    out_dir: Path,
) -> None:
    """
    4 painéis:
      (a) FailRate × θ       (b) TTD bruto × θ
      (c) TTD efetivo × θ   (d) Ganho AF-KD vs Baseline (LSTM-AF-KD − LSTM-Base)
    """
    fig = plt.figure(figsize=(17, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.52, wspace=0.38)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    fig.suptitle(
        f"Análise 3 — Robustez ao Limiar θ (seed {seed})\n"
        "Dominância de AF-KD/TransKD deve persistir para θ razoável",
        fontsize=13, fontweight="bold",
    )

    panels_abc = [
        ("FailRate",      "(a) FailRate × θ\n↓ menor é melhor"),
        ("TTD_bruto_s",   "(b) TTD bruto × θ\naviso: exclui não detectados"),
        ("TTD_efetivo_s", "(c) TTD efetivo × θ\ninclui penalidade"),
    ]
    for ax, (col, title) in zip(axes[:3], panels_abc):
        for mdl in _MODEL_ORDER:
            grp = df[df["modelo"] == mdl]
            if grp.empty:
                continue
            c, ls, mk = _PALETTE[mdl]
            lw = 2.4 if "TransKD" in mdl or "AF-KD" in mdl else 1.3
            ax.plot(grp["theta"], grp[col], color=c, linestyle=ls,
                    linewidth=lw, marker=mk, markersize=3.5,
                    label=_SHORT[mdl])
        ax.axvline(theta_global, color="black", linestyle=":", linewidth=1.2,
                   alpha=0.65, label=f"θ_global={theta_global:.2f}")
        if col == "FailRate":
            ax.axhline(FAIL_BUDGET, color="red", linestyle="--",
                       linewidth=0.9, alpha=0.55)
            ax.set_ylim(-0.02, 1.02)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("θ", fontsize=10)
        ax.set_ylabel(col.replace("_", " "), fontsize=10)
        ax.legend(fontsize=7.5, ncol=2)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, alpha=0.35)

    # Painel (d) — Ganho ablação: LSTM-AF-KD vs LSTM-TransKD
    ax = axes[3]
    d_afkd   = df[df["modelo"] == "LSTM-AF-KD"].set_index("theta")
    d_transkd = df[df["modelo"] == "LSTM-TransKD"].set_index("theta")
    d_base   = df[df["modelo"] == "LSTM-Baseline"].set_index("theta")
    common   = d_afkd.index.intersection(d_transkd.index).intersection(d_base.index)

    width = (common[1] - common[0]) * 0.35 if len(common) > 1 else 0.02
    ax.bar(common - width / 2,
           (d_base.loc[common, "TTD_efetivo_s"] - d_afkd.loc[common, "TTD_efetivo_s"]).values,
           width=width, alpha=0.75, color=_PALETTE["LSTM-AF-KD"][0],
           label="ΔTTDef: Base−AFKD (↑ AFKD domina)")
    ax.bar(common + width / 2,
           (d_base.loc[common, "TTD_efetivo_s"] - d_transkd.loc[common, "TTD_efetivo_s"]).values,
           width=width, alpha=0.75, color=_PALETTE["LSTM-TransKD"][0],
           label="ΔTTDef: Base−TransKD (↑ TransKD domina)")

    ax2r = ax.twinx()
    ax2r.plot(common,
              (d_base.loc[common, "FailRate"] - d_afkd.loc[common, "FailRate"]).values,
              color=_PALETTE["LSTM-AF-KD"][0], linestyle="--",
              linewidth=1.8, marker="s", markersize=4,
              label="ΔFailRate: Base−AFKD")
    ax2r.plot(common,
              (d_base.loc[common, "FailRate"] - d_transkd.loc[common, "FailRate"]).values,
              color=_PALETTE["LSTM-TransKD"][0], linestyle=":",
              linewidth=1.8, marker="^", markersize=4,
              label="ΔFailRate: Base−TransKD")
    ax2r.set_ylabel("ΔFailRate", fontsize=10)
    ax2r.tick_params(axis="y", labelsize=9)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(
        "(d) Ablação de Teacher: Ganho vs LSTM-Baseline\n"
        "★ Barras positivas = modelo domina o Baseline",
        fontsize=11,
    )
    ax.set_xlabel("θ", fontsize=10)
    ax.set_ylabel("ΔTTDef (s)", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, ncol=1)
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    _save_fig(fig, out_dir / f"a3_theta_sensitivity_seed{seed}")


# Análise 4 — Sensibilidade de m  (★ NOVO — nunca executada antes)

def _plot_a4_m_sensitivity(
    yfr: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    theta_global: float,
    m_grid: List[int],
    window: int,
    dt: float,
    seed: int,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Para cada m ∈ m_grid e cada modelo: F1, FailRate, TTD_prog, TTD total.
    Pergunta: o AF-KD é robusto à escolha de m?
    Ablação: LSTM-AF-KD vs LSTM-TransKD com m variando.
    """
    rows = []
    for m_ in m_grid:
        for mdl, probs in model_probs.items():
            # F1
            y_ep   = (yfr.max(axis=1) > 0).astype(int)
            y_pred = ((probs >= theta_global).sum(axis=1) >= m_).astype(int)
            f1     = float(f1_score(y_ep, y_pred, zero_division=0))

            fr, _ = _failrate_thr(yfr, probs, theta_global, m_, window)
            ttd_v  = _ttd_vec(yfr, probs, theta_global, m_, window, penalize=True) * dt

            rows.append({
                "m":      m_,
                "modelo": mdl,
                "F1":     round(f1,  5),
                "FR":     round(fr,  5),
                "TTD":    round(float(np.mean(ttd_v)) if len(ttd_v) > 0 else float("nan"), 5),
            })

    df_m = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Análise 4 — Sensibilidade ao Parâmetro m (seed {seed})\n"
        f"θ fixo = {theta_global:.3f}  |  m ∈ {m_grid}  |  "
        "★ ablação: LSTM-AFKD vs LSTM-TransKD",
        fontsize=12, fontweight="bold",
    )
    panels = [
        ("F1",  "F1  (↑ melhor)"),
        ("FR",  "FailRate  (↓ melhor)"),
        ("TTD", "TTD efetivo (s)  (↓ melhor)"),
    ]
    for ax, (col, ylabel) in zip(axes, panels):
        for mdl in _MODEL_ORDER:
            grp = df_m[df_m["modelo"] == mdl]
            c, ls, mk = _PALETTE[mdl]
            lw = 2.5 if "TransKD" in mdl or "AF-KD" in mdl else 1.3
            ax.plot(grp["m"], grp[col], color=c, linestyle=ls,
                    linewidth=lw, marker=mk, markersize=6,
                    label=_SHORT[mdl])
        if col == "FR":
            ax.axhline(FAIL_BUDGET, color="red", linestyle="--",
                       linewidth=0.9, alpha=0.55, label=f"budget={FAIL_BUDGET}")
            ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("m (frames consecutivos)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(m_grid)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.35)

    plt.tight_layout()
    _save_fig(fig, out_dir / f"a4_m_sensitivity_seed{seed}")
    return df_m


# Análise 5 — Agregação multi-seed (barras + IC bootstrap)

def _plot_a5_multiseed_summary(
    all_results: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    """
    Barras com IC 95% bootstrap para F1, FailRate e TTD — todos os 5 modelos.
    Baseado nas métricas já calculadas (θ por modelo — não precisa de frame_probs).
    """
    rows = []
    for sd in all_results:
        for name, r in sd["results"].items():
            rows.append(r)
    df = pd.DataFrame(rows)

    metrics = [("F1", "F1 (↑)"), ("FailRate", "FailRate (↓)"), ("TTD", "TTD total (s) (↓)")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle(
        "Análise 5 — Sumário Multi-Seed (média ± IC 95% bootstrap)\n"
        f"Seeds: {SEEDS}  |  θ por modelo (FR ≤ {FAIL_BUDGET:.2f})  |  m={TTD_M}  |  K={K_AGG}",
        fontsize=12, fontweight="bold",
    )

    x_pos = np.arange(len(_MODEL_ORDER))
    bar_w = 0.55

    for ax, (col, ylabel) in zip(axes, metrics):
        means, errs, colors = [], [], []
        for mdl in _MODEL_ORDER:
            vals = df[df.apply(
                lambda r: f"{r['Modelo']}-{r['Metodo']}" == mdl, axis=1
            )][col].dropna().values
            if len(vals) == 0:
                means.append(float("nan")); errs.append(0.0); colors.append("gray")
                continue
            boot = [np.mean(np.random.choice(vals, len(vals), replace=True))
                    for _ in range(2000)]
            ci95 = np.percentile(boot, [2.5, 97.5])
            means.append(np.mean(vals))
            errs.append((ci95[1] - ci95[0]) / 2)
            colors.append(_PALETTE[mdl][0])

        bars = ax.bar(x_pos, means, width=bar_w, color=colors, alpha=0.82,
                      yerr=errs, capsize=5, error_kw={"linewidth": 1.5})
        if col == "FailRate":
            ax.axhline(FAIL_BUDGET, color="red", linestyle="--",
                       linewidth=1.0, alpha=0.6, label=f"budget={FAIL_BUDGET}")
            ax.legend(fontsize=9)

        # Rótulo de valor nas barras
        for bar, m_, e in zip(bars, means, errs):
            if not np.isnan(m_):
                ax.text(bar.get_x() + bar.get_width() / 2, m_ + e + 0.003,
                        f"{m_:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([_SHORT[m] for m in _MODEL_ORDER],
                           rotation=22, ha="right", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11)
        ax.grid(True, alpha=0.30, axis="y")

    plt.tight_layout()
    _save_fig(fig, out_dir / "a5_multiseed_summary")
    print(f"    Salvo: a5_multiseed_summary.pdf / .png")


# Ponto de entrada das análises de robustez

def run_robustness_analyses(
    seed: int,
    yfr_test: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    theta_global: float,
    out_dir: Path,
    m: int = TTD_M,
    window: int = T,
) -> None:
    """
    Executa as 4 análises de robustez embutidas para uma seed.

    Parâmetros
    ----------
    seed         : identificador da seed
    yfr_test     : (N, T) frame labels do conjunto de teste
    model_probs  : dict {nome_modelo: (N, T) frame_probs aggregadas K=6}
    theta_global : θ simétrico calibrado na validação
    out_dir      : diretório de saída (será criado se não existir)
    m            : número de frames consecutivos (protocolo)
    window       : janela temporal (frames por episódio)
    """
    ensure_dir(out_dir)
    dt = 10.0 / window

    # Filtrar apenas episódios críticos
    crit = np.array([np.any(yt > 0) for yt in yfr_test])
    yfr  = yfr_test[crit]
    mprobs = {k: v[crit] for k, v in model_probs.items()}
    N_crit = int(crit.sum())

    print(f"\n{'='*72}")
    print(f"  ANÁLISES DE ROBUSTEZ — seed={seed}  |  N_críticos={N_crit}")
    print(f"  θ_global={theta_global:.3f}  |  m={m}  |  K_AGG={K_AGG}")
    print(f"  Modelos: {list(mprobs.keys())}")
    print(f"{'='*72}")

    # Análise 1 — θ simétrico
    print("\n  [1/4] Análise 1 — θ Simétrico Global...")
    df_sym = _sym_table_multi(yfr, mprobs, THETA_SENS_GRID, m, window, dt)
    df_sym.to_csv(out_dir / f"a1_theta_simetrico_seed{seed}.csv", index=False)
    _plot_a1_theta_symmetric(df_sym, seed, theta_global, out_dir)
    _print_summary_multi(df_sym, theta_global, seed)  # tabela de texto no log

    # Análise 2 — Cobertura × TTD
    print("  [2/4] Análise 2 — Curva Cobertura × TTD...")
    _plot_a2_coverage_ttd(yfr, mprobs, theta_global, seed, m, window, dt, out_dir)

    # Análise 3 — Sensibilidade de θ
    print("  [3/4] Análise 3 — Sensibilidade do θ...")
    _plot_a3_theta_sensitivity(df_sym, seed, theta_global, out_dir)

    # Análise 4 — Sensibilidade de m
    print("  [4/4] Análise 4 — Sensibilidade do m...")
    df_m = _plot_a4_m_sensitivity(yfr, mprobs, theta_global,
                                   M_SENS_GRID, window, dt, seed, out_dir)
    df_m.to_csv(out_dir / f"a4_m_sensitivity_seed{seed}.csv", index=False)

    print(f"\n  4 análises concluídas para seed {seed}  ->  {out_dir}")


def run_all_robustness_analyses(
    all_results: List[Dict[str, Any]],
    probs_dir: Path,
    output_dir: Path,
) -> None:
    """
    Carrega frame_probs salvos e executa todas as análises de robustez
    (Análises 1–4) por seed + Análise 5 (multi-seed agregada).
    """
    theta_medio = float(np.mean([r["theta_global"] for r in all_results]))
    rob_dir = ensure_dir(output_dir / "robustez")

    print(f"\n{'='*80}")
    print(f"ANÁLISES DE ROBUSTEZ — θ_global médio (ref. Análises 1/3)={theta_medio:.3f}  |  seeds={[r['seed'] for r in all_results]}")
    print(f"Nota: Análises 1/3 usam θ_global (LSTM-Baseline); avaliação principal usa θ por modelo.")
    print(f"{'='*80}")

    _MODEL_FILE_MAP = {
        "LSTM-Baseline":        "lstm_base",
        "LSTM-AF-KD":           "lstm_afkd",
        "LSTM-TransKD":         "lstm_transkd",
        "Transformer-Baseline": "trans_base",
        "Transformer-AF-KD":    "trans_afkd",
    }

    for res in all_results:
        seed = res["seed"]
        theta = res["theta_global"]   # θ_global do LSTM-Baseline — para Análises 1/3 (θ simétrico)
        seed_dir = ensure_dir(rob_dir / f"seed{seed}")

        # Carregar frame_probs
        try:
            model_probs: Dict[str, np.ndarray] = {}
            yfr_test = np.load(probs_dir / f"{EXP_NAME}_yfr_test_seed{seed}.npy")
            for mdl, fname in _MODEL_FILE_MAP.items():
                model_probs[mdl] = np.load(probs_dir / f"{EXP_NAME}_{fname}_seed{seed}.npy")
        except FileNotFoundError as e:
            print(f"  aviso: seed {seed}: arquivo não encontrado — {e}")
            continue

        try:
            run_robustness_analyses(
                seed=seed,
                yfr_test=yfr_test,
                model_probs=model_probs,
                theta_global=theta,   # θ_global para Análises 1/3
                out_dir=seed_dir,
            )
        except Exception as exc:
            print(f"  Erro nas análises de seed {seed}: {exc}")

    # Análise 5 — multi-seed agregada
    print("\n  [5/5] Análise 5 — Sumário multi-seed (barras + bootstrap)...")
    try:
        _plot_a5_multiseed_summary(all_results, rob_dir)
    except Exception as exc:
        print(f"  Erro na análise multi-seed: {exc}")

    # CSV consolidado de m-sensitivity (todas as seeds)
    try:
        frames_m = []
        for res in all_results:
            seed = res["seed"]
            f = rob_dir / f"seed{seed}" / f"a4_m_sensitivity_seed{seed}.csv"
            if f.exists():
                df_ = pd.read_csv(f)
                df_["seed"] = seed
                frames_m.append(df_)
        if frames_m:
            df_m_all = pd.concat(frames_m, ignore_index=True)
            df_m_all.to_csv(rob_dir / "a4_m_sensitivity_all_seeds.csv", index=False)
            print(f"  CSV consolidado m-sensitivity: {rob_dir / 'a4_m_sensitivity_all_seeds.csv'}")
    except Exception as exc:
        print(f"  Erro ao consolidar m-sensitivity: {exc}")

    print(f"\n  Todas as análises de robustez concluídas -> {rob_dir}")


# 10) MAIN

def main():
    parser = argparse.ArgumentParser(
        description="Teacher Ablation Study — BiLSTM / Transformer / None"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Rodar apenas esta seed (útil para debug). Ex: --seed 42",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Pular treino; usar frame_probs já salvos em disco (só análises).",
    )
    args = parser.parse_args()

    seeds_to_run = [args.seed] if args.seed is not None else SEEDS

    print("=" * 80)
    print("TEACHER ABLATION STUDY")
    print("  Student : LSTM causal | Transformer causal")
    print("  Teacher : BiLSTM (shared) | TransformerTeacher (oráculo) | None")
    print("  Análises: θ-simétrico | Cobertura×TTD | θ-sensitivity | m-sensitivity | multi-seed")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice:       {device}")
    print(f"Seeds:        {seeds_to_run}")
    print(f"m (TTD):      {TTD_M}")
    print(f"M_SENS_GRID:  {M_SENS_GRID}")
    print(f"FailRate_max: {FAIL_BUDGET}")
    print(f"K_AGG:        {K_AGG}")
    print(f"θ grid:       {THETA_GLOBAL_GRID}")
    print(f"θ sens grid:  {len(THETA_SENS_GRID)} pontos em [{THETA_SENS_GRID[0]}, {THETA_SENS_GRID[-1]}]\n")

    output_dir = ensure_dir(Path(f"./resultados_{EXP_NAME}"))
    probs_dir  = ensure_dir(DATA_DIR / "frame_probs")

    # Fase 1: Treino e inferência
    all_results: List[Dict[str, Any]] = []

    if not args.skip_train:
        for seed in seeds_to_run:
            result = run_single_seed_unified(seed, device)
            all_results.append(result)

        print("\n" + "=" * 80)
        print("AGREGANDO RESULTADOS DE MÉTRICAS")
        print("=" * 80)
        aggregate_and_export_results(all_results, output_dir)

    else:
        # Modo --skip-train: reconstruir all_results a partir dos CSVs salvos
        print("  ⚡ --skip-train: lendo resultados salvos em disco...")
        csv_path = output_dir / "results_all_seeds_unified.csv"
        if not csv_path.exists():
            print(f"  ✗ {csv_path} não encontrado. Rode sem --skip-train primeiro.")
            return
        df_saved = pd.read_csv(csv_path)
        for seed in seeds_to_run:
            df_s = df_saved[df_saved["Seed"] == seed]
            theta_g = float(df_s["Theta"].mean()) if not df_s.empty else 0.10
            results_dict = {}
            for _, row in df_s.iterrows():
                name = f"{row['Modelo']}-{row['Metodo']}"
                results_dict[name] = dict(row)
            all_results.append({
                "seed": seed,
                "theta_global": theta_g,
                "theta_scan": [],
                "results": results_dict,
            })
        print(f"  Carregados resultados para seeds: {[r['seed'] for r in all_results]}")

    # Fase 2: Análises de robustez embutidas
    print("\n" + "=" * 80)
    print("ANÁLISES DE ROBUSTEZ E FAIRNESS (embutidas)")
    print("=" * 80)
    try:
        run_all_robustness_analyses(all_results, probs_dir, output_dir)
    except Exception as exc:
        print(f"  Erro nas análises de robustez: {exc}")
        import traceback; traceback.print_exc()

    # Resumo final
    rob_dir = output_dir / "robustez"
    print("\n" + "=" * 80)
    print("EXECUÇÃO CONCLUÍDA")
    print("=" * 80)
    print(f"\nMétricas principais em: {output_dir}/")
    print(f"  ├─ results_all_seeds_unified.csv")
    print(f"  ├─ summary_unified.csv")
    print(f"  └─ table_comparison_latex.tex")
    print(f"\nAnálises de robustez em: {rob_dir}/")
    print(f"  ├─ seed42/  seed43/  seed44/  seed45/")
    print(f"  │   ├─ a1_theta_simetrico_seedXX.csv/.pdf/.png  (θ-simétrico, 5 modelos)")
    print(f"  │   ├─ a2_coverage_ttd_seedXX.pdf/.png          (cobertura × TTD, 3 painéis)")
    print(f"  │   ├─ a2_coverage_data_seedXX.csv              (dados da curva de cobertura)")
    print(f"  │   ├─ a3_theta_sensitivity_seedXX.pdf/.png     (θ-sensitivity, 4 painéis)")
    print(f"  │   └─ a4_m_sensitivity_seedXX.csv/.pdf/.png    (m-sensitivity, 3 painéis)")
    print(f"  ├─ a4_m_sensitivity_all_seeds.csv               <- consolidado multi-seed")
    print(f"  └─ a5_multiseed_summary.pdf/.png                <- barras + bootstrap")
    print(f"\nFrame probs em: {probs_dir}/")
    print(f"  ├─ lstm_base_seedXX.npy | lstm_afkd_seedXX.npy | lstm_transkd_seedXX.npy")
    print(f"  ├─ trans_base_seedXX.npy | trans_afkd_seedXX.npy")
    print(f"  └─ yfr_test_seedXX.npy")
    print(f"\n★ Ablação central:")
    print(f"  LSTM-AF-KD [BiLSTM->LSTM]  vs  LSTM-TransKD [Transformer->LSTM]")
    print(f"  -> student idêntico (LSTM causal), teacher diferente")
    print(f"  -> todas as análises destacam este par em verde tracejado")


if __name__ == "__main__":
    main()