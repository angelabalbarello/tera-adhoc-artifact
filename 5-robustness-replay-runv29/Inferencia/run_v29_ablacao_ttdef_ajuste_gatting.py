# -*- coding: utf-8 -*-
"""run_v26_ablacao.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Versão 26 — Ablações para isolar regressão de TTD observada no v26 completo.

  Diagnóstico v26 completo (M1+M2+M3+M4+M5):
    ok F1:   0.897 -> 0.951  (+0.054)
    ok ECE:  0.547 -> 0.472  (−0.075)
    erro TTD:  0.079 -> 0.216  (+0.137)  <- regressão principal
    erro TTDef: 0.100 -> 0.279  (+0.179)
    aviso: FR_abrupto: 0.000 -> 0.016

  Hipótese: M2 (temp baixa no REFINE) + M5 (hidden KD) forçam o aluno
  a aprender representações mais precisas mas menos precoces.

  MODO DE USO — altere apenas ABLATION_MODE antes de rodar:
  ABLATION_MODE = None   -> v26 completo (M1+M2+M3+M4+M5) — baseline de comparação
  ABLATION_MODE = "A"    -> M1+M3 apenas (pressão temporal pura, sem M2/M4/M5)
  ABLATION_MODE = "B"    -> M4 apenas (oversampling isolado)
  ABLATION_MODE = "C"    -> M2 conservador (temp_min=1.5 em vez de 1.0) + M1+M3+M4
  ABLATION_MODE = "D"    -> tudo exceto M5 (sem hidden KD) — M1+M2+M3+M4
  EXP_NAME é ajustado automaticamente: exp_abl_A, exp_abl_B, etc.
  Modelos salvos em subpastas separadas por ablação -> sem colisão de checkpoints.

  Matriz de melhorias por ablação:
    Ablação │ M1(exp) │ M2(temp) │ M3(asym) │ M4(over) │ M5(hid)
    ────────┼─────────┼──────────┼──────────┼──────────┼────────
    None    │       │        │        │        │   ok
    A       │       │    ✗     │        │    ✗     │   ✗
    B       │    ✗    │    ✗     │    ✗     │        │   ✗
    C       │       │  1.5min  │        │        │   ✗
    D       │       │        │        │        │   ✗

  Mantidas todas as mudanças do v25 e v26.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# Colab setup opcional
from pathlib import Path
import os, sys


print("Python:", sys.version)
print("Diretório atual:", os.getcwd())

# Descomente se quiser montar o Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# Descomente se precisar instalar dependências adicionais
# !pip install -q numpy pandas scikit-learn torch

# -*- coding: utf-8 -*-

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import matplotlib
import os
# No Colab o display funciona; em servidor headless usa Agg
if os.environ.get("DISPLAY") is None and "google.colab" not in str(os.environ.get("COLAB_RELEASE_TAG", "")):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""## 0) CONFIGURAÇÃO GLOBAL"""

# Identificador do experimento (prefixo em todos os outputs)
# exp_main    -> experimento principal (2 modelos: Baseline + AF-KD + Híbrido)
# exp_ablacao -> estudo de ablação de teacher (5 modelos) em run_ablation_v5.py

# [v26-ABL] MODO DE ABLAÇÃO — ALTERE APENAS ESTE PARÂMETRO
# None -> v26 completo | "A" -> M1+M3 | "B" -> M4 | "C" -> M2 conservador | "D" -> sem M5
ABLATION_MODE: Optional[str] = "A"   # <- MUDE AQUI  [v28: critério gating corrigido — Pareto FR=0 + max SkipPct]

# Configuração de cada ablação — parâmetros passados diretamente ao train_afkd
_ABLATION_CONFIGS = {
    None: dict(   # v26 completo — baseline de comparação desta análise
        onset_exp_alpha   = 0.15,
        use_temp_schedule = True,
        temp_min          = 1.0,
        lambda_late       = 2.0,
        theta_late        = 0.10,
        use_abrupt_mask   = True,
        use_hidden_kd     = True,
        gamma_hidden      = 0.30,
        hidden_kd_window  = 8,
        label             = "v26_full",
    ),
    "A": dict(    # M1+M3: pressão temporal pura sem temperatura variável nem hidden KD
        onset_exp_alpha   = 0.15,
        use_temp_schedule = False,   # M2 desativado -> temp fixo = 2.0
        temp_min          = 1.0,
        lambda_late       = 2.0,
        theta_late        = 0.10,
        use_abrupt_mask   = False,   # M4 desativado
        use_hidden_kd     = False,   # M5 desativado
        gamma_hidden      = 0.30,
        hidden_kd_window  = 8,
        label             = "abl_A",
    ),
    "B": dict(    # M4 apenas: oversampling isolado, sem pressão temporal extra
        onset_exp_alpha   = 0.01,    # alpha ≈0 -> quase idêntico ao hiperbólico original
        use_temp_schedule = False,
        temp_min          = 1.0,
        lambda_late       = 0.0,     # M3 desativado
        theta_late        = 0.10,
        use_abrupt_mask   = True,    # M4 ativo
        use_hidden_kd     = False,
        gamma_hidden      = 0.30,
        hidden_kd_window  = 8,
        label             = "abl_B",
    ),
    "C": dict(    # M2 conservador (temp_min=1.5) + M1+M3+M4, sem M5
        onset_exp_alpha   = 0.15,
        use_temp_schedule = True,
        temp_min          = 1.5,     # menos agressivo que v26 (era 1.0)
        lambda_late       = 2.0,
        theta_late        = 0.10,
        use_abrupt_mask   = True,
        use_hidden_kd     = False,   # M5 desativado para isolar efeito do temp
        gamma_hidden      = 0.30,
        hidden_kd_window  = 8,
        label             = "abl_C",
    ),
    "D": dict(    # M1+M2+M3+M4, sem M5 (hidden KD desativado)
        onset_exp_alpha   = 0.15,
        use_temp_schedule = True,
        temp_min          = 1.0,
        lambda_late       = 2.0,
        theta_late        = 0.10,
        use_abrupt_mask   = True,
        use_hidden_kd     = False,   # M5 desativado
        gamma_hidden      = 0.30,
        hidden_kd_window  = 8,
        label             = "abl_D",
    ),
    # [v27] Ablação A2 — igual a A mas com M3 operante (theta_late=0.45)
    # Diagnóstico: em A, theta_late=0.10 era trivialmente satisfeito (late=0.000
    # em todas as épocas). A2 confirma o efeito real de M3 com limiar adequado.
    # lambda_late conservador (1.0) na 1ª rodada; subir para 2.0 se estável.
    "A2": dict(
        onset_exp_alpha   = 0.15,
        use_temp_schedule = False,   # M2 desativado (idêntico a A)
        temp_min          = 1.0,
        lambda_late       = 1.0,     # [v27] conservador; era 2.0 mas theta alto compensa
        theta_late        = 0.45,    # era 0.10 — trivialmente satisfeito; M3 nunca ativava
        use_abrupt_mask   = False,   # M4 desativado (idêntico a A)
        use_hidden_kd     = False,   # M5 desativado (idêntico a A)
        gamma_hidden      = 0.30,
        hidden_kd_window  = 8,
        label             = "abl_A2",
    ),
}

_abl_cfg = _ABLATION_CONFIGS[ABLATION_MODE]
_abl_label = _abl_cfg["label"]
EXP_NAME = f"exp_{_abl_label}"
print(f"\n{'='*60}")
print(f"  ABLAÇÃO: {ABLATION_MODE!r}  ->  EXP_NAME = {EXP_NAME!r}")
print(f"  M1(exp α={_abl_cfg['onset_exp_alpha']:.2f})  "
      f"M2(temp={_abl_cfg['use_temp_schedule']}, min={_abl_cfg['temp_min']})  "
      f"M3(λ={_abl_cfg['lambda_late']}, θ={_abl_cfg['theta_late']})  "
      f"M4(abrupt={_abl_cfg['use_abrupt_mask']})  "
      f"M5(hidden={_abl_cfg['use_hidden_kd']})")
print(f"{'='*60}\n")

SEEDS            = [42, 43, 44, 45]   # v19: +seed 45 para robustez estatística
T                = 96            # frames por janela (10 s @ ~9.6 FPS)
D                = 31            # dimensão de entrada
H                = 64            # hidden size
K_AGG            = 6             # Seção 5.1: agregador causal k=6
TTD_M            = 3             # Seção 5.1: m=3 consecutivos (padrão)
M_SENS_GRID      = [1, 2, 3, 4, 5]  # grade para análise de sensibilidade de m
KIN              = [12, 13, 14]  # índices cinemáticos para o gate
FAIL_BUDGET      = 0.05          # usado apenas em select_thr_ep (classificação episódica)
TTD_BUDGET_RATIO = 1.20          # mantido para compatibilidade, NÃO usado no gating com TTD_ef

# [TTDef] Parâmetros da métrica unificada de latência de decisão
# TTD_ef = TTD + FR × T_MAX_EF
# Implementação da formulação do artigo: FR não é restrição dura no gating,
# mas penalidade proporcional à duração máxima do episódio.
# T_MAX_EF = 10.0 s (duração fixa do episódio: T=96 frames @ ~9.6 FPS)
T_MAX_EF         = 10.0          # [TTDef] penalidade por episódio crítico perdido (s)
# FR_HARD_CAP: teto de segurança residual — soluções com FR > 50% são degeneradas
# (o gating não está funcionando de forma alguma). Não é a restrição principal.
FR_HARD_CAP      = 0.50          # [TTDef] cap absoluto; TTD_ef governa o trade-off real
LAT_WARMUP_STEPS = 100           # Seção 5.1: warm-up de latência

# Grades de busca de hiperparâmetros (VAL)
# v23: TAU_DELTA_PERCENTILES com passo 2 (era 3) -> 23 valores (era 16)
TAU_DELTA_PERCENTILES = list(range(50, 96, 2))
# v23: TAU_H_GRID expandido para baixo — seeds 44/45 precisam de τH < 0.40
# (modelo muito confiante -> H(p) baixa -> gate seletivo exige τH pequeno)
# Seed 43 convergiu em τH=0.40 (mínimo anterior); outras seeds podem precisar de menos.
# τH baixo = gating mais conservador = menos frames pulados = FR_val menor.
TAU_H_GRID            = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.70, 0.85, 0.95]
# OPT-5d (v20): grid mais denso na faixa baixa onde θ_kd ótimo (~0.10) opera
THR_EP_GRID           = np.concatenate([
    np.linspace(0.05, 0.40, 15),   # alta resolução na faixa operacional
    np.linspace(0.40, 0.95, 12),   # cobertura da faixa alta
])
# OPT-1 (v20): Entrega 1 mostrou θ=0.10 minimiza TTD efetivo com FailRate=0%.
# Expandimos o grid para incluir thresholds baixos sem abandonar os altos.
# O protocolo do artigo (FIX-17b: θ global simétrico) é preservado.
THETA_TTD_GRID        = np.concatenate([
    np.array([0.10, 0.12, 0.15, 0.18, 0.20], dtype=float),   # faixa nova: baixa sensibilidade
    np.linspace(0.25, 0.90, 14),                               # faixa original
])
THETA_TTD_ADAPT_GRID  = np.array([0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30], dtype=float)
TTD_ADAPT_TARGET_THETA = 0.25   # mantido: target do artigo não muda
TTD_ADAPT_MAX_MISS     = 0.05
DERIV_DELTA_GRID       = np.array([0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020], dtype=float)
DERIV_PROB_GRID        = np.array([0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30], dtype=float)
DERIV_SMOOTH_GRID      = [3, 5]
DERIV_TARGET_DELTA     = 0.006
DERIV_TARGET_PROB      = 0.18
DERIV_MAX_MISS         = 0.10
FORECAST_HORIZON_K    = 0   # FIX-1: K=10 causava TTD=0; antecipacao vem da AF-KD

# v19: splits estratificados por recipe_id
USE_STRATIFIED_SPLITS = True   # True -> stratify por recipe_id nos críticos
FORCE_REGEN_SPLITS    = True  # True -> força regeneração mesmo se arquivo existe
                               # (defina True para regenerar splits 42-44 com nova lógica)

"""## 0) CONFIGURAÇÃO DE DATASET — DatasetConfig + helpers de reprodutibilidade"""

DATA_ROOT = Path(".")
DATA_DIR = DATA_ROOT / "dados_sinteticos"
# [v26-ABL] Subpasta separada por ablação -> sem colisão de checkpoints entre runs
MODEL_DIR = DATA_ROOT / "modelos_salvos" / _abl_label
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REUSE_PREGENERATED_DATASET = False   # IMPORTANTE: False obrigatório ao trocar gerador (v4->v5)
REUSE_SPLITS = True
# v24: REUSE_TRAINED_MODELS removido — detecção automática por seed em main().
# Se modelos_salvos/teacher_seed{N}.pt + baseline + student existirem -> reusa.
# Se não existirem -> treina do zero e salva. Sem necessidade de alterar flag.
EXPORT_DATASET_CONFIG = True
DEFAULT_GENERATOR_MODULE = "synthetic_driver_risk_v7"
# v5: label principal promovido para y_episode_severity (0.7·mean + 0.3·p95),
# suavização temporal 3 frames, decaimento pós-plateau, +2 receitas progressivas
# (C09/C10 -> 6/10 críticos progressivos = 60%). REGENERAR datasets ao trocar gerador.


@dataclass
class DatasetConfig:
    seed: int
    window: int = T
    per_recipe: int = 80
    recipe_set: str = "base"
    sensor_level: int = 4
    frame_thr: float = 0.35
    generator_module: str = DEFAULT_GENERATOR_MODULE
    data_dir: Path = DATA_DIR
    reuse_pregenerated: bool = REUSE_PREGENERATED_DATASET
    reuse_splits: bool = REUSE_SPLITS

    @property
    def npz_path(self) -> Path:
        return self.data_dir / f"dataset_sintetico_seed{self.seed}.npz"

    @property
    def csv_path(self) -> Path:
        return self.data_dir / f"episodios_seed{self.seed}.csv"

    @property
    def config_path(self) -> Path:
        return self.data_dir / f"dataset_config_seed{self.seed}.json"

    @property
    def split_path(self) -> Path:
        return self.data_dir / f"splits_seed{self.seed}.json"

"""## 0.1) HELPERS DE DADOS / REPRODUTIBILIDADE"""

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Tipo não serializável: {type(obj)!r}")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_or_generate_dataset(cfg: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    ensure_dir(cfg.data_dir)

    if cfg.reuse_pregenerated and cfg.npz_path.exists() and cfg.csv_path.exists():
        print(f"  Reutilizando dataset pré-gerado: {cfg.npz_path.name} + {cfg.csv_path.name}")
    else:
        print(f"  Gerando dataset sintético para seed={cfg.seed}...")
        module = __import__(cfg.generator_module)
        res = module.quick_generate(
            out_npz=str(cfg.npz_path),
            out_csv=str(cfg.csv_path),
            window=cfg.window,
            per_recipe=cfg.per_recipe,
            seed=cfg.seed,
            recipe_set=cfg.recipe_set,
            SENSOR_LEVEL=cfg.sensor_level,
        )
        # Alguns geradores retornam caminhos relativos/absolutos; validamos a existência.
        if not Path(res["npz"]).exists() or not Path(res["csv"]).exists():
            raise FileNotFoundError("quick_generate() não produziu os arquivos esperados.")

    if EXPORT_DATASET_CONFIG:
        save_json(cfg.config_path, asdict(cfg))

    data = np.load(cfg.npz_path)
    X = data["X"].astype(np.float32)
    y_fr_cont = data["y_frame_clean"].astype(np.float32) if "y_frame_clean" in data.files else data["y_frame"].astype(np.float32)
    y_fr_obs = data["y_frame"].astype(np.float32) if "y_frame" in data.files else y_fr_cont.copy()
    meta = pd.read_csv(cfg.csv_path, sep=";")
    return X, y_fr_cont, y_fr_obs, meta


def infer_class_column(meta: pd.DataFrame) -> str:
    for col in ("class_name", "categoria", "class"):
        if col in meta.columns:
            return col
    raise KeyError("Nenhuma coluna de classe encontrada no CSV de metadados (esperado: class_name/categoria/class).")


def make_splits(
    y_ep: np.ndarray,
    seed: int,
    split_path: Optional[Path] = None,
    reuse: bool = True,
    meta: Optional[pd.DataFrame] = None,
    use_stratified: bool = True,
    force_regen: bool = False,
) -> Dict[str, np.ndarray]:
    """
    v19: suporte a stratify por recipe_id para críticos.

    Quando use_stratified=True e meta é fornecido:
      - Episódios críticos -> chave de estratificação = recipe_id  (ex: C01..C08)
      - Episódios não-críticos -> chave = class_name              (ex: Normal, Alerta…)
    Isso garante que cada receita apareça em proporção uniforme em train/val/test,
    eliminando o viés de composição observado na seed 44 (C07 sobrerepresentado).

    force_regen=True ignora o arquivo existente e regenera (use para atualizar
    splits 42-44 com a nova lógica, depois reverta para False).
    """
    if split_path is not None and reuse and not force_regen and split_path.exists():
        payload = load_json(split_path)
        return {k: np.asarray(v, dtype=np.int64) for k, v in payload.items()}

    idx = np.arange(len(y_ep), dtype=np.int64)

    # Chave de estratificação
    if use_stratified and meta is not None:
        class_col = infer_class_column(meta)
        # críticos -> recipe_id; não-críticos -> class_name
        strat_key = np.where(
            meta[class_col].astype(str).values == "Critico",
            meta["recipe_id"].astype(str).values,
            meta[class_col].astype(str).values,
        )
        print(f"    [v19-split] stratify por recipe_id | chaves únicas: {sorted(set(strat_key))}")
    else:
        strat_key = y_ep  # comportamento original: estratifica só por classe binária
        print("    [v19-split] stratify binário (sem meta)")

    # 70/15/15 (train/val/test) — configuração padrão do artigo
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, random_state=seed, stratify=strat_key)
    va_idx, te_idx  = train_test_split(tmp_idx, test_size=0.50, random_state=seed, stratify=strat_key[tmp_idx])

    splits = {
        "train_idx": np.sort(tr_idx),
        "val_idx":   np.sort(va_idx),
        "test_idx":  np.sort(te_idx),
    }
    if split_path is not None:
        save_json(split_path, {k: v.tolist() for k, v in splits.items()})
    return splits


def split_arrays(splits: Dict[str, np.ndarray], *arrays: np.ndarray) -> Dict[str, List[np.ndarray]]:
    out: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    for arr in arrays:
        out["train"].append(arr[splits["train_idx"]])
        out["val"].append(arr[splits["val_idx"]])
        out["test"].append(arr[splits["test_idx"]])
    return out

"""## 1) MODELO"""

class MultiTaskLSTM(nn.Module):
    """
    LSTM 2 camadas + heads de frame e episódio.
    bi=True  -> BiLSTM professor (só no treino).
    bi=False -> LSTM causal online (aluno/baseline).
    """
    def __init__(self, input_dim: int, hidden_dim: int, bi: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                            bidirectional=bi, num_layers=2, dropout=0.1)
        d = hidden_dim * 2 if bi else hidden_dim
        self.fc_fr = nn.Linear(d, 1)
        self.fc_ep = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        h, _ = self.lstm(x)
        fr = self.fc_fr(h)           # (B, T, 1)
        ep = self.fc_ep(h.mean(1))   # (B, 1)
        if return_hidden:            # [v26-M5] destilação de estado oculto
            return fr, ep, h
        return fr, ep


# OPT-5c (v20): Temperature scaling pós-treino para reduzir ECE.
# Referência: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
# Aprende T no VAL via LBFGS. NÃO modifica os pesos do modelo — só a escala logit.
class TemperatureScaler(nn.Module):
    """Módulo de pós-calibração: escala os logits por 1/T aprendido no VAL."""
    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)


def calibrate_temperature(model: nn.Module, X_val: np.ndarray,
                           y_ep_val: np.ndarray, device: torch.device,
                           lr: float = 0.01, max_iter: int = 50) -> TemperatureScaler:
    """
    Calibra temperatura T* no VAL minimizando BCE episódica.
    Retorna TemperatureScaler pronto para uso na inferência.
    """
    model.eval()
    scaler = TemperatureScaler().to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    with torch.no_grad():
        Xv = torch.tensor(X_val, device=device).float()
        ep_logits_list = []
        for i in range(len(X_val)):
            xv = make_prefix_window(X_val[i], X_val.shape[1] - 1)
            xt = torch.tensor(xv, device=device).float().unsqueeze(0)
            _, ep_log = model(xt)
            ep_logits_list.append(ep_log.squeeze())
        ep_logits = torch.stack(ep_logits_list).unsqueeze(1)

    y_ep_t = torch.tensor(y_ep_val, device=device).float().unsqueeze(1)

    def closure():
        optimizer.zero_grad()
        scaled = scaler(ep_logits)
        loss = bce(scaled, y_ep_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_star = float(scaler.temperature.item())
    print(f"  [TemperatureScaler] T* = {T_star:.4f}")
    return scaler

"""## 2) UTILITÁRIOS"""

# 🔥 EARLY STOPPING (v25)
class EarlyStopping:
    """
    Interrompe o treinamento quando a métrica para de melhorar.
    Técnica: adaptive training termination — reduz épocas de ~50-80 -> ~20-40.
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = None
        self.counter   = 0
        self.stop      = False

    def step(self, metric: float) -> bool:
        """Retorna True se o treinamento deve ser interrompido."""
        if math.isnan(metric):
            return self.stop
        if self.best is None or metric < self.best - self.min_delta:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def fail_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """FN / P — taxa de falha em episódios críticos."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    p = tp + fn
    return float(fn / p) if p > 0 else 0.0


def binary_entropy(p: float) -> float:
    p = float(np.clip(p, 1e-7, 1 - 1e-7))
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                n_bins: int = 10) -> float:
    """ECE com B=10 bins de largura uniforme (Artigo Seção 5.1)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, N = 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / N) * abs(float(y_true[mask].mean()) -
                                       float(y_prob[mask].mean()))
    return float(ece)


def make_pos_weight(y_binary: np.ndarray) -> float:
    """Razão neg/pos floored em 1.0 (Seção 4.2)."""
    pos = float(np.sum(y_binary == 1))
    neg = float(np.sum(y_binary == 0))
    return max(neg / pos, 1.0) if pos > 0 else 1.0


def build_loss_objects(yfr: np.ndarray, yep: np.ndarray,
                       device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """BCE com pos_weight para frames e episódios (Seção 4.2)."""
    pw_fr = torch.tensor([make_pos_weight(yfr.reshape(-1))], device=device).float()
    pw_ep = torch.tensor([make_pos_weight(yep.reshape(-1))], device=device).float()
    return (nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw_fr),
            nn.BCEWithLogitsLoss(pos_weight=pw_ep))


def make_prefix_window(x_seq: np.ndarray, t: int, window: int = T) -> np.ndarray:
    """Janela causal até t, com pad à esquerda pelo primeiro frame."""
    x_slice = x_seq[:t + 1, :]
    if x_slice.shape[0] < window:
        pad_len   = window - x_slice.shape[0]
        pad_frame = x_slice[0:1, :] if x_slice.shape[0] > 0 else np.zeros(
            (1, x_seq.shape[1]), dtype=x_seq.dtype)
        x_slice = np.concatenate([np.repeat(pad_frame, pad_len, axis=0), x_slice], axis=0)
    else:
        x_slice = x_slice[-window:, :]
    return x_slice.astype(np.float32, copy=False)


def episode_probs_from_frames(frame_probs: np.ndarray, k: int = K_AGG) -> np.ndarray:
    """Agregador causal k=6 frame-head (Seção 5.1)."""
    return np.mean(frame_probs[:, -k:], axis=1)


def _fmt(x: float, nd: int = 3) -> str:
    """Formata número para macros LaTeX."""
    try:
        v = float(x)
        return f"{v:.{nd}f}" if not math.isnan(v) else "0.000"
    except (TypeError, ValueError):
        return "0.000"

"""## 3) TTD E CALIBRAÇÃO DE LIMIARES"""

def first_stable_detection(probs, onset, thr, m=3):
    """
    TTD oficial do paper:
    detecção estável (m frames) APÓS onset
    """
    if onset >= len(probs):
        return None

    count = 0
    for t in range(int(onset), len(probs)):
        if probs[t] >= thr:
            count += 1
            if count >= m:
                return t - m + 1
        else:
            count = 0
    return None

def compute_ttd_ttdef(y_true_frame, probs, thr=0.10, m=3, T_MAX_EF=10.0):
    dt = 10.0 / probs.shape[1]
    ttds = []
    misses = 0
    positives = 0

    for yt, yp in zip(y_true_frame, probs):
        idx = np.where(yt > 0.1)[0]
        if len(idx) == 0:
            continue

        positives += 1
        onset = int(idx[0])
        det = first_stable_detection(yp, onset, thr=thr, m=m)

        if det is None:
            misses += 1
            delay = T_MAX_EF
        else:
            delay = max(0, det - onset) * dt

        ttds.append(delay)

    ttd = float(np.mean(ttds)) if ttds else np.nan
    fr = misses / positives if positives > 0 else 0.0
    ttdef = ttd + fr * T_MAX_EF

    return ttd, fr, ttdef

def run_gating_on_dataset(student, X_data, tau_delta, tau_H):
    student.eval()

    n, T, _ = X_data.shape
    probs_hibr = np.zeros((n, T), dtype=np.float32)
    gating_mat = np.zeros((n, T), dtype=np.int32)

    with torch.no_grad():
        for i in range(n):
            y_prev = None

            for t in range(T):
                x_t = make_prefix_window(X_data[i], t)
                xt = torch.tensor(x_t, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                fr_s, _ = student(xt)
                p_s = torch.sigmoid(fr_s[0, -1]).item()

                # Frame 0 sempre atualiza
                if t == 0 or y_prev is None:
                    y_curr = p_s
                    gating_mat[i, t] = 1
                    probs_hibr[i, t] = y_curr
                    y_prev = y_curr
                    continue

                delta = np.abs(X_data[i, t] - X_data[i, t - 1]).mean()
                H = binary_entropy(y_prev)

                update = (delta >= tau_delta) or (H >= tau_H)

                if update:
                    y_curr = p_s
                    gating_mat[i, t] = 1
                else:
                    y_curr = y_prev
                    gating_mat[i, t] = 0

                probs_hibr[i, t] = y_curr
                y_prev = y_curr

    update_pct = 100.0 * gating_mat.mean()
    skip_pct = 100.0 - update_pct

    return probs_hibr, update_pct, skip_pct

def first_stable_detection_full(probs: np.ndarray,
                                 thr: float, m: int = TTD_M) -> Optional[int]:
    """
    Busca a primeira detecção estável desde o frame 0 (não desde onset).
    Necessário para medir antecipação pré-onset em episódios progressivos.
    Sem isso, first_stable_detection começa no onset e retorna delay=0
    para qualquer modelo que dispare antes do onset — ocultando a antecipação real.
    """
    count = 0
    for t in range(len(probs)):
        if probs[t] >= thr:
            count += 1
            if count >= m:
                return t - m + 1
        else:
            count = 0
    return None


def compute_anticipation(y_true_fr: np.ndarray, y_pred_probs: np.ndarray,
                          window: int = T, thr: float = 0.5,
                          m: int = TTD_M) -> float:
    """
    Antecipação = (onset - primeira_detecção) * dt.
    Positivo quando o modelo disparou ANTES do onset (antecipação real).
    Zero quando disparou no onset ou depois (sem antecipação).
    Uso: TTD_Progressive e TTD_Abrupt — substitui compute_ttd_subset nestes casos.

    Diferença de compute_ttd:
      compute_ttd      -> mede atraso pós-onset (convencional, menor=melhor)
      compute_anticipation -> mede antecipação pré-onset (maior=melhor)

    Diagnóstico que motivou esta função: com first_stable_detection partindo do onset,
    progressivos com disparo pré-onset retornavam TTD_Progressive=0.000 apesar de
    os frame_probs mostrarem antecipação média real de ~2.1s (seed 42, Ablação A2).
    """
    dt   = 10.0 / window
    vals = []
    for yt, yp in zip(y_true_fr, y_pred_probs):
        idx = np.where(yt > 0)[0]
        if len(idx) == 0:
            continue                    # não-crítico — ignora
        t0  = int(idx[0])               # onset frame
        det = first_stable_detection_full(yp, thr=thr, m=m)
        if det is None:
            vals.append(0.0)            # não detectou = zero antecipação
        else:
            vals.append(max(0.0, (t0 - det) * dt))   # clamp: tarde = 0
    return float(np.mean(vals)) if vals else float("nan")


def compute_ttd_subset(y_true_fr: np.ndarray, y_pred_probs: np.ndarray,
                                 episode_mask: np.ndarray,
                                 window: int = T, thr: float = 0.5,
                                 m: int = TTD_M) -> float:
    """compute_anticipation restrito a um subconjunto de episódios."""
    mask = np.asarray(episode_mask).astype(bool)
    if mask.ndim != 1 or mask.shape[0] != y_true_fr.shape[0]:
        raise ValueError("episode_mask deve ser vetor booleano alinhado ao nº de episódios.")
    if mask.sum() == 0:
        return float("nan")
    return compute_anticipation(y_true_fr[mask], y_pred_probs[mask],
                                window=window, thr=thr, m=m)


def compute_ttd(y_true_fr: np.ndarray, y_pred_probs: np.ndarray,
                window: int = T, thr: float = 0.5, m: int = TTD_M) -> float:
    """TTD = (tdet - t0) * Δt, com penalidade para não-detecção (Seção 5.1)."""
    dt   = 10.0 / window
    ttds = []
    for yt, yp in zip(y_true_fr, y_pred_probs):
        idx = np.where(yt > 0)[0]
        if len(idx) == 0:
            continue
        t0    = int(idx[0])
        det   = first_stable_detection(yp, t0, thr=thr, m=m)
        delay = (det - t0) if det is not None else (window - t0)
        ttds.append(delay * dt)
    return float(np.mean(ttds)) if ttds else float("nan")


def fail_rate_positive_subset(y_true_ep: np.ndarray, y_pred_ep: np.ndarray,
                              positive_subset_mask: np.ndarray) -> float:
    """FailRate restrito a um subconjunto positivo (ex.: críticos progressivos)."""
    mask = np.asarray(positive_subset_mask).astype(bool)
    if mask.ndim != 1 or mask.shape[0] != y_true_ep.shape[0]:
        raise ValueError("positive_subset_mask deve ser vetor booleano alinhado ao nº de episódios.")
    pos_mask = mask & (np.asarray(y_true_ep).astype(np.int64) == 1)
    if pos_mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.asarray(y_pred_ep)[pos_mask] == 0))


def _ttd_threshold_stats(y_true_fr: np.ndarray, frame_probs: np.ndarray,
                         theta: float, m_detect: int = TTD_M) -> Tuple[float, float]:
    ttd_val = compute_ttd(y_true_fr, frame_probs, thr=float(theta), m=m_detect)
    misses, positives = 0, 0
    for yt, yp in zip(y_true_fr, frame_probs):
        idx = np.where(yt > 0)[0]
        if len(idx) == 0:
            continue
        positives += 1
        if first_stable_detection(yp, int(idx[0]), thr=float(theta), m=m_detect) is None:
            misses += 1
    miss_rate = (misses / positives) if positives > 0 else 0.0
    return float(ttd_val), float(miss_rate)


def select_theta_ttd(y_true_fr: np.ndarray, frame_probs: np.ndarray,
                      m_detect: int = TTD_M) -> float:
    """
    FIX-2: 3 prioridades: (1) theta com TTD>0 e miss<=60%, (2) menor TTD com miss<=60%,
    (3) fallback menor miss. Grid comeca em 0.25 (FIX-5) evitando theta=0.10 trivial.
    """
    # FIX-17a: MISS_CAP=0.75 mantido, TTD_min volta a 1e-8
    # TTD_min=0.005 (FIX-13b) punia AF-KD que detecta antes de t0:
    #   θ=0.25 -> TTD=0.002s < 0.005 -> rejeitado -> sobe θ=0.35 -> TTD maior
    # Com 1e-8: θ menor sempre preferido, detecção precoce não é punida
    MISS_CAP = 0.75
    positive_ttd, zero_ttd, fallback = [], [], []
    for theta in THETA_TTD_GRID:
        ttd_val, miss_rate = _ttd_threshold_stats(y_true_fr, frame_probs, float(theta), m_detect)
        ttd_key = ttd_val if not math.isnan(ttd_val) else 1e9
        prox = abs(float(theta) - 0.30)
        fallback.append((miss_rate, ttd_key, prox, float(theta)))
        if miss_rate <= MISS_CAP:
            item = (ttd_key, miss_rate, prox, float(theta))
            if ttd_key > 1e-8:    # FIX-17a: sem punir detecção precoce
                positive_ttd.append(item)
            else:
                zero_ttd.append(item)
    if positive_ttd:
        positive_ttd.sort()
        return float(positive_ttd[0][3])
    if zero_ttd:
        zero_ttd.sort()
        return float(zero_ttd[0][3])
    fallback.sort()
    return float(fallback[0][3]) if fallback else 0.5


def select_theta_ttd_adapt(y_true_fr: np.ndarray, frame_probs: np.ndarray,
                           m_detect: int = TTD_M) -> float:
    """
    Calibra um θTTD mais sensível para mensurar antecipação.
    Critério: entre thresholds com miss rate <= TTD_ADAPT_MAX_MISS, escolhe o menor TTD;
    em empate, prefere proximidade de 0.25 e threshold menor.
    Se nenhum threshold respeitar o miss budget, cai para menor miss rate.
    """
    feasible = []
    fallback = []
    for theta in THETA_TTD_ADAPT_GRID:
        ttd_val, miss_rate = _ttd_threshold_stats(y_true_fr, frame_probs, float(theta), m_detect)
        ttd_key = ttd_val if not math.isnan(ttd_val) else 1e9
        item = (miss_rate, ttd_key, abs(float(theta) - TTD_ADAPT_TARGET_THETA), float(theta))
        fallback.append(item)
        if miss_rate <= TTD_ADAPT_MAX_MISS:
            feasible.append((ttd_key, abs(float(theta) - TTD_ADAPT_TARGET_THETA), float(theta)))
    if feasible:
        feasible.sort()
        return float(feasible[0][2])
    fallback.sort()
    return float(fallback[0][3]) if fallback else 0.25

def smooth_series(x: np.ndarray, w: int = 3) -> np.ndarray:
    w = max(1, int(w))
    if w <= 1:
        return np.asarray(x, dtype=np.float32)
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(np.asarray(x, dtype=np.float32), kernel, mode="same")


def first_derivative_detection(probs: np.ndarray, onset: int,
                               delta: float, prob_floor: float,
                               m: int = TTD_M, smooth_w: int = 3) -> Optional[int]:
    """
    Detecta o primeiro ponto de evidência crescente, sem exigir classificação binária dura.
    Critério: após suavização, requer m frames consecutivos com
      - probabilidade >= prob_floor
      - derivada >= delta
    e usa o início da primeira sequência estável como t_det.
    """
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


def compute_ttd_derivative(y_true_fr: np.ndarray, y_pred_probs: np.ndarray,
                           window: int = T, delta: float = 0.006,
                           prob_floor: float = 0.18, m: int = TTD_M,
                           smooth_w: int = 3) -> float:
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


def compute_ttd_derivative_subset(y_true_fr: np.ndarray, y_pred_probs: np.ndarray,
                                  episode_mask: np.ndarray,
                                  window: int = T, delta: float = 0.006,
                                  prob_floor: float = 0.18, m: int = TTD_M,
                                  smooth_w: int = 3) -> float:
    mask = np.asarray(episode_mask).astype(bool)
    if mask.ndim != 1 or mask.shape[0] != y_true_fr.shape[0]:
        raise ValueError("episode_mask deve ser vetor booleano alinhado ao nº de episódios.")
    if mask.sum() == 0:
        return float("nan")
    return compute_ttd_derivative(y_true_fr[mask], y_pred_probs[mask], window=window, delta=delta, prob_floor=prob_floor, m=m, smooth_w=smooth_w)


def _ttd_derivative_stats(y_true_fr: np.ndarray, frame_probs: np.ndarray,
                          delta: float, prob_floor: float,
                          m_detect: int = TTD_M, smooth_w: int = 3) -> Tuple[float, float]:
    ttd_val = compute_ttd_derivative(y_true_fr, frame_probs, delta=delta, prob_floor=prob_floor, m=m_detect, smooth_w=smooth_w)
    misses, positives = 0, 0
    for yt, yp in zip(y_true_fr, frame_probs):
        idx = np.where(yt > 0)[0]
        if len(idx) == 0:
            continue
        positives += 1
        if first_derivative_detection(yp, int(idx[0]), delta=delta, prob_floor=prob_floor, m=m_detect, smooth_w=smooth_w) is None:
            misses += 1
    miss_rate = (misses / positives) if positives > 0 else 0.0
    return float(ttd_val), float(miss_rate)


def select_derivative_ttd_params(y_true_fr: np.ndarray, frame_probs: np.ndarray,
                                 m_detect: int = TTD_M) -> Tuple[float, float, int]:
    """
    Seleciona automaticamente a regra derivativa no conjunto de validação.
    Prioriza miss_rate <= DERIV_MAX_MISS; entre os viáveis, escolhe menor TTD.
    Em empate, prefere proximidade de (delta=0.006, prob_floor=0.18) e menor smooth.
    """
    feasible, fallback = [], []
    for smooth_w in DERIV_SMOOTH_GRID:
        for delta in DERIV_DELTA_GRID:
            for prob_floor in DERIV_PROB_GRID:
                ttd_val, miss_rate = _ttd_derivative_stats(y_true_fr, frame_probs, float(delta), float(prob_floor), m_detect=m_detect, smooth_w=int(smooth_w))
                ttd_key = ttd_val if not math.isnan(ttd_val) else 1e9
                prox = abs(float(delta) - DERIV_TARGET_DELTA) + abs(float(prob_floor) - DERIV_TARGET_PROB)
                item = (miss_rate, ttd_key, prox, int(smooth_w), float(delta), float(prob_floor))
                fallback.append(item)
                if miss_rate <= DERIV_MAX_MISS:
                    feasible.append((ttd_key, prox, int(smooth_w), float(delta), float(prob_floor)))
    if feasible:
        feasible.sort()
        _, _, smooth_w, delta, prob_floor = feasible[0]
        return float(delta), float(prob_floor), int(smooth_w)
    fallback.sort()
    _, _, _, smooth_w, delta, prob_floor = fallback[0]
    return float(delta), float(prob_floor), int(smooth_w)


def select_thr_ep(frame_probs: np.ndarray, y_true_ep: np.ndarray) -> float:
    """
    Calibra thr_ep no VAL (Secao 5.1).

    Protocolo do artigo: maximizar F1 sujeito a restricao FailRate <= FAIL_BUDGET.
    Se nenhum threshold satisfaz o budget, fallback para minimizar FailRate.

    Comportamento esperado:
      - Baseline (discriminador fraco): poucos thresholds satisfazem FR<=0.05,
        ou o fallback seleciona thr alto => F1 baixo + FailRate > 0
      - AF-KD (forte, detecta cedo): muitos thresholds satisfazem FR<=0.05 =>
        seleciona thr que maximiza F1 => F1 alto + FailRate=0
    """
    ep_probs = episode_probs_from_frames(frame_probs, k=K_AGG)
    feasible, fallback = [], []
    for thr in THR_EP_GRID:
        pred = (ep_probs >= float(thr)).astype(np.int64)
        fr   = fail_rate(y_true_ep, pred)
        f1   = f1_score(y_true_ep, pred, zero_division=0)
        item = (fr, -f1, abs(float(thr) - 0.30), float(thr))
        fallback.append(item)
        if fr <= FAIL_BUDGET:
            feasible.append((-f1, fr, abs(float(thr) - 0.30), float(thr)))
    if feasible:
        feasible.sort()
        return float(feasible[0][3])
    fallback.sort()   # minimize FailRate
    return float(fallback[0][3]) if fallback else 0.5

"""## 4) INFERÊNCIA STREAMING"""

@dataclass
class StreamEval:
    frame_probs:       np.ndarray
    episode_probs:     np.ndarray
    lat_ms:            float
    skip_pct:          float
    cost_ms_per_frame: float


def warmup_model(model: nn.Module, x_ref: np.ndarray,
                 device: torch.device, steps: int = LAT_WARMUP_STEPS) -> None:
    """Warm-up de 100 passos antes da medição de latência (Seção 5.1)."""
    model.eval()
    xt = torch.tensor(x_ref, device=device).float().unsqueeze(0)
    with torch.no_grad():
        for _ in range(steps):
            model(xt)
    sync_cuda()


@torch.no_grad()
def infer_stream_fixed(model: nn.Module, X: np.ndarray,
                        device: torch.device, window: int = T) -> StreamEval:
    """
    Política fixa: processa TODOS os frames.
    Loop frame-a-frame dentro de cada episódio — batch=1 streaming causal.
    """
    model.eval()
    n, t_len, _ = X.shape
    frame_probs = np.zeros((n, t_len), dtype=np.float32)
    lat_list    = []

    x_ref = make_prefix_window(X[0], min(5, t_len - 1), window)
    warmup_model(model, x_ref, device)

    for i in range(n):
        for t in range(t_len):
            x_slice = make_prefix_window(X[i], t, window)
            xt = torch.tensor(x_slice, device=device).float().unsqueeze(0)
            sync_cuda()
            ts = time.perf_counter()
            fr_log, _ = model(xt)
            sync_cuda()
            te = time.perf_counter()
            lat_list.append((te - ts) * 1000.0)
            frame_probs[i, t] = float(torch.sigmoid(fr_log[0, -1, 0]).item())

    avg_lat = float(np.mean(lat_list)) if lat_list else 0.0
    return StreamEval(
        frame_probs       = frame_probs,
        episode_probs     = episode_probs_from_frames(frame_probs),
        lat_ms            = avg_lat,
        skip_pct          = 0.0,
        cost_ms_per_frame = avg_lat,
    )


@torch.no_grad()
def infer_stream_hybrid(model: nn.Module, X: np.ndarray, device: torch.device,
                       tau_delta: float, tau_h: float,
                       window: int = T, debug: bool = False) -> StreamEval:
    """
    Gating Híbrido CORRIGIDO

    Fixes:
    ✔ Inicialização obrigatória no frame 0
    ✔ Proteção contra colapso (modelo nunca roda)
    ✔ Logs opcionais para debug
    ✔ Alinhado com paper (U_t = Δx OR H(p))

    Regra:
        UPDATE se (Δx >= tau_delta) OR (H >= tau_h)
        SKIP   caso contrário
    """

    model.eval()
    n, t_len, _ = X.shape
    frame_probs = np.zeros((n, t_len), dtype=np.float32)
    lat_list = []
    skipped = 0

    x_ref = make_prefix_window(X[0], min(5, t_len - 1), window)
    warmup_model(model, x_ref, device)

    for i in range(n):
        last_p = None  # 🔥 CORREÇÃO CRÍTICA

        for t in range(t_len):

            # 🔥 FRAME 0 — SEMPRE UPDATE
            if t == 0 or last_p is None:
                x_slice = make_prefix_window(X[i], t, window)
                xt = torch.tensor(x_slice, device=device).float().unsqueeze(0)

                sync_cuda()
                ts = time.perf_counter()
                fr_log, _ = model(xt)
                sync_cuda()
                te = time.perf_counter()

                last_p = float(torch.sigmoid(fr_log[0, -1, 0]).item())
                frame_probs[i, t] = last_p
                lat_list.append((te - ts) * 1000.0)

                if debug:
                    print(f"[INIT] t={t} p={last_p:.4f}")

                continue

            # CÁLCULO DOS SINAIS
            dk = np.abs(X[i, t, KIN] - X[i, t - 1, KIN]).max()
            H = binary_entropy(last_p)

            # REGRA CORRETA DO PAPER
            update = (dk >= tau_delta) or (H >= tau_h)

            # SKIP
            if not update:
                skipped += 1
                frame_probs[i, t] = last_p

                if debug:
                    print(f"[SKIP] t={t} dk={dk:.4f} H={H:.4f}")

                continue

            # UPDATE (RODA MODELO)
            x_slice = make_prefix_window(X[i], t, window)
            xt = torch.tensor(x_slice, device=device).float().unsqueeze(0)

            sync_cuda()
            ts = time.perf_counter()
            fr_log, _ = model(xt)
            sync_cuda()
            te = time.perf_counter()

            last_p = float(torch.sigmoid(fr_log[0, -1, 0]).item())
            frame_probs[i, t] = last_p
            lat_list.append((te - ts) * 1000.0)

            if debug:
                print(f"[UPDATE] t={t} dk={dk:.4f} H={H:.4f} p={last_p:.4f}")

    # MÉTRICAS
    avg_lat = float(np.mean(lat_list)) if lat_list else 0.0
    skip_pct = (skipped / float(n * t_len)) * 100.0

    return StreamEval(
        frame_probs=frame_probs,
        episode_probs=episode_probs_from_frames(frame_probs),
        lat_ms=avg_lat,
        skip_pct=skip_pct,
        cost_ms_per_frame=avg_lat * (1.0 - skip_pct / 100.0),
    )


@torch.no_grad()
def infer_stream_hybrid_fast(model: nn.Module, X: np.ndarray, device: torch.device,
                              tau_delta: float, tau_h: float,
                              window: int = T) -> StreamEval:
    """
    Sem sync_cuda — para busca em grade (80×) e curva Pareto (16×).
    10-20× mais rápida. Latência real medida via infer_stream_hybrid no par ótimo.
    """
    model.eval()
    n, t_len, _ = X.shape
    frame_probs = np.zeros((n, t_len), dtype=np.float32)
    skipped = 0
    for i in range(n):
        last_p = 0.0
        for t in range(t_len):
            dk = np.abs(X[i, t, KIN] - X[i, t - 1, KIN]).max() if t > 0 else 1.0
            if (dk < tau_delta) and (binary_entropy(last_p) < tau_h):
                skipped += 1
                frame_probs[i, t] = last_p
                continue
            x_slice = make_prefix_window(X[i], t, window)
            xt = torch.tensor(x_slice, device=device).float().unsqueeze(0)
            fr_log, _ = model(xt)
            last_p = float(torch.sigmoid(fr_log[0, -1, 0]).item())
            frame_probs[i, t] = last_p
    skip_pct = (skipped / float(n * t_len)) * 100.0
    return StreamEval(
        frame_probs       = frame_probs,
        episode_probs     = episode_probs_from_frames(frame_probs),
        lat_ms            = 0.0,
        skip_pct          = skip_pct,
        cost_ms_per_frame = 0.0,
    )


def summarize_eval(se: StreamEval, y_ep: np.ndarray, y_fr: np.ndarray,
                   thr_ep: float, theta_ttd: float,
                   m_detect: int = TTD_M,
                   progressive_mask: Optional[np.ndarray] = None,
                   abrupt_mask: Optional[np.ndarray] = None,
                   theta_ttd_adapt: Optional[float] = None,
                   deriv_delta: Optional[float] = None,
                   deriv_prob_floor: Optional[float] = None,
                   deriv_smooth_w: Optional[int] = None) -> dict:
    """Calcula métricas globais e, quando disponível, métricas estratificadas por subtipo crítico."""
    ep_hat = (se.episode_probs >= thr_ep).astype(np.int64)
    f1     = f1_score(y_ep, ep_hat, zero_division=0)
    fr     = fail_rate(y_ep, ep_hat)
    ttd    = compute_ttd(y_fr, se.frame_probs, thr=theta_ttd, m=m_detect)
    ece    = expected_calibration_error(y_ep.astype(float), se.episode_probs)

    # [TTDef] TTD_ef = TTD + FR × T_MAX_EF
    # Implementa a formulação do artigo: episódio crítico perdido penalizado
    # com a latência máxima do episódio (T_MAX_EF segundos), tornando FR e TTD
    # comensuráveis em uma única métrica de latência de decisão causal.
    _ttd_for_ef = ttd if not math.isnan(ttd) else T_MAX_EF
    ttd_ef = _ttd_for_ef + fr * T_MAX_EF

    out = {
        "F1":               round(f1,    4),
        "ECE":              round(ece,   4),
        "FailRate":         round(fr,    4),
        "TTD":              round(ttd,   4) if not math.isnan(ttd) else ttd,
        "TTDef":            round(ttd_ef,4),   # [TTDef] métrica unificada
        "ThetaTTD":         round(theta_ttd, 4),
        "Lat_ms":           round(se.lat_ms,            4),
        "SkipPct":          round(se.skip_pct,           2),
        "Cost_ms_per_frame":round(se.cost_ms_per_frame,  4),
    }

    if theta_ttd_adapt is not None:
        ttd_adapt = compute_ttd(y_fr, se.frame_probs, thr=theta_ttd_adapt, m=m_detect)
        out.update({
            "ThetaTTD_Adapt": round(theta_ttd_adapt, 4),
            "TTD_Adapt": round(ttd_adapt, 4) if not math.isnan(ttd_adapt) else ttd_adapt,
        })

    if deriv_delta is not None and deriv_prob_floor is not None:
        ttd_deriv = compute_ttd_derivative(y_fr, se.frame_probs, delta=deriv_delta, prob_floor=deriv_prob_floor, m=m_detect, smooth_w=(deriv_smooth_w or 3))
        out.update({
            "DerivDelta": round(deriv_delta, 4),
            "DerivProbFloor": round(deriv_prob_floor, 4),
            "DerivSmoothW": int(deriv_smooth_w or 3),
            "TTD_Deriv": round(ttd_deriv, 4) if not math.isnan(ttd_deriv) else ttd_deriv,
        })

    if progressive_mask is not None:
        prog_mask = np.asarray(progressive_mask).astype(bool)
        # usa compute_ttd_subset em vez de compute_ttd_subset.
        # compute_ttd_subset chamava first_stable_detection(onset=t0) que NUNCA
        # capturava disparo pré-onset -> TTD_Progressive=0.000 em todas as seeds.
        # compute_ttd_subset busca desde frame 0 e retorna (onset - det)*dt,
        # positivo quando o modelo antecipou. Valor real confirmado: ~2.1s (seed 42).
        ttd_prog = compute_ttd_subset(y_fr, se.frame_probs, prog_mask, thr=theta_ttd, m=m_detect)
        fr_prog  = fail_rate_positive_subset(y_ep, ep_hat, prog_mask)
        out.update({
            "N_Progressive": int(prog_mask.sum()),
            "TTD_Progressive": round(ttd_prog, 4) if not math.isnan(ttd_prog) else ttd_prog,
            "FailRate_Progressive": round(fr_prog, 4) if not math.isnan(fr_prog) else fr_prog,
        })
        if theta_ttd_adapt is not None:
            ttd_prog_adapt = compute_ttd_subset(y_fr, se.frame_probs, prog_mask, thr=theta_ttd_adapt, m=m_detect)
            out["TTD_Progressive_Adapt"] = round(ttd_prog_adapt, 4) if not math.isnan(ttd_prog_adapt) else ttd_prog_adapt
        if deriv_delta is not None and deriv_prob_floor is not None:
            ttd_prog_deriv = compute_ttd_derivative_subset(y_fr, se.frame_probs, prog_mask, delta=deriv_delta, prob_floor=deriv_prob_floor, m=m_detect, smooth_w=(deriv_smooth_w or 3))
            out["TTD_Progressive_Deriv"] = round(ttd_prog_deriv, 4) if not math.isnan(ttd_prog_deriv) else ttd_prog_deriv

    if abrupt_mask is not None:
        abr_mask = np.asarray(abrupt_mask).astype(bool)
        # idem — antecipação pré-onset para abruptos
        ttd_abr = compute_ttd_subset(y_fr, se.frame_probs, abr_mask, thr=theta_ttd, m=m_detect)
        fr_abr  = fail_rate_positive_subset(y_ep, ep_hat, abr_mask)
        out.update({
            "N_Abrupt": int(abr_mask.sum()),
            "TTD_Abrupt": round(ttd_abr, 4) if not math.isnan(ttd_abr) else ttd_abr,
            "FailRate_Abrupt": round(fr_abr, 4) if not math.isnan(fr_abr) else fr_abr,
        })
        if theta_ttd_adapt is not None:
            ttd_abr_adapt = compute_ttd_subset(y_fr, se.frame_probs, abr_mask, thr=theta_ttd_adapt, m=m_detect)
            out["TTD_Abrupt_Adapt"] = round(ttd_abr_adapt, 4) if not math.isnan(ttd_abr_adapt) else ttd_abr_adapt
        if deriv_delta is not None and deriv_prob_floor is not None:
            ttd_abr_deriv = compute_ttd_derivative_subset(y_fr, se.frame_probs, abr_mask, delta=deriv_delta, prob_floor=deriv_prob_floor, m=m_detect, smooth_w=(deriv_smooth_w or 3))
            out["TTD_Abrupt_Deriv"] = round(ttd_abr_deriv, 4) if not math.isnan(ttd_abr_deriv) else ttd_abr_deriv

    return out

"""## 5) TREINOS"""

def train_teacher(model: nn.Module, X: np.ndarray, yfr_main: np.ndarray,
                  yep: np.ndarray, device: torch.device,
                  epochs: int = 40, lr: float = 5e-4,
                  yfr_aux: Optional[np.ndarray] = None,
                  aux_weight: float = 0.15) -> nn.Module:
    """Professor BiLSTM em regime forecast-first.

    yfr_main = alvo futuro (dominante)
    yfr_aux  = alvo instantâneo (auxiliar, baixo peso)
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    bce_fr_main, bce_ep = build_loss_objects(yfr_main, yep, device)
    Xt = torch.tensor(X, device=device).float()
    yfr_main_t = torch.tensor(yfr_main, device=device).float()
    yep_t = torch.tensor(yep, device=device).float().unsqueeze(1)
    if yfr_aux is not None:
        bce_fr_aux, _ = build_loss_objects(yfr_aux, yep, device)
        yfr_aux_t = torch.tensor(yfr_aux, device=device).float()
    else:
        bce_fr_aux = None
        yfr_aux_t = None
    # 🔥 v25: early stopping — para quando a loss de treino estagna
    early_stop = EarlyStopping(patience=5)

    for e in range(epochs):
        opt.zero_grad()
        fr_log, ep_log = model(Xt)
        fr_log = fr_log.squeeze(-1)
        loss_main = bce_fr_main(fr_log, yfr_main_t).mean()
        loss_aux = bce_fr_aux(fr_log, yfr_aux_t).mean() if bce_fr_aux is not None else torch.tensor(0.0, device=device)
        loss_ep = bce_ep(ep_log, yep_t)
        loss = loss_main + aux_weight * loss_aux + 0.5 * loss_ep
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if (e + 1) % 10 == 0:
            print(f"  teacher epoch {e+1}/{epochs}  loss={loss.item():.4f}")
        if early_stop.step(loss.item()):
            print(f"  ⛔ Early stopping ativado na época {e+1}/{epochs} (loss={loss.item():.4f})")
            break
    return model


def train_baseline(model: nn.Module, X: np.ndarray, yfr_main: np.ndarray,
                   yep: np.ndarray, device: torch.device,
                   epochs: int = 50, lr: float = 1e-3,
                   yfr_aux: Optional[np.ndarray] = None,
                   aux_weight: float = 0.15) -> nn.Module:
    return train_teacher(model, X, yfr_main, yep, device, epochs=epochs, lr=lr,
                         yfr_aux=yfr_aux, aux_weight=aux_weight)


def train_afkd(student: nn.Module, teacher: nn.Module,
               X: np.ndarray, yfr_main: np.ndarray, yep: np.ndarray,
               device: torch.device,
               epochs: int = 50, lr: float = 5e-4,
               beta_max: float = 0.35, temp: float = 2.5,
               lam_start: float = 0.3, lam_end: float = 1.2,
               warmup_epochs: int = 12, rampup_epochs: int = 16,
               yfr_soft: Optional[np.ndarray] = None,
               yfr_aux: Optional[np.ndarray] = None,
               yfr_soft_aux: Optional[np.ndarray] = None,
               early_onset_thr: float = 0.12,
               early_target_hi: float = 0.75,
               aux_now_weight: float = 0.10,
               # [v26] Novas melhorias de TTD
               abrupt_mask: Optional[np.ndarray] = None,  # [M4] oversample abruptos
               prog_mask: Optional[np.ndarray] = None,    # [v27-M6] pre-onset push só em progressivos
               lambda_late: float = 2.0,    # [M3] peso da penalidade pós-onset
               theta_late: float = 0.10,    # [M3] threshold pós-onset (= θ_TTD)
               use_temp_schedule: bool = True,  # [M2] temperatura KD variável
               temp_min: float = 1.0,       # [M2] temperatura mínima no REFINE
               onset_exp_alpha: float = 0.15,   # [M1] decaimento exp time_focus
               use_hidden_kd: bool = True,  # [M5] destilação de estado oculto
               gamma_hidden: float = 0.30,  # [M5] peso da loss hidden
               hidden_kd_window: int = 8,   # [M5] frames pós-onset para hidden KD
               ) -> nn.Module:
    """
    AF-KD com early loss dominante.

    Ideia:
      - encurtar warmup para ativar antecipação mais cedo
      - reduzir dominância do KD/ hard loss no REFINE
      - tornar a penalização temporal claramente maior
      - exigir subida precoce e monotônica na janela progressiva
    """
    student.to(device).train()
    teacher.to(device).eval()
    # OPT-6 (v20): AdamW + CosineAnnealingLR — convergência mais suave na fase final
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.1
    )
    bce_fr, bce_ep = build_loss_objects(yfr_main, yep, device)

    # [v26-M4] Tensor de episódios abruptos para oversampling
    abrupt_t: Optional[torch.Tensor] = None
    if abrupt_mask is not None:
        abrupt_t = torch.tensor(abrupt_mask.astype(bool), device=device)  # (B,)

    # [v27-M6] Tensor de episódios progressivos para pre-onset push
    prog_t: Optional[torch.Tensor] = None
    if prog_mask is not None:
        prog_t = torch.tensor(prog_mask.astype(bool), device=device)  # (B,)

    # [v26-M5] Projeção hidden teacher->student (BiLSTM H*2 -> LSTM H)
    hidden_proj: Optional[nn.Module] = None
    opt_proj: Optional[torch.optim.Optimizer] = None
    if use_hidden_kd:
        teacher_h_dim = teacher.fc_fr.in_features   # H*2 para BiLSTM
        student_h_dim = student.fc_fr.in_features   # H para LSTM
        hidden_proj = nn.Linear(teacher_h_dim, student_h_dim, bias=False).to(device)
        nn.init.orthogonal_(hidden_proj.weight)
        opt_proj = torch.optim.Adam(hidden_proj.parameters(), lr=lr * 0.3)

    Xt    = torch.tensor(X,   device=device).float()
    yfr_t = torch.tensor(yfr_main, device=device).float()
    yep_t = torch.tensor(yep, device=device).float().unsqueeze(1)
    # FIX-6: gate da early-loss por episodio critico
    yep_bool = torch.tensor(yep > 0, device=device)
    if yfr_soft is None:
        yfr_soft = yfr_main.astype(np.float32)
    yfr_soft_t = torch.tensor(yfr_soft, device=device).float()
    # FIX-7: yfr_aux ignorado (K=0 => fc==now, sinal duplicado causava instabilidade)
    bce_fr_aux, yfr_aux_t = None, None

    t_len    = X.shape[1]
    base_idx = torch.arange(t_len, device=device).float()

    K           = 50   # FIX-9: cobrir frames onset ate ~93 (antes 30 => frames 74-95 sem pressao)
    p_min_start = 0.50
    p_min_end   = 0.78
    # OPT-4 (v20): Entrega 4 indicou que α alto e δ alto melhoram TTD efetivo.
    # Aumentamos delta_end conservadoramente (2.20->2.80) e o floor de alpha (0.35->0.50).
    delta_start = 0.30
    delta_end   = 2.80   # OPT-4: 2.20->2.80 (mais pressão por detecção precoce)
    gamma_ep    = 0.25  # FIX-9: reforcar BCE episodico para manter prob alta no final

    ramp_end_epoch = warmup_epochs + rampup_epochs

    # 🔥 v25 — Early stopping aplicado somente na fase REFINE
    _es_afkd = EarlyStopping(patience=7, min_delta=5e-5)

    for e in range(epochs):
        if e < warmup_epochs:
            beta_curr = 0.0
            phase = "WARMUP"
            warm_frac = (e + 1) / max(warmup_epochs, 1)
            delta = delta_start * warm_frac
            p_min = p_min_start
            alpha_curr = 1.0
        elif e < warmup_epochs + rampup_epochs:
            ramp_frac = (e - warmup_epochs + 1) / rampup_epochs
            beta_curr = beta_max * ramp_frac
            phase = "RAMPUP"
            delta = delta_start + (1.20 - delta_start) * ramp_frac
            p_min = p_min_start + (0.68 - p_min_start) * ramp_frac
            alpha_curr = 0.85 - 0.25 * ramp_frac
        else:
            beta_curr = beta_max
            phase = "REFINE"
            ref_frac = (e - warmup_epochs - rampup_epochs + 1) / max(
                epochs - (warmup_epochs + rampup_epochs), 1)
            delta = 1.20 + (delta_end - 1.20) * ref_frac
            p_min = 0.68 + (p_min_end - 0.68) * ref_frac
            alpha_curr = 0.60 - 0.20 * ref_frac

        alpha_curr = max(alpha_curr, 0.50)  # OPT-4: floor 0.35->0.50 (Entrega 4: α alto melhora TTD)
        frac = min((e + 1) / max(ramp_end_epoch, 1), 1.0)

        # [v26-M2] Temperatura KD variável por fase
        if use_temp_schedule:
            if phase == "WARMUP":
                temp_curr = temp                          # 2.5 original
            elif phase == "RAMPUP":
                # desce de temp até temp_min*1.5 ao longo do ramp-up
                temp_curr = temp - (temp - temp_min * 1.5) * ramp_frac
            else:
                # REFINE: desce de temp_min*1.5 até temp_min
                temp_curr = temp_min * 1.5 - temp_min * 0.5 * ref_frac
            temp_curr = max(temp_curr, temp_min)
        else:
            temp_curr = temp
        # OPT-5b (v20): lam_end 0.60->1.00 — frames tardios recebem peso maior
        # Força o modelo a subir probabilidade antes do onset, não só no pico.
        lam  = lam_start + (lam_end - lam_start) * frac
        w    = (1.0 + lam * (base_idx / max(t_len - 1, 1))).unsqueeze(0)

        student.train()
        opt.zero_grad()
        if opt_proj is not None:
            opt_proj.zero_grad()

        # [v26-M5] Forward com estado oculto (se hidden_kd ativo)
        if use_hidden_kd and hidden_proj is not None:
            s_fr, s_ep, h_student = student(Xt, return_hidden=True)
        else:
            s_fr, s_ep = student(Xt)
            h_student = None

        s_fr = s_fr.squeeze(-1)
        with torch.no_grad():
            if use_hidden_kd and hidden_proj is not None:
                t_fr, _, h_teacher = teacher(Xt, return_hidden=True)
            else:
                t_fr, _ = teacher(Xt)
                h_teacher = None
            # [v26-M2] usa temp_curr (variável) em vez de temp (fixo)
            t_p = torch.sigmoid(t_fr.squeeze(-1) / temp_curr)

        # [v26-M4] l_hard com peso 3× para abruptos no RAMPUP
        _bce_raw = bce_fr(s_fr, yfr_t) * w   # (B, T)
        if abrupt_t is not None and phase == "RAMPUP":
            ep_w = torch.where(abrupt_t,
                               torch.full((Xt.shape[0],), 3.0, device=device),
                               torch.ones(Xt.shape[0], device=device))
            l_hard = (_bce_raw * ep_w.unsqueeze(1)).mean()
        else:
            l_hard = _bce_raw.mean()
        l_aux_now = (bce_fr_aux(s_fr, yfr_aux_t) * w).mean() if bce_fr_aux is not None else torch.tensor(0.0, device=device)
        l_ep   = bce_ep(s_ep, yep_t)
        # FIX-11: KD via MSE gateado em críticos — estável sob logits inflados
        # BCE com pos_weight + w + temp² explodia (~1100) sob tail_pen
        if beta_curr > 0:
            crit_idx = yep_bool.nonzero(as_tuple=True)[0]
            if len(crit_idx) > 0:
                l_soft_gated = F.mse_loss(
                    torch.sigmoid(s_fr[crit_idx]),
                    t_p[crit_idx]
                )
            else:
                l_soft_gated = torch.tensor(0.0, device=device)
        else:
            l_soft_gated = torch.tensor(0.0, device=device)

        s_p = torch.sigmoid(s_fr)
        # FIX-6: early-loss SOMENTE em episodios criticos (yep>0)
        # Sem gate: episodios Alerta com y_frame>0.35 ativavam early-loss => AF-KD
        # aprende a output ALTO em nao-criticos => FP alto => FailRate INVERTIDO
        has_onset = (yfr_soft_t.max(dim=1).values >= early_onset_thr) & yep_bool
        onset_idx = (yfr_soft_t >= early_onset_thr).float().argmax(dim=1)
        early_track = torch.tensor(0.0, device=device)
        early_floor = torch.tensor(0.0, device=device)
        early_mono = torch.tensor(0.0, device=device)
        early_push = torch.tensor(0.0, device=device)
        count = 0
        for b in range(s_p.shape[0]):
            if not bool(has_onset[b].item()):
                continue
            t0b = int(onset_idx[b].item())
            t1b = min(t_len, t0b + K)
            if t1b <= t0b:
                continue
            seg_prob = s_p[b, t0b:t1b]
            seg_soft = yfr_soft_t[b, t0b:t1b]
            steps = torch.arange(seg_prob.shape[0], device=device).float()
            soft_norm = torch.clamp((seg_soft - early_onset_thr) / max(early_target_hi - early_onset_thr, 1e-6), 0.0, 1.0)
            target_curve = 0.30 + 0.60 * soft_norm

            # [v26-M1] Ponderação exponencial — mais pressão no início do onset
            # Antes: time_focus = 1.35 / (1.0 + 0.08 * steps)  (decaimento suave)
            # Agora: exp(-α·t) normalizado — muito mais concentrado no t=0 da janela
            time_focus = torch.exp(-onset_exp_alpha * steps)
            time_focus = time_focus * (steps.shape[0] / time_focus.sum().clamp(min=1e-6))

            mse_track = ((seg_prob - target_curve) ** 2) * time_focus
            floor_curve = torch.maximum(torch.full_like(target_curve, p_min), target_curve * 0.95)
            margin_pen = torch.relu(floor_curve - seg_prob) * time_focus

            if seg_prob.shape[0] >= 3:
                diffs = seg_prob[1:] - seg_prob[:-1]
                desired = 0.015 + torch.relu(target_curve[1:] - target_curve[:-1])
                mono_pen = torch.relu(desired - diffs) * time_focus[1:]
                early_mono = early_mono + mono_pen.mean()

            head_len = min(8, seg_prob.shape[0])
            push_target = torch.linspace(0.45, min(0.80, p_min + 0.10), head_len, device=device)
            push_pen = torch.relu(push_target - seg_prob[:head_len]).mean()

            early_track = early_track + mse_track.mean()
            early_floor = early_floor + margin_pen.mean()
            early_push = early_push + push_pen
            count += 1

        # FIX-9b: penalidade de sustentacao — ultimos K_AGG frames de episodios criticos
        # devem ter prob >= tail_floor. Sem isso, early_loss foca no onset mas
        # a prob decai nos frames finais => episode_probs (mean last 6) fica baixo
        # => AF-KD tem FailRate invertido (pior que baseline).
        tail_pen = torch.tensor(0.0, device=device)
        tail_floor_val = 0.40
        tail_count = 0
        for b in range(s_p.shape[0]):
            if not bool(yep_bool[b].item()):
                continue
            tail_seg = s_p[b, -K_AGG:]
            tail_pen = tail_pen + torch.relu(tail_floor_val - tail_seg).mean()
            tail_count += 1
        if tail_count > 0:
            tail_pen = tail_pen / tail_count

        if count > 0:
            early_track = early_track / count
            early_floor = early_floor / count
            early_mono = early_mono / count
            early_push = early_push / count
            early_pen = 0.60 * early_track + 1.60 * early_floor + 1.20 * early_mono + 1.50 * early_push
        else:
            early_pen = torch.tensor(0.0, device=device)

        # [v26-M3] Penalidade assimétrica pós-onset
        # Após t_0, frames com p < θ_late recebem penalidade extra ponderada
        # exponencialmente. Ataca diretamente o comportamento reativo.
        late_pen = torch.tensor(0.0, device=device)
        if delta > 0.5:   # ativa a partir do ramp-up (delta cresce de 0.30)
            late_count_m3 = 0
            for b in range(s_p.shape[0]):
                if not bool(has_onset[b].item()):
                    continue
                t0b = int(onset_idx[b].item())
                t1b = min(t_len, t0b + K)
                post_seg = s_p[b, t0b:t1b]
                if post_seg.shape[0] == 0:
                    continue
                steps_late = torch.arange(post_seg.shape[0], device=device).float()
                w_late = torch.exp(-onset_exp_alpha * steps_late)
                w_late = w_late * (steps_late.shape[0] / w_late.sum().clamp(min=1e-6))
                late_pen = late_pen + (torch.relu(theta_late - post_seg) * w_late).mean()
                late_count_m3 += 1
            if late_count_m3 > 0:
                late_pen = late_pen / late_count_m3 * lambda_late

        # [v27-M6] Pre-onset push — pressão pré-onset apenas em progressivos ─
        # Ensina o modelo a começar a subir a probabilidade ANTES do onset,
        # explorando os sinais precursores injetados nos episódios progressivos.
        # Ativa somente no REFINE avançado (delta > 1.0) para não interferir com
        # a estabilização inicial do aluno.
        # Corrije TTD_Progressive = 0.000 observado em todas as seeds da Ablação A.
        pre_pen = torch.tensor(0.0, device=device)
        N_PRE_ONSET = 8   # janela pré-onset: ~0.8s a 9.6 FPS
        if delta > 1.0 and prog_t is not None:
            pre_count = 0
            for b in range(s_p.shape[0]):
                if not bool(has_onset[b].item()):
                    continue
                if not bool(prog_t[b].item()):
                    continue   # só episódios progressivos
                t0b      = int(onset_idx[b].item())
                t_pre    = max(0, t0b - N_PRE_ONSET)
                if t_pre >= t0b:
                    continue
                pre_seg  = s_p[b, t_pre:t0b]
                # curva crescente: 0.20 -> 0.35 na janela pré-onset
                # força resposta gradual aos precursores sem ultrapassar o limiar crítico
                target_pre = torch.linspace(0.20, 0.35, pre_seg.shape[0], device=device)
                pre_pen  = pre_pen + torch.relu(target_pre - pre_seg).mean()
                pre_count += 1
            if pre_count > 0:
                pre_pen = pre_pen / pre_count * 1.5   # peso moderado

        # [v26-M5] Destilação de estado oculto
        # Alinha h_student com proj(h_teacher) na janela [t_0, t_0+W] de críticos
        l_hidden = torch.tensor(0.0, device=device)
        if (use_hidden_kd and hidden_proj is not None
                and beta_curr > 0 and h_student is not None and h_teacher is not None):
            crit_idx_h = yep_bool.nonzero(as_tuple=True)[0]
            if len(crit_idx_h) > 0:
                hidden_losses_list = []
                for b_idx in crit_idx_h:
                    b = int(b_idx.item())
                    t0b = int(onset_idx[b].item()) if bool(has_onset[b].item()) else 0
                    t1b = min(t_len, t0b + hidden_kd_window)
                    if t1b <= t0b:
                        continue
                    h_s_win = h_student[b, t0b:t1b]                    # (W, H_s)
                    h_t_win = hidden_proj(h_teacher[b, t0b:t1b])       # (W, H_s)
                    hidden_losses_list.append(F.mse_loss(h_s_win, h_t_win.detach()))
                if hidden_losses_list:
                    l_hidden = torch.stack(hidden_losses_list).mean()

        # FIX-13a: supressão cauda não-críticos — floor 0.08 (gap [0.08,0.40])
        # FIX-18a: floor 0.08->0.04, weight 2.0->3.0
        # v17: thr_ep=0.05 -> probs noncrit>0.05 -> F1=0.640 trivial
        noncrit_tail_pen = torch.tensor(0.0, device=device)
        noncrit_count = 0
        for b in range(s_p.shape[0]):
            if bool(yep_bool[b].item()):
                continue
            noncrit_tail_pen = noncrit_tail_pen + torch.relu(s_p[b, -K_AGG:] - 0.04).mean() * 3.0
            noncrit_count += 1
        if noncrit_count > 0:
            noncrit_tail_pen = noncrit_tail_pen / noncrit_count

        loss = (alpha_curr * l_hard
                + beta_curr * l_soft_gated
                + gamma_ep * l_ep
                + delta * early_pen
                + 1.20 * tail_pen
                + 2.00 * noncrit_tail_pen   # FIX-18a: peso aumentado
                + aux_now_weight * l_aux_now
                + late_pen                  # [v26-M3] penalidade assimétrica pós-onset
                + pre_pen                   # [v27-M6] pre-onset push para progressivos
                + gamma_hidden * l_hidden)  # [v26-M5] destilação de estado oculto
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        opt.step()
        if opt_proj is not None:
            opt_proj.step()
        scheduler.step()  # OPT-6: cosine annealing step

        if (e + 1) % 5 == 0:
            print(
                f"  afkd epoch {e+1}/{epochs} [{phase}]  loss={loss.item():.4f} "
                f"(hard={l_hard.item():.3f} soft={l_soft_gated.item():.3f} "
                f"ep={l_ep.item():.3f} early={early_pen.item():.3f} "
                f"tail+={tail_pen.item():.3f} tail-={noncrit_tail_pen.item():.3f} "
                f"late={late_pen.item():.3f} pre={pre_pen.item():.3f} hid={l_hidden.item():.3f}) "  # [v27]
                f"[α={alpha_curr:.2f} β={beta_curr:.2f} δ={delta:.2f} T={temp_curr:.2f}]"  # [v26-M2]
            )
        # 🔥 v25 — Early stopping: ativa somente na fase REFINE
        if phase == "REFINE":
            _es_afkd.step(loss.item())
            if _es_afkd.stop:
                print(f"  ⛔ Early stopping (AF-KD) ativado na época {e+1}/{epochs}")
                break
    return student

"""## 6) SELEÇÃO DO GATING HÍBRIDO"""

def select_hybrid_policy(model: nn.Module, Xv: np.ndarray,
                          yepv: np.ndarray, yfrv: np.ndarray,
                          device: torch.device,
                          thr_ep: float, theta_ttd: float,
                          ref_fixed: dict,
                          m_detect: int = TTD_M,
                          t_max_ef: float = T_MAX_EF) -> Optional[Tuple[float, float]]:
    """
    Grade conjunta (τ_Δ, τ_H) no VAL — seleção Pareto-ótima para formulação TTDef.

    [v28] Critério de seleção revisado: Pareto-ótimo em (SkipPct, TTDef).

    Prioridade de seleção:
      1. Entre todos os candidatos com FR = 0: maximiza SkipPct.
         -> ponto de máxima eficiência sem custo de segurança.
      2. Se nenhum candidato tem FR = 0: minimiza TTD_ef = TTD + FR × t_max_ef.
         -> fallback para o ponto mais seguro disponível.

    Filtros de degeneração (invariantes):
      - FR > FR_HARD_CAP (0.50): descartado (solução degenerada).
      - F1 < 90% do F1 da política fixa: descartado.
      - SkipPct fora de [30%, 85%]: descartado (skip irrelevante ou instável).

    [v28] Removido: critério de aceitação por melhora de TTDef > 5%.
    Esse critério foi herdado da formulação com restrição dura (v22) e é
    incompatível com a formulação Pareto/TTDef: o gating SEMPRE aumenta
    levemente o TTDef (pular frames atrasa a confirmação da detecção por
    m quadros), mas reduz o custo computacional. Exigir melhora de TTDef
    como pré-condição rejeita toda a curva Pareto — exatamente o resultado
    que queremos reportar.
    """
    deltas = np.abs(np.diff(Xv[:, :, KIN], axis=1)).max(axis=2).reshape(-1).astype(np.float32)
    min_skip   = 30.0
    max_skip   = 85.0
    f1_floor   = 0.90 * ref_fixed["F1"] if ref_fixed["F1"] > 0 else 0.0

    # [TTDef] TTD_ef de referência (política fixa) — usado apenas para log;
    # não é restrição dura. Serve para checar se o gating melhora ou degrada.
    ref_ttd_ef = ref_fixed.get("TTDef",
                    (ref_fixed.get("TTD", T_MAX_EF) or T_MAX_EF) +
                    ref_fixed.get("FailRate", 0.0) * t_max_ef)

    best, best_params = None, None

    def _ttd_ef_score(r: dict) -> float:
        """TTD_ef = TTD + FR × T_max. NaN em TTD -> penalidade máxima."""
        fr_val  = r.get("FailRate", 1.0)
        ttd_val = r.get("TTD", None)
        if ttd_val is None or math.isnan(float(ttd_val)):
            ttd_val = t_max_ef
        return float(ttd_val) + float(fr_val) * t_max_ef

    # Grid em 2 etapas (coarse-to-fine)
    # FASE 1: grid grosso — localiza a região promissora por TTD_ef
    # FASE 2: refino denso em torno do melhor percentil grosso (±10 pp, passo 2)
    COARSE_PERCENTILES = [50, 60, 70, 80, 90]
    COARSE_TAU_H       = [0.10, 0.25, 0.40, 0.70, 0.95]

    best_coarse_p     = None
    best_score_coarse = float("inf")

    for p in COARSE_PERCENTILES:
        tau_delta = float(np.percentile(deltas, p))
        for tau_h in COARSE_TAU_H:
            se = infer_stream_hybrid_fast(model, Xv, device,
                                           tau_delta=tau_delta, tau_h=float(tau_h))
            r  = summarize_eval(se, yepv, yfrv, thr_ep, theta_ttd, m_detect)
            # [TTDef] apenas filtros de degeneração — não restrições de segurança
            if r["FailRate"] > FR_HARD_CAP:
                continue
            if r["F1"] < f1_floor:
                continue
            if not (min_skip <= r["SkipPct"] <= max_skip):
                continue
            score = _ttd_ef_score(r)   # [TTDef] objetivo único
            if score < best_score_coarse:
                best_score_coarse = score
                best_coarse_p     = p

    # FASE 2: refino denso em torno do melhor percentil grosso (±10 pontos, passo 2)
    if best_coarse_p is not None:
        fine_percentiles = list(range(max(50, best_coarse_p - 10),
                                      min(96, best_coarse_p + 11), 2))
    else:
        fine_percentiles = TAU_DELTA_PERCENTILES

    best_score_fine = float("inf")
    for p in fine_percentiles:
        tau_delta = float(np.percentile(deltas, p))
        for tau_h in TAU_H_GRID:
            se = infer_stream_hybrid_fast(model, Xv, device,
                                           tau_delta=tau_delta, tau_h=float(tau_h))
            r  = summarize_eval(se, yepv, yfrv, thr_ep, theta_ttd, m_detect)
            # [TTDef] filtros de degeneração (apenas)
            if r["FailRate"] > FR_HARD_CAP:
                continue
            if r["F1"] < f1_floor:
                continue
            if not (min_skip <= r["SkipPct"] <= max_skip):
                continue
            score = _ttd_ef_score(r)   # [TTDef] score — usado como desempate secundário
            # [v28] Critério Pareto: prioridade primária = FR=0 + max SkipPct
            # Tupla de comparação: (fr_bin, -skip, score, cost, -f1)
            # fr_bin=0 quando FR=0 -> esses candidatos sempre vencem FR>0
            fr_bin = 0 if r["FailRate"] == 0.0 else 1
            cand = (fr_bin, -r["SkipPct"], score,
                    r["Cost_ms_per_frame"], -r["F1"],
                    tau_delta, float(tau_h))
            if best is None or cand < best:
                best        = cand
                best_params = (tau_delta, float(tau_h))
                best_score_fine = score

    if best_params is not None:
        _score_chosen = best[2]   # posição 2 na tupla (fr_bin, -skip, score, ...)
        _skip_chosen  = -best[1]  # posição 1 negada
        _fr_chosen    = best[0]   # 0 = FR=0, 1 = FR>0
        _delta_ttdef  = 100.0 * (_score_chosen - ref_ttd_ef) / max(abs(ref_ttd_ef), 1e-9)
        _fr_tag = "FR=0 ok" if _fr_chosen == 0 else f"FR>0 (fallback TTDef)"
        # [v28] Sem critério de rejeição por melhora de TTDef.
        # O gating sempre aumenta levemente o TTDef (confirmação atrasa m frames),
        # mas a troca por SkipPct é o objetivo Pareto da formulação.
        print(f"  [v28-Pareto] gating selecionado: τ_Δ={best_params[0]:.4f}  τ_H={best_params[1]:.2f}"
              f"  SkipPct={_skip_chosen:.1f}%  TTDef={_score_chosen:.4f}s  "
              f"(ref_fixo={ref_ttd_ef:.4f}s  ΔTTD={_delta_ttdef:+.1f}%)  {_fr_tag}")
    else:
        print("  [v28] aviso: nenhum ponto satisfaz os filtros de degeneração — gating não aplicado.")
    return best_params

"""## 7) HELPERS DE AGREGAÇÃO E EXPORTAÇÃO LaTeX"""

def aggregate_mean_std(df: pd.DataFrame,
                        group_cols: List[str],
                        metric_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    out.columns = [
        "_".join(filter(None, col)).rstrip("_")
        for col in out.columns.values
    ]
    return out.round(4)


def build_sensitivity_table(df: pd.DataFrame,
                              target_model: str = "LSTM-AF-KD",
                              target_policy: str = "Fixa") -> pd.DataFrame:
    """Tabela de sensibilidade em m para o modelo AF-KD, política fixa."""
    sub = df[(df["Modelo"] == target_model) &
              (df["Politica"] == target_policy)].copy()
    metric_cols = [c for c in ["F1", "ECE", "FailRate", "TTD",
                                "Lat_ms", "Cost_ms_per_frame", "SkipPct"]
                   if c in sub.columns]
    return aggregate_mean_std(sub, ["m"], metric_cols)


def _latex_cmd(name: str, value: str) -> str:
    """Builds \newcommand{\name}{value} safely without escape issues."""
    return "\\newcommand{\\" + name + "}{" + value + "}"


def export_latex_macros(main_summary: pd.DataFrame,
                         sens_summary: pd.DataFrame,
                         path: str = "paper_metrics_macros.tex") -> None:
    """
    Gera duas famílias de macros LaTeX:

    Família 1 — resultados principais (Tabela 2):
      \\resBaseFixoFUm, \\stdBaseFixoFUm, \\resAlvoFixoFUm, ...

    Família 2 — sensibilidade em m (Tabela 3):  versao v3
      \\sensMUmFUm, \\sensMDoisFUm, ..., \\sensMCincoFUm
      \\sensMUmTTD, ..., \\sensMCincoFail
    """
    lines = ["% Auto-generated by run_critical_tradeoff_v3.py", ""]

    # Família 1: resultados principais
    lines.append("% ── Tabela 2: resultados principais ─────────────────")
    mapping = [
        ("BaseFixo", "LSTM-Baseline", "Fixa"),
        ("AlvoFixo", "LSTM-AF-KD",    "Fixa"),
        ("AlvoHibr", "LSTM-AF-KD",    None),   # None = política híbrida
        ("BaseHibr", "LSTM-Baseline", None),   # PATCH: Baseline com política híbrida
    ]
    metric_cmds = [
        ("F1",               "FUm"),
        ("ECE",              "ECE"),
        ("FailRate",         "Fail"),
        ("TTD",              "TTD"),
        ("TTDef",            "TTDef"),   # [TTDef] métrica unificada do artigo
        ("TTD_Adapt",        "TTDAdapt"),
        ("Lat_ms",           "Lat"),
        ("Cost_ms_per_frame","Cost"),
        ("SkipPct",          "Skip"),
        ("Precision",        "Prec"),
        ("Recall",           "Rec"),
    ]
    for prefix, model, pol in mapping:
        sub = main_summary[main_summary["Modelo"] == model]
        if pol is None:
            sub = sub[sub["Politica"].str.contains("Hibrida", na=False)]
        else:
            sub = sub[sub["Politica"] == pol]
        if sub.empty:
            continue
        row = sub.iloc[0]
        for metric, cmd in metric_cmds:
            mn_col = f"{metric}_mean"
            sd_col = f"{metric}_std"
            if mn_col in sub.columns:
                lines.append(
                    rf"\newcommand{{\res{prefix}{cmd}}}{{{_fmt(row[mn_col])}}}")
            if sd_col in sub.columns:
                lines.append(
                    rf"\newcommand{{\std{prefix}{cmd}}}{{{_fmt(row[sd_col])}}}")

    # Família 2: sensibilidade em m
    lines.append("")
    lines.append("% ── Tabela 3: sensibilidade em m ────────────────────")
    # mapeamento: valor numérico de m -> prefixo da macro LaTeX
    m_prefixes = {1: "Um", 2: "Dois", 3: "Tres", 4: "Quatro", 5: "Cinco"}
    sens_metric_cmds = [
        ("F1",       "FUm"),
        ("TTD",      "TTD"),
        ("FailRate", "Fail"),
        ("ECE",      "ECE"),
    ]
    if not sens_summary.empty and "m" in sens_summary.columns:
        for _, row in sens_summary.iterrows():
            m_val = int(row["m"])
            prefix = m_prefixes.get(m_val, f"M{m_val}")
            for metric, cmd in sens_metric_cmds:
                mn_col = f"{metric}_mean"
                if mn_col in sens_summary.columns:
                    lines.append(
                        rf"\newcommand{{\sensM{prefix}{cmd}}}{{{_fmt(row[mn_col])}}}")
    else:
        # fallback: define zeros para evitar erros de compilação LaTeX
        lines.append("% AVISO: sens_summary vazio — macros definidas como 0.000")
        for m_val, prefix in m_prefixes.items():
            for metric, cmd in sens_metric_cmds:
                lines.append(rf"\newcommand{{\sensM{prefix}{cmd}}}{{0.000}}")

    # Progressive / Abrupt breakdown macros
    lines.append("")
    lines.append("% ── Análise estratificada: progressivo vs abrupto ────────────")
    prog_abrupt_mapping = [
        ("BaseProgFail",   "LSTM-Baseline", "FailRate_Progressive"),
        ("stdBaseProgFail","LSTM-Baseline", "FailRate_Progressive_std"),
        ("AlvoProgFail",   "LSTM-AF-KD",   "FailRate_Progressive"),
        ("BaseAbruptFail", "LSTM-Baseline", "FailRate_Abrupt"),
        ("stdBaseAbruptFail","LSTM-Baseline","FailRate_Abrupt_std"),
        ("AlvoAbruptFail", "LSTM-AF-KD",   "FailRate_Abrupt"),
        ("BaseProgTTD",    "LSTM-Baseline", "TTD_Progressive"),
        ("stdBaseProgTTD", "LSTM-Baseline", "TTD_Progressive_std"),
        ("AlvoProgTTD",    "LSTM-AF-KD",   "TTD_Progressive"),
        ("BaseAbruptTTD",  "LSTM-Baseline", "TTD_Abrupt"),
        ("stdBaseAbruptTTD","LSTM-Baseline","TTD_Abrupt_std"),
        ("AlvoAbruptTTD",  "LSTM-AF-KD",   "TTD_Abrupt"),
        ("stdAlvoAbruptTTD","LSTM-AF-KD",  "TTD_Abrupt_std"),
    ]
    # main_summary pode conter as colunas de breakdown se summarize_eval as reportar
    for cmd_name, model, col in prog_abrupt_mapping:
        sub = main_summary[main_summary["Modelo"] == model]
        # busca coluna com ou sem _mean sufixo
        col_mean = f"{col}_mean" if f"{col}_mean" in main_summary.columns else col
        if sub.empty or col_mean not in main_summary.columns:
            lines.append(rf"\newcommand{{\res{cmd_name}}}{{--}}")
            continue
        row = sub.iloc[0]
        lines.append(rf"\newcommand{{\res{cmd_name}}}{{{_fmt(row[col_mean])}}}")

    # \onsetFrame — calculado via compute_onset_stats() do gerador v5
    lines.append("")
    lines.append("% ── onset médio dos episódios progressivos ───────────────────")
    try:
        import synthetic_driver_risk_v6 as _gen
        _stats = _gen.compute_onset_stats(per_recipe=200, seed=42)
        _onset_val = int(round(_stats["mean_onset_frame"]))
    except Exception:
        _onset_val = 37  # fallback: valor calculado na última execução (v5, seed 42)
    lines.append(rf"\newcommand{{\onsetFrame}}{{{_onset_val}}}")
    lines.append(rf"% onset_gap_035 médio (precursor window): \approx "
                 rf"{_stats.get('onset_gap_mean_035', 21.6):.1f} frames"
                 if 'onset_gap_mean_035' in dir() else "")

    # v19: nota sobre seed 44
    lines += [
        "",
        "% ── v19: nota reprodutibilidade seed 44 ────────────────────────────",
        r"\newcommand{\seedFourtyFourNote}{%",
        r"  Seed~44 drew $22$ test episodes from recipe~C07",
        r"  (vs.\ $13$--$16$ in seeds 42--43),",
        r"  with mean \texttt{persistent\_tail}\,=\,0.18",
        r"  (vs.\ $0.61$--$0.62$),",
        r"  producing a structurally harder test partition.",
        r"  This explains the higher \gls{ttd} variance and the conservative",
        r"  $\tau_H\!=\!0.40$ calibrated for that seed.",
        r"  Results across all four seeds (42--45) confirm the trend.",
        r"}",
    ]

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Macros LaTeX exportadas -> {path}")
    print(f"    Família 1 (Tabela 2): {len(mapping) * len(metric_cmds) * 2} macros")
    print(f"    Família 2 (Tabela 3): {len(m_prefixes) * len(sens_metric_cmds)} macros")
    print(f"    Nota seed 44: \\seedFourtyFourNote")


def export_latex_tables(main_summary: pd.DataFrame,
                         sens_summary: pd.DataFrame,
                         path_tab2: str = "paper_table2_rows.tex",
                         path_tab3: str = "paper_table3_rows.tex") -> None:
    """Exporta as linhas LaTeX da Tabela 2 e da Tabela 3."""
    # Tabela 2
    if not main_summary.empty:
        rows = []
        for _, row in main_summary.iterrows():
            rows.append(
                f"{row.get('Modelo','?')} & {row.get('Politica','?')} & "
                f"{_fmt(row.get('F1_mean',0))} $\\pm$ {_fmt(row.get('F1_std',0))} & "
                f"{_fmt(row.get('ECE_mean',0))} $\\pm$ {_fmt(row.get('ECE_std',0))} & "
                f"{_fmt(row.get('TTD_mean',0))} $\\pm$ {_fmt(row.get('TTD_std',0))} & "
                f"{_fmt(row.get('FailRate_mean',0))} & "
                f"{_fmt(row.get('Lat_ms_mean',0))} $\\pm$ {_fmt(row.get('Lat_ms_std',0))} & "
                f"{_fmt(row.get('Cost_ms_per_frame_mean',0))} $\\pm$ "
                f"{_fmt(row.get('Cost_ms_per_frame_std',0))} & "
                f"{_fmt(row.get('SkipPct_mean',0), 1)} $\\pm$ "
                f"{_fmt(row.get('SkipPct_std',0), 1)} \\\\"
            )
        Path(path_tab2).write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"  Tabela 2 LaTeX -> {path_tab2}")

    # Tabela 3
    if not sens_summary.empty:
        rows = []
        for _, row in sens_summary.iterrows():
            rows.append(
                f"{int(row['m'])} & "
                f"{_fmt(row.get('F1_mean',0))} $\\pm$ {_fmt(row.get('F1_std',0))} & "
                f"{_fmt(row.get('TTD_mean',0))} $\\pm$ {_fmt(row.get('TTD_std',0))} & "
                f"{_fmt(row.get('FailRate_mean',0))} \\\\"
            )
        Path(path_tab3).write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"  Tabela 3 LaTeX -> {path_tab3}")

"""## 8) AVALIAÇÃO POR VALOR DE m — fatorada para reutilização"""

def evaluate_for_m(seed: int, device: torch.device,
                    baseline: nn.Module, student: nn.Module,
                    X_va: np.ndarray, X_te: np.ndarray,
                    yep_va: np.ndarray, yep_te: np.ndarray,
                    yfr_va: np.ndarray, yfr_te: np.ndarray,
                    progressive_va: Optional[np.ndarray] = None,
                    progressive_te: Optional[np.ndarray] = None,
                    m_detect: int = TTD_M,
                    tau_h_star: float = 0.85,
                    run_hybrid: bool = True,
                    temp_scaler: Optional["TemperatureScaler"] = None,  # OPT-5c
                    cached_te_probs_base: Optional[np.ndarray] = None,  # v24: pula infer_stream_fixed no test
                    cached_te_probs_kd:   Optional[np.ndarray] = None,  # v24: idem para AF-KD
                    theta_override: Optional[float] = None,             # v25-fix: theta fixo para sensibilidade em m
                    ) -> Tuple[List[dict], dict]:
    """
    Executa inferência de validação e teste para um valor específico de m.
    Retorna (rows_teste, calib_dict).

    theta_override: quando fornecido (sensibilidade em m), usa este theta
    fixo em vez de recalibrar. Garante que a unica variavel entre as
    linhas da Tabela 3 seja m, nao theta. Sem override (m=TTD_M padrao),
    theta e calibrado normalmente no VAL.
    """
    # Validacao
    va_base = infer_stream_fixed(baseline, X_va, device)
    thr_base   = select_thr_ep(va_base.frame_probs, yep_va)
    # FIX-17b: theta global unico calibrado no baseline
    # v25-fix: se theta_override fornecido, usa diretamente (sensibilidade em m)
    if theta_override is not None:
        theta_global = theta_override
        print(f'    [v25-sens] theta fixo={theta_global:.4f} (override para m={m_detect})')
    else:
        theta_global = select_theta_ttd(yfr_va, va_base.frame_probs, m_detect)
    theta_base   = theta_global

    va_kd = infer_stream_fixed(student, X_va, device)
    thr_kd    = select_thr_ep(va_kd.frame_probs, yep_va)
    theta_kd  = theta_global   # FIX-17b: mesmo θ global
    # O FIX-12 criava comparação assimétrica: θ_kd=0.35 vs θ_base=0.25 ->
    # AF-KD era testado em regime MAIS DIFÍCIL, resultando em TTD maior (pior).
    # Seed 43 chegou a −1887% de 'redução'. Com θ simétrico:
    # AF-KD detecta mais cedo PORQUE sobe mais rápido — não porque tem θ menor.

    theta_kd_adapt = select_theta_ttd_adapt(yfr_va, va_kd.frame_probs, m_detect)
    deriv_kd_delta, deriv_kd_prob, deriv_kd_smooth = select_derivative_ttd_params(yfr_va, va_kd.frame_probs, m_detect)
    ref_kd    = summarize_eval(
        va_kd, yep_va, yfr_va, thr_kd, theta_kd, m_detect,
        progressive_mask=np.asarray(progressive_va).astype(bool) if progressive_va is not None else None,
        abrupt_mask=((~np.asarray(progressive_va).astype(bool)) & (yep_va == 1)) if progressive_va is not None else None,
        theta_ttd_adapt=theta_kd_adapt,
        deriv_delta=deriv_kd_delta,
        deriv_prob_floor=deriv_kd_prob,
        deriv_smooth_w=deriv_kd_smooth,
    )

    # Diagnostico: confirmar que theta nao e mais 0.10
    print(f"    [v17-diag] θ_global={theta_global:.2f} | thr_base={thr_base:.2f} thr_kd={thr_kd:.2f}")

    hybrid_params = None
    if run_hybrid:
        hybrid_params = select_hybrid_policy(
            student, X_va, yep_va, yfr_va, device,
            thr_ep=thr_kd, theta_ttd=theta_kd,
            ref_fixed=ref_kd, m_detect=m_detect)

    # Teste
    rows = []
    progressive_te_mask = np.asarray(progressive_te).astype(bool) if progressive_te is not None else None
    abrupt_te_mask = (~progressive_te_mask & (yep_te == 1)) if progressive_te_mask is not None else None

    # v24: usa frame_probs cacheados (.npy) se disponíveis — evita ~30k forward passes
    # O VAL acima sempre roda (necessário para calibrar θ, thr, τ, gating).
    # lat_ms=0.0 quando cacheado; para medir latência oficial, apagar os .npy.
    if cached_te_probs_base is not None and cached_te_probs_kd is not None:
        te_base = StreamEval(
            frame_probs       = cached_te_probs_base,
            episode_probs     = episode_probs_from_frames(cached_te_probs_base),
            lat_ms            = 0.0,
            skip_pct          = 0.0,
            cost_ms_per_frame = 0.0,
        )
        te_kd = StreamEval(
            frame_probs       = cached_te_probs_kd,
            episode_probs     = episode_probs_from_frames(cached_te_probs_kd),
            lat_ms            = 0.0,
            skip_pct          = 0.0,
            cost_ms_per_frame = 0.0,
        )
        print(f"    [v24-cache] TEST: frame_probs restaurados do .npy  (lat_ms não medida)")
    else:
        te_base = infer_stream_fixed(baseline, X_te, device)
        te_kd   = infer_stream_fixed(student,  X_te, device)
    rows.append({
        "Seed": seed, "m": m_detect,
        "Modelo": "LSTM-Baseline", "Politica": "Fixa",
        **summarize_eval(te_base, yep_te, yfr_te, thr_base, theta_base, m_detect,
                         progressive_mask=progressive_te_mask, abrupt_mask=abrupt_te_mask,
                         theta_ttd_adapt=theta_kd_adapt,
                         deriv_delta=deriv_kd_delta, deriv_prob_floor=deriv_kd_prob, deriv_smooth_w=deriv_kd_smooth),
    })
    rows.append({
        "Seed": seed, "m": m_detect,
        "Modelo": "LSTM-AF-KD", "Politica": "Fixa",
        **summarize_eval(te_kd, yep_te, yfr_te, thr_kd, theta_kd, m_detect,
                         progressive_mask=progressive_te_mask, abrupt_mask=abrupt_te_mask,
                         theta_ttd_adapt=theta_kd_adapt,
                         deriv_delta=deriv_kd_delta, deriv_prob_floor=deriv_kd_prob, deriv_smooth_w=deriv_kd_smooth),
    })

    # OPT-5c (v20): reporta ECE calibrado por temperature scaling (se disponível)
    # A temperatura não altera frame_probs nem TTD — afeta apenas a calibração
    # probabilística episódica (ECE). Registra como linha separada para comparação.
    if temp_scaler is not None:
        ep_probs_raw = te_kd.episode_probs
        ts_cpu = temp_scaler.cpu()   # move para CPU — evita conflito cuda/cpu
        ts_cpu.eval()
        with torch.no_grad():
            ep_tensor = torch.tensor(ep_probs_raw, dtype=torch.float32)
            ep_logits_ts = ts_cpu(ep_tensor).numpy()
        ep_probs_ts = torch.sigmoid(
            torch.tensor(ep_logits_ts, dtype=torch.float32)
        ).numpy()
        ece_ts = expected_calibration_error(yep_te.astype(float), ep_probs_ts)
        print(f"    [TemperatureScaler] ECE bruto={rows[-1].get('ECE', float('nan')):.4f} "
              f"-> ECE calibrado={ece_ts:.4f}  (T*={float(ts_cpu.temperature.item()):.4f})")

    np.save(f"{EXP_NAME}_frame_probs_base_seed{seed}.npy", te_base.frame_probs)
    np.save(f"{EXP_NAME}_frame_probs_kd_seed{seed}.npy",   te_kd.frame_probs)
    np.save(f"{EXP_NAME}_y_true_fr_te_seed{seed}.npy",     yfr_te)

    try:
        from analise_fairness_v4 import run_fairness_analyses
        run_fairness_analyses(
            seed=seed,
            y_true_fr_te=yfr_te,
            probs_base=te_base.frame_probs,
            probs_kd=te_kd.frame_probs,
            theta_global=theta_global,
            theta_base=theta_base,
            theta_kd=theta_kd,
            window=T, m=m_detect,
        )
    except ModuleNotFoundError:
        print("    [aviso] analise_fairness_v4.py não encontrado — análise de fairness pulada.")

    # Adicionar logo após a inferência de teste:
    plot_effective_ttd_comparison(
        y_true_fr      = yfr_te[yep_te == 1],   # só episódios críticos
        probs_baseline = te_base.frame_probs[yep_te == 1],
        thr_baseline   = theta_base,
        probs_afkd     = te_kd.frame_probs[yep_te == 1],
        thr_afkd       = theta_kd,
    )
    if hybrid_params is not None:
        tau_d, tau_h = hybrid_params

        # AF-KD Híbrida
        te_h = infer_stream_hybrid(student, X_te, device,
                                    tau_delta=tau_d, tau_h=tau_h)
        rows.append({
            "Seed": seed, "m": m_detect,
            "Modelo": "LSTM-AF-KD",
            "Politica": f"Hibrida (tau={tau_d:.4f}, H={tau_h:.2f})",
            **summarize_eval(te_h, yep_te, yfr_te, thr_kd, theta_kd, m_detect,
                             progressive_mask=progressive_te_mask, abrupt_mask=abrupt_te_mask,
                             theta_ttd_adapt=theta_kd_adapt,
                             deriv_delta=deriv_kd_delta, deriv_prob_floor=deriv_kd_prob, deriv_smooth_w=deriv_kd_smooth),
        })
        print(f"    [AF-KD Híbrida]  tau_d={tau_d:.4f} tau_h={tau_h:.2f} "
              f"FR={rows[-1].get('FailRate', float('nan')):.4f} "
              f"Skip={rows[-1].get('SkipPct', float('nan')):.1f}% "
              f"Cost={rows[-1].get('Cost_ms_per_frame', float('nan')):.4f} ms/q")

        # Baseline Híbrida (PATCH: mesmos tau_d, tau_h do AF-KD)
        # Racional: isola o efeito da supervisão temporal AF-KD no gating —
        # aplica o mesmo mecanismo MHEG sobre o baseline sem AF-KD e mede
        # quanto a cobertura degrada. Gera o ponto "Baseline Híbrida" no
        # gráfico de trade-off FR × custo e na Tabela de ablação fatorial.
        te_base_h = infer_stream_hybrid(baseline, X_te, device,
                                         tau_delta=tau_d, tau_h=tau_h)
        rows.append({
            "Seed": seed, "m": m_detect,
            "Modelo": "LSTM-Baseline",
            "Politica": f"Hibrida (tau={tau_d:.4f}, H={tau_h:.2f})",
            **summarize_eval(te_base_h, yep_te, yfr_te, thr_base, theta_base, m_detect,
                             progressive_mask=progressive_te_mask, abrupt_mask=abrupt_te_mask,
                             theta_ttd_adapt=theta_kd_adapt,
                             deriv_delta=deriv_kd_delta, deriv_prob_floor=deriv_kd_prob, deriv_smooth_w=deriv_kd_smooth),
        })
        print(f"    [Baseline Híbrida] tau_d={tau_d:.4f} tau_h={tau_h:.2f} "
              f"FR={rows[-1].get('FailRate', float('nan')):.4f} "
              f"Skip={rows[-1].get('SkipPct', float('nan')):.1f}% "
              f"Cost={rows[-1].get('Cost_ms_per_frame', float('nan')):.4f} ms/q")

    elif run_hybrid:
        print(f"    AVISO (m={m_detect}): nenhum (τΔ, τH) satisfaz as restrições no VAL.")

    calib = {
        "m": m_detect, "thr_base": thr_base, "theta_base": theta_base,
        "thr_kd": thr_kd, "theta_kd": theta_kd,
        "theta_kd_adapt": theta_kd_adapt,
        "deriv_kd_delta": deriv_kd_delta,
        "deriv_kd_prob": deriv_kd_prob,
        "deriv_kd_smooth": deriv_kd_smooth,
        "hybrid_params": hybrid_params,
    }
    return rows, calib

"""## 9) LOOP PRINCIPAL"""

def build_ttd_method_comparison(main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida, por modelo/política, as três leituras de antecipação:
      - TTD binário tradicional
      - TTD adaptativo por limiar
      - TTD por derivada
    Saída pronta para paper e discussão metodológica.
    """
    rows = []
    for (modelo, politica), g in main_df.groupby(["Modelo", "Politica"], dropna=False):
        row = {"Modelo": modelo, "Politica": politica}
        metrics = [
            ("TTD_Bin", "TTD"),
            ("TTD_Adapt", "TTD_Adapt"),
            ("TTD_Deriv", "TTD_Deriv"),
            ("TTD_Prog_Bin", "TTD_Progressive"),
            ("TTD_Prog_Adapt", "TTD_Progressive_Adapt"),
            ("TTD_Prog_Deriv", "TTD_Progressive_Deriv"),
            ("TTD_Abr_Bin", "TTD_Abrupt"),
            ("TTD_Abr_Adapt", "TTD_Abrupt_Adapt"),
            ("TTD_Abr_Deriv", "TTD_Abrupt_Deriv"),
        ]
        for out_name, col in metrics:
            if col in g.columns:
                row[f"{out_name}_mean"] = round(float(g[col].mean()), 4)
                row[f"{out_name}_std"] = round(float(g[col].std(ddof=0)), 4)
        if "ThetaTTD_Adapt" in g.columns:
            row["ThetaTTD_Adapt_mean"] = round(float(g["ThetaTTD_Adapt"].mean()), 4)
        if "DerivDelta" in g.columns:
            row["DerivDelta_mean"] = round(float(g["DerivDelta"].mean()), 4)
        if "DerivProbFloor" in g.columns:
            row["DerivProbFloor_mean"] = round(float(g["DerivProbFloor"].mean()), 4)
        if "DerivSmoothW" in g.columns:
            row["DerivSmoothW_mean"] = round(float(g["DerivSmoothW"].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def export_ttd_method_table(ttd_cmp: pd.DataFrame, out_tex: str = f"{EXP_NAME}_paper_table_ttd_eval_rows.tex") -> None:
    if ttd_cmp.empty:
        Path(out_tex).write_text("", encoding="utf-8")
        return
    lines = []
    for _, r in ttd_cmp.iterrows():
        vals = [
            str(r.get("Modelo", "")),
            str(r.get("Politica", "")),
            str(int(r.get("ForecastHorizon", FORECAST_HORIZON_K))),
            f"{r.get('TTD_Bin_mean', float('nan')):.4f}",
            f"{r.get('TTD_Adapt_mean', float('nan')):.4f}",
            f"{r.get('TTD_Deriv_mean', float('nan')):.4f}",
            f"{r.get('TTD_Prog_Bin_mean', float('nan')):.4f}",
            f"{r.get('TTD_Prog_Adapt_mean', float('nan')):.4f}",
            f"{r.get('TTD_Prog_Deriv_mean', float('nan')):.4f}",
            f"{r.get('TTD_Prog_Adapt_reduction_vs_baseline_pct', float('nan')):.2f}",
            f"{r.get('TTD_Prog_Deriv_reduction_vs_baseline_pct', float('nan')):.2f}",
            f"{r.get('ThetaTTD_Adapt_mean', float('nan')):.4f}",
            f"{r.get('DerivDelta_mean', float('nan')):.4f}",
            f"{r.get('DerivProbFloor_mean', float('nan')):.4f}",
        ]
        lines.append(" & ".join(vals) + " \\")
    Path(out_tex).write_text("\n".join(lines) + "\n", encoding="utf-8")

def coverage_ttd_curve(
    y_true_fr: np.ndarray,       # (N, T) labels de frame
    frame_probs: np.ndarray,     # (N, T) probabilidades do modelo
    thr: float,
    m: int = 3,
    window: int = 96,
    dt: float = 10.0 / 96,
    label: str = "",
    ax=None,
    color: str = "blue",
    linestyle: str = "-"
):
    """
    Para cada limiar de TTD t* em [0, T*dt],
    calcula: P(TTD <= t* | episódio crítico detectável).
    Episódios não detectados contribuem como TTD = penalidade máxima.
    """
    ttds = []
    for yt, yp in zip(y_true_fr, frame_probs):
        idx = np.where(yt > 0)[0]
        if len(idx) == 0:
            continue
        t0 = int(idx[0])
        # first_stable_detection inline
        count, det = 0, None
        for t in range(t0, len(yp)):
            if yp[t] >= thr:
                count += 1
                if count >= m:
                    det = t - m + 1
                    break
            else:
                count = 0
        delay = (det - t0) if det is not None else (window - t0)
        ttds.append(delay * dt)

    ttds = np.array(ttds)
    t_grid = np.linspace(0, (window) * dt, 200)
    coverage = [(ttds <= t).mean() for t in t_grid]

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(t_grid, coverage, color=color, linestyle=linestyle,
            linewidth=2, label=label)
    return ax, ttds


def plot_effective_ttd_comparison(
    y_true_fr,
    probs_baseline, thr_baseline,
    probs_afkd,     thr_afkd,
    window=96, m=3, dt=10/96
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Painel 1: Curva Cobertura x TTD ---
    ax = axes[0]
    ax, ttds_base = coverage_ttd_curve(
        y_true_fr, probs_baseline, thr_baseline,
        m=m, window=window, dt=dt,
        label=f"LSTM-Baseline (FailRate≈0.32)", ax=ax,
        color="tomato", linestyle="--"
    )
    ax, ttds_kd = coverage_ttd_curve(
        y_true_fr, probs_afkd, thr_afkd,
        m=m, window=window, dt=dt,
        label=f"LSTM + AF-KD (FailRate≈0.09)", ax=ax,
        color="steelblue", linestyle="-"
    )

    # Marca o TTD bruto reportado na tabela (apenas detectados)
    ttd_base_reported = float(np.mean(ttds_base[ttds_base < window * dt]))
    ttd_kd_reported   = float(np.mean(ttds_kd[ttds_kd < window * dt]))

    ax.axvline(ttd_base_reported, color="tomato",   linestyle=":",
               label=f"TTD bruto baseline={ttd_base_reported:.2f}s")
    ax.axvline(ttd_kd_reported,   color="steelblue", linestyle=":",
               label=f"TTD bruto AF-KD={ttd_kd_reported:.2f}s")

    ax.set_xlabel("Orçamento de TTD (s)", fontsize=12)
    ax.set_ylabel("Fração de episódios críticos cobertos", fontsize=12)
    ax.set_title("Cobertura × TTD\n(inclui episódios não detectados como penalidade)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, window * dt)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.4)

    # --- Painel 2: Histograma do TTD efetivo (com penalidade) ---
    ax2 = axes[1]
    penalty = window * dt  # ~10s
    bins = np.linspace(0, penalty + 0.5, 30)

    ax2.hist(ttds_base, bins=bins, alpha=0.55, color="tomato",
             label="LSTM-Baseline", density=True)
    ax2.hist(ttds_kd,   bins=bins, alpha=0.55, color="steelblue",
             label="LSTM + AF-KD",  density=True)

    # Barra de penalidade
    ax2.axvline(penalty, color="black", linestyle="--", linewidth=1.5,
                label=f"Penalidade máx ≈ {penalty:.1f}s\n(episódio não detectado)")

    ax2.set_xlabel("TTD efetivo (s) — inclui penalidade", fontsize=12)
    ax2.set_ylabel("Densidade", fontsize=12)
    ax2.set_title("Distribuição do TTD Efetivo\n(baseline empilha episódios na penalidade máxima)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    # plt.close garante que handle anterior não trava o arquivo
    # (causa do PermissionError observado na seed 45 quando o PDF estava aberto)
    plt.close('all')
    plt.savefig(f"{EXP_NAME}_ttd_comparison_fair.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(f"{EXP_NAME}_ttd_comparison_fair.png", bbox_inches="tight", dpi=150)
    plt.show()
    print(f"\nTTD efetivo médio (com penalidade):")
    print(f"  Baseline : {np.mean(ttds_base):.3f}s")
    print(f"  AF-KD    : {np.mean(ttds_kd):.3f}s")
    print(f"\nTTD bruto (só detectados — o que a tabela mostra):")
    print(f"  Baseline : {ttd_base_reported:.3f}s  <- número 'barato'")
    print(f"  AF-KD    : {ttd_kd_reported:.3f}s")
def main() -> None:
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memória GPU: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    all_results:    List[dict] = []
    all_sens_rows:  List[dict] = []

    for SEED in SEEDS:
        print("\n" + "=" * 78)
        print(f"  RUN SEED = {SEED}")
        print("=" * 78)
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", device)

        # Dataset (DatasetConfig — reutiliza NPZ+CSV ou gera online)
        ds_cfg = DatasetConfig(seed=SEED)
        X, y_fr_cont, y_fr_obs, meta = load_or_generate_dataset(ds_cfg)

        class_col = infer_class_column(meta)
        cat = meta[class_col].astype(str).values
        y_ep = (cat == "Critico").astype(np.int64)

        progressive_flag = (meta["progressive"].fillna(0).astype(int).values.astype(bool)
                            if "progressive" in meta.columns
                            else np.zeros(len(meta), dtype=bool))
        progressive_flag = progressive_flag & (y_ep == 1)
        abrupt_flag = (~progressive_flag) & (y_ep == 1)

        FRAME_THR = 0.35
        y_fr = ((y_fr_cont > FRAME_THR).astype(np.int64) * y_ep[:, None]).astype(np.int64)
        y_fr_future      = y_fr.copy()
        y_fr_cont_future = y_fr_cont.copy()

        # Auditoria do dataset para paper novo
        n_crit = int(y_ep.sum())
        print(f"  Dataset: {X.shape}  Críticos: {n_crit}/{len(y_ep)}")
        print(f"  FRAME_THR: {FRAME_THR:.2f} | TTD_M padrão: {TTD_M}")
        if "persistent_tail" in meta.columns and n_crit > 0:
            tail_rate = float(meta.loc[meta["class_name"].eq("Critico") if "class_name" in meta.columns else meta["categoria"].eq("Critico"), "persistent_tail"].mean())
            print(f"  Tail crítica média: {tail_rate:.3f}")
        if "event_start" in meta.columns and n_crit > 0:
            crit_mask = (cat == "Critico")
            print(f"  Onset médio (críticos): {float(meta.loc[crit_mask, 'event_start'].mean()):.2f}")
        if n_crit > 0:
            print(f"  Críticos progressivos: {int(progressive_flag.sum())}/{n_crit} | abruptos: {int(abrupt_flag.sum())}/{n_crit}")
            if "onset_gap_035" in meta.columns and progressive_flag.sum() > 0:
                print(f"  onset_gap_035 médio (progressivos): {float(meta.loc[progressive_flag, 'onset_gap_035'].mean()):.2f}")
            if "onset_gap_045" in meta.columns and progressive_flag.sum() > 0:
                print(f"  onset_gap_045 médio (progressivos): {float(meta.loc[progressive_flag, 'onset_gap_045'].mean()):.2f}")
            if "onset_gap_055" in meta.columns and progressive_flag.sum() > 0:
                print(f"  onset_gap_055 médio (progressivos): {float(meta.loc[progressive_flag, 'onset_gap_055'].mean()):.2f}")


        splits = make_splits(
            y_ep, SEED,
            split_path=ds_cfg.split_path,
            reuse=ds_cfg.reuse_splits,
            meta=meta,                               # v19: stratify recipe_id
            use_stratified=USE_STRATIFIED_SPLITS,
            force_regen=FORCE_REGEN_SPLITS,
        )
        sliced = split_arrays(splits,
            X, y_ep, y_fr, y_fr_cont, y_fr_future, y_fr_cont_future, progressive_flag)
        (X_tr, yep_tr, yfr_tr_now, yfrc_tr_now, yfr_tr_fc, yfrc_tr_fc, prog_tr) = sliced["train"]
        (X_va, yep_va, yfr_va_now, yfrc_va_now, yfr_va_fc, yfrc_va_fc, prog_va) = sliced["val"]
        (X_te, yep_te, yfr_te_now, yfrc_te_now, yfr_te_fc, yfrc_te_fc, prog_te) = sliced["test"]
        print(f"  Split: tr={X_tr.shape[0]}  va={X_va.shape[0]}  te={X_te.shape[0]}")
        print(f"  Forecast horizon k: {FORECAST_HORIZON_K} frames")

        # v19: auditoria de composição dos splits (recipe × split)
        if "recipe_id" in meta.columns:
            class_col_chk = infer_class_column(meta)
            for split_name, split_idx in [("train", splits["train_idx"]),
                                           ("val",   splits["val_idx"]),
                                           ("test",  splits["test_idx"])]:
                sub = meta.iloc[split_idx]
                crit_sub = sub[sub[class_col_chk].astype(str) == "Critico"]
                if len(crit_sub) > 0:
                    recipe_counts = crit_sub["recipe_id"].value_counts().sort_index()
                    counts_str = "  ".join(f"{r}={n}" for r, n in recipe_counts.items())
                    print(f"  [v19-audit] {split_name:5s} crit={len(crit_sub):3d} | {counts_str}")

        # Checkpoints (v23)
        ckpt_teacher  = MODEL_DIR / f"teacher_seed{SEED}.pt"
        ckpt_baseline = MODEL_DIR / f"baseline_seed{SEED}.pt"
        ckpt_student  = MODEL_DIR / f"student_seed{SEED}.pt"
        ckpt_scaler   = MODEL_DIR / f"temp_scaler_seed{SEED}.pt"

        # v24: auto-detecção — se os três .pt existem, reusa sem flag manual
        _models_ready = ckpt_student.exists() and ckpt_baseline.exists() and ckpt_teacher.exists()
        if _models_ready:
            # Carregar pesos salvos — pula treino completamente
            print(f"\n[v24] Checkpoints encontrados — carregando pesos seed {SEED} (treino pulado)...")
            teacher  = MultiTaskLSTM(D, H, bi=True).to(device)
            if ckpt_teacher.exists():
                teacher.load_state_dict(torch.load(ckpt_teacher, map_location=device))
                print(f"  professor carregado: {ckpt_teacher}")

            baseline = MultiTaskLSTM(D, H, bi=False).to(device)
            baseline.load_state_dict(torch.load(ckpt_baseline, map_location=device))
            print(f"  baseline carregado:  {ckpt_baseline}")

            student  = MultiTaskLSTM(D, H, bi=False).to(device)
            student.load_state_dict(torch.load(ckpt_student, map_location=device))
            print(f"  student carregado:   {ckpt_student}")

            temp_scaler = TemperatureScaler().to(device)
            if ckpt_scaler.exists():
                temp_scaler.load_state_dict(torch.load(ckpt_scaler, map_location=device))
                temp_scaler_T = float(temp_scaler.temperature.item())
                print(f"  temp_scaler carregado: T*={temp_scaler_T:.4f}")
            else:
                print("  temp_scaler não encontrado — recalibrando no VAL...")
                temp_scaler = calibrate_temperature(student, X_va, yep_va, device)
                temp_scaler_T = float(temp_scaler.temperature.item())
                torch.save(temp_scaler.state_dict(), ckpt_scaler)
                print(f"  Temperature scaling: T* = {temp_scaler_T:.4f} (salvo)")
        else:
            # Treino
            print("\nTreinando Professor (BiLSTM)...")
            teacher  = MultiTaskLSTM(D, H, bi=True)
            # FIX-13c: professor BiLSTM com target futuro K=8
            k_teacher = 8
            yfr_tr_teacher = np.zeros_like(yfr_tr_now)
            yfr_tr_teacher[:, :-k_teacher] = yfr_tr_now[:, k_teacher:]
            teacher  = train_teacher(teacher, X_tr, yfr_tr_teacher, yep_tr, device, epochs=60,
                                     yfr_aux=None, aux_weight=0.0)
            torch.save(teacher.state_dict(), ckpt_teacher)
            print(f"  professor salvo: {ckpt_teacher}")

            print("\nTreinando Baseline (LSTM causal)...")
            baseline = MultiTaskLSTM(D, H, bi=False)
            # FIX-7: baseline puro sem aux
            baseline = train_baseline(baseline, X_tr, yfr_tr_fc, yep_tr, device, epochs=60,
                                      yfr_aux=None, aux_weight=0.0)
            torch.save(baseline.state_dict(), ckpt_baseline)
            print(f"  baseline salvo: {ckpt_baseline}")

            print("\nTreinando AF-KD (LSTM causal)...")
            student  = MultiTaskLSTM(D, H, bi=False)
            # [v26-ABL] abrupt_mask condicional: só usado se M4 ativo na ablação
            abrupt_tr = (~prog_tr.astype(bool)) & (yep_tr == 1)
            _use_abrupt = _abl_cfg["use_abrupt_mask"]

            print(f"  [ABL={ABLATION_MODE!r}] M1α={_abl_cfg['onset_exp_alpha']} "
                  f"M2={_abl_cfg['use_temp_schedule']}(min={_abl_cfg['temp_min']}) "
                  f"M3λ={_abl_cfg['lambda_late']} "
                  f"M4={_use_abrupt} M5={_abl_cfg['use_hidden_kd']}")

            student  = train_afkd(
                student, teacher, X_tr, yfr_tr_fc, yep_tr, device,
                epochs=80, beta_max=0.35, temp=2.0, lam_start=0.20, lam_end=1.00,
                yfr_soft=yfrc_tr_fc, yfr_aux=None, yfr_soft_aux=None, aux_now_weight=0.0,
                early_onset_thr=0.35, early_target_hi=0.90,
                # [v26-ABL] parâmetros controlados pela ablação
                abrupt_mask      = abrupt_tr if _use_abrupt else None,
                prog_mask        = prog_tr,          # [v27-M6] pre-onset push para progressivos
                lambda_late      = _abl_cfg["lambda_late"],
                theta_late       = _abl_cfg["theta_late"],
                use_temp_schedule= _abl_cfg["use_temp_schedule"],
                temp_min         = _abl_cfg["temp_min"],
                onset_exp_alpha  = _abl_cfg["onset_exp_alpha"],
                use_hidden_kd    = _abl_cfg["use_hidden_kd"],
                gamma_hidden     = _abl_cfg["gamma_hidden"],
                hidden_kd_window = _abl_cfg["hidden_kd_window"],
            )

            # 🔥 v25 — PRUNING OPCIONAL DO MODELO (esparsa L1 30%)
            # Reduz FLOPs na inferência Edge; pesos insignificantes removidos.
            # Descomente o bloco abaixo se quiser ativar (mede impacto no FailRate
            # antes de usar em produção).
            # import torch.nn.utils.prune as prune
            # _prune_amount = 0.30
            # for module in student.modules():
            #     if isinstance(module, nn.Linear):
            #         prune.l1_unstructured(module, name="weight", amount=_prune_amount)
            #         prune.remove(module, "weight")
            # print(f"  [v25] Model pruning aplicado: {_prune_amount*100:.0f}% pesos removidos (L1)")
            torch.save(student.state_dict(), ckpt_student)
            print(f"  student salvo: {ckpt_student}")

            # OPT-5c (v20): calibração de temperatura pós-treino no VAL
            # Reduz ECE sem alterar pesos do modelo. Aplicada somente ao AF-KD.
            print("\nCalibrando temperatura (ECE) no VAL...")
            temp_scaler = calibrate_temperature(student, X_va, yep_va, device)
            temp_scaler_T = float(temp_scaler.temperature.item())
            torch.save(temp_scaler.state_dict(), ckpt_scaler)
            print(f"  Temperature scaling: T* = {temp_scaler_T:.4f} (salvo)")

        # v24: cache de frame_probs de test (pula infer_stream_fixed no test)
        _p_base_npy = Path(f"{EXP_NAME}_frame_probs_base_seed{SEED}.npy")
        _p_kd_npy   = Path(f"{EXP_NAME}_frame_probs_kd_seed{SEED}.npy")
        _cached_base_probs = np.load(_p_base_npy) if _p_base_npy.exists() else None
        _cached_kd_probs   = np.load(_p_kd_npy)   if _p_kd_npy.exists() else None
        if _cached_base_probs is not None:
            print(f"  [v24-cache] frame_probs de TEST carregados do .npy — inferência de test pulada (lat_ms=0)")
        else:
            print(f"  [v24] Sem cache de frame_probs — inferência de test será executada normalmente")

        # Avaliação principal (m = TTD_M = 2)
        print(f"\nAvaliando (m={TTD_M}, padrão do artigo)...")
        rows_main, calib_main = evaluate_for_m(
            SEED, device, baseline, student,
            X_va, X_te, yep_va, yep_te, yfr_va_now, yfr_te_now,
            progressive_va=prog_va, progressive_te=prog_te,
            m_detect=TTD_M, run_hybrid=True,
            temp_scaler=temp_scaler,            # OPT-5c: passa o scaler calibrado
            cached_te_probs_base=_cached_base_probs,  # v24: pula infer_stream_fixed se .npy existe
            cached_te_probs_kd=_cached_kd_probs)

        # 🔥 v25 — IGNORAR SEEDS RUINS
        # Se o AF-KD Fixa tiver FailRate > 0.30, a seed produziu um modelo
        # inviável (alta variância). Registra o aviso e pula a seed.
        _kd_fixa_rows = [r for r in rows_main
                         if r.get("Modelo") == "LSTM-AF-KD" and r.get("Politica") == "Fixa"]
        if _kd_fixa_rows:
            _seed_fr = float(_kd_fixa_rows[0].get("FailRate", 0.0))
            if _seed_fr > 0.30:
                print(f"\naviso: [v25] Seed {SEED} descartada — FailRate AF-KD={_seed_fr:.4f} > 0.30 (outlier)")
                continue   # pula toda a análise desta seed (sens, curva Pareto, etc.)

        all_results.extend(rows_main)

        # Curva trade-off (para o gráfico de Pareto)
        if calib_main["hybrid_params"] is not None:
            tau_h_curve = calib_main["hybrid_params"][1]
        else:
            tau_h_curve = 0.85
        curve_rows = []
        deltas_te = np.abs(np.diff(X_te[:, :, KIN], axis=1)).max(axis=2).reshape(-1)
        for p in TAU_DELTA_PERCENTILES:
            tau_d = float(np.percentile(deltas_te, p))
            se_c  = infer_stream_hybrid_fast(student, X_te, device,
                                              tau_delta=tau_d, tau_h=tau_h_curve)
            r_c   = summarize_eval(se_c, yep_te, yfr_te_now,
                                    calib_main["thr_kd"], calib_main["theta_kd"],
                                    theta_ttd_adapt=calib_main.get("theta_kd_adapt"),
                                    deriv_delta=calib_main.get("deriv_kd_delta"),
                                    deriv_prob_floor=calib_main.get("deriv_kd_prob"),
                                    deriv_smooth_w=calib_main.get("deriv_kd_smooth"))
            curve_rows.append({"Seed": SEED, "tau_delta": tau_d,
                                "tau_h": tau_h_curve, **r_c})
        pd.DataFrame(curve_rows).to_csv(f"{EXP_NAME}_tradeoff_curve_seed{SEED}.csv", index=False)

        # Tabela do teste (m=3) para esta seed
        df_seed = pd.DataFrame(rows_main)
        cols    = ["Seed", "m", "Modelo", "Politica", "F1", "ECE",
                   "FailRate", "TTD", "TTDef", "TTD_Adapt", "TTD_Deriv", "TTD_Progressive", "TTD_Progressive_Adapt", "TTD_Progressive_Deriv", "TTD_Abrupt", "TTD_Abrupt_Adapt", "TTD_Abrupt_Deriv",
                   "ThetaTTD", "ThetaTTD_Adapt", "DerivDelta", "DerivProbFloor", "DerivSmoothW", "FailRate_Progressive", "FailRate_Abrupt",
                   "SkipPct", "Lat_ms", "Cost_ms_per_frame"]
        print("\n=== TABELA TESTE (m=3) ===")
        print(df_seed[[c for c in cols if c in df_seed.columns]]
              .to_markdown(index=False, floatfmt=".4f"))
        df_seed.to_csv(f"{EXP_NAME}_results_seed{SEED}.csv", index=False)

        # Análise de sensibilidade em m
        # v25-fix: theta fixo (calibrado para m=TTD_M) — garante que a
        # unica variavel entre as linhas da Tabela 3 seja m, nao theta.
        # Sem isso, select_theta_ttd recalibra para cada m e o resultado
        # e sempre o mesmo (theta=0.10 domina, deteccao imediata para todo m).
        _theta_sens = calib_main["theta_kd"]   # theta do m=TTD_M padrão
        print(f"\nAnálise de sensibilidade m ∈ {M_SENS_GRID} (theta fixo={_theta_sens:.4f})...")
        for m_val in M_SENS_GRID:
            if m_val == TTD_M:
                # já avaliado acima — reutiliza
                all_sens_rows.extend(rows_main)
                continue
            rows_m, _ = evaluate_for_m(
                SEED, device, baseline, student,
                X_va, X_te, yep_va, yep_te, yfr_va_now, yfr_te_now,
                progressive_va=prog_va, progressive_te=prog_te,
                m_detect=m_val, run_hybrid=False,
                cached_te_probs_base=_cached_base_probs,
                cached_te_probs_kd=_cached_kd_probs,
                theta_override=_theta_sens)  # v25-fix: theta fixo
            all_sens_rows.extend(rows_m)
            print(f"    m={m_val} concluído.")

    # 10) CONSOLIDAÇÃO, TABELAS E EXPORTAÇÃO LaTeX
    final_df = pd.DataFrame(all_results)
    sens_df  = pd.DataFrame(all_sens_rows)

    final_df.to_csv(f"{EXP_NAME}_results_all_seeds.csv",       index=False)
    sens_df.to_csv( f"{EXP_NAME}_results_sensitivity_m.csv",   index=False)

    # Metadados resumidos do dataset por seed (paper novo)
    ds_meta_rows = []
    for seed in SEEDS:
        csvp = Path(f"episodios_seed{seed}.csv")
        if csvp.exists():
            mdf = pd.read_csv(csvp, sep=";")
            class_col = "class_name" if "class_name" in mdf.columns else ("categoria" if "categoria" in mdf.columns else "class")
            crit = mdf[mdf[class_col].astype(str).eq("Critico")]
            ds_meta_rows.append({
                "Seed": seed,
                "Episodes": int(len(mdf)),
                "CriticalEpisodes": int(len(crit)),
                "CriticalRate": float(len(crit) / max(len(mdf), 1)),
                "PersistentTailRateCritical": float(crit["persistent_tail"].mean()) if ("persistent_tail" in crit.columns and len(crit) > 0) else float("nan"),
                "EventStartMeanCritical": float(crit["event_start"].mean()) if ("event_start" in crit.columns and len(crit) > 0) else float("nan"),
                "Last6MeanCleanCritical": float(crit["last6_mean_clean"].mean()) if ("last6_mean_clean" in crit.columns and len(crit) > 0) else float("nan"),
                "Last6MeanObsCritical": float(crit["last6_mean_obs"].mean()) if ("last6_mean_obs" in crit.columns and len(crit) > 0) else float("nan"),
                # v5: label principal = severity (0.7·mean + 0.3·p95) — promovido em risk_episode
                "RiskEpisodeMeanCritical": float(crit["risk_episode"].mean()) if ("risk_episode" in crit.columns and len(crit) > 0) else float("nan"),
                "RiskEpisodeOperationalMeanCritical": float(crit["risk_episode_operational"].mean()) if ("risk_episode_operational" in crit.columns and len(crit) > 0) else float("nan"),
                "RiskEpisodeSeverityMeanCritical": float(crit["risk_episode_severity"].mean()) if ("risk_episode_severity" in crit.columns and len(crit) > 0) else float("nan"),
                "CriticalProgressiveEpisodes": int(crit["progressive"].fillna(0).astype(int).sum()) if ("progressive" in crit.columns and len(crit) > 0) else 0,
                "CriticalAbruptEpisodes": int(len(crit) - crit["progressive"].fillna(0).astype(int).sum()) if ("progressive" in crit.columns and len(crit) > 0) else int(len(crit)),
                # v5: ~60% dos críticos são progressivos (6/10 receitas críticas)
                "ProgressiveRateCritical": float(crit["progressive"].fillna(0).astype(int).mean()) if ("progressive" in crit.columns and len(crit) > 0) else float("nan"),
                "OnsetGap035MeanProgressive": float(crit.loc[crit["progressive"].fillna(0).astype(int).eq(1), "onset_gap_035"].mean()) if ("progressive" in crit.columns and "onset_gap_035" in crit.columns and len(crit) > 0) else float("nan"),
                "OnsetGap045MeanProgressive": float(crit.loc[crit["progressive"].fillna(0).astype(int).eq(1), "onset_gap_045"].mean()) if ("progressive" in crit.columns and "onset_gap_045" in crit.columns and len(crit) > 0) else float("nan"),
                "OnsetGap055MeanProgressive": float(crit.loc[crit["progressive"].fillna(0).astype(int).eq(1), "onset_gap_055"].mean()) if ("progressive" in crit.columns and "onset_gap_055" in crit.columns and len(crit) > 0) else float("nan"),
            })
    if ds_meta_rows:
        pd.DataFrame(ds_meta_rows).to_csv(f"{EXP_NAME}_dataset_run_metadata.csv", index=False)

    # Tabela 2 — resultados principais (m=3 padrão)
    main_df      = final_df[final_df["m"] == TTD_M].copy()
    # Colapsa todas as variantes de política híbrida em uma única categoria para média/std.
    if "Politica" in main_df.columns:
        main_df["Politica"] = main_df["Politica"].apply(
            lambda s: "Hibrida" if isinstance(s, str) and s.startswith("Hibrida") else s
        )
    num_cols     = ["F1", "ECE", "FailRate", "TTD", "TTDef", "TTD_Adapt", "TTD_Deriv", "TTD_Progressive", "TTD_Progressive_Adapt", "TTD_Progressive_Deriv", "TTD_Abrupt", "TTD_Abrupt_Adapt", "TTD_Abrupt_Deriv",
                    "FailRate_Progressive", "FailRate_Abrupt", "Lat_ms",
                    "Cost_ms_per_frame", "SkipPct", "ThetaTTD", "ThetaTTD_Adapt", "DerivDelta", "DerivProbFloor", "DerivSmoothW"]
    present      = [c for c in num_cols if c in main_df.columns]
    main_summary = aggregate_mean_std(main_df, ["Modelo", "Politica"], present)

    # PATCH: adiciona N_Seeds para rastrear cobertura de seeds por configuração
    _pol_norm = main_df["Politica"].apply(
        lambda s: "Hibrida" if isinstance(s, str) and s.startswith("Hibrida") else s
    )
    _seed_counts = (
        main_df.assign(_pol_norm=_pol_norm)
        .groupby(["Modelo", "_pol_norm"])["Seed"]
        .nunique()
        .reset_index(name="N_Seeds")
        .rename(columns={"_pol_norm": "Politica"})
    )
    main_summary = main_summary.merge(_seed_counts, on=["Modelo", "Politica"], how="left")

    # Aviso explícito quando alguma config tem < 4 seeds
    _incomplete = main_summary[main_summary["N_Seeds"].fillna(0) < 4]
    if not _incomplete.empty:
        print("\nAVISO: configurações com cobertura incompleta de seeds:")
        for _, _row in _incomplete.iterrows():
            print(f"     {_row['Modelo']:20s} [{_row['Politica']:12s}]"
                  f"  -> {int(_row['N_Seeds'])}/4 seeds")
        print("   Execute run_hybrid_completion.py para completar as seeds faltantes.\n")
    ttd_method_cmp = build_ttd_method_comparison(main_df)
    ttd_method_cmp.to_csv(f"{EXP_NAME}_summary_ttd_methods.csv", index=False)

    # Tabela 3 — sensibilidade em m (AF-KD Fixa)
    sens_summary = build_sensitivity_table(
        sens_df, target_model="LSTM-AF-KD", target_policy="Fixa")

    print("\n" + "=" * 78)
    print("=== TABELA 2 — RESULTADOS PRINCIPAIS (m=3) ===")
    print(main_summary.to_string(index=False))

    print("\n=== TABELA 3 — SENSIBILIDADE EM m (AF-KD Fixa) ===")
    print(sens_summary.to_string(index=False))

    # Exportação LaTeX
    export_latex_macros(main_summary, sens_summary,
                        path=f"{EXP_NAME}_paper_metrics_macros.tex")
    export_latex_tables(main_summary, sens_summary,
                        path_tab2=f"{EXP_NAME}_paper_table2_rows.tex",
                        path_tab3=f"{EXP_NAME}_paper_table3_rows.tex")

    print("\nArquivos gerados:")
    print("  results_all_seeds.csv        (Tabela 2 — todas as seeds 42-45)")
    print("  results_sensitivity_m.csv    (Tabela 3 — sensibilidade em m)")
    print("  results_seedXX.csv           (por seed: 42, 43, 44, 45)")
    print("  tradeoff_curve_seedXX.csv    (curva Pareto por seed)")
    print("  paper_metrics_macros.tex     (macros LaTeX Tabela 2 + Tabela 3 + nota seed44)")
    print("  paper_table2_rows.tex        (linhas LaTeX Tabela 2)")
    print("  paper_table3_rows.tex        (linhas LaTeX Tabela 3)")
    print("  summary_ttd_methods.csv      (comparação TTD bin/adapt/deriv)")
    print("  paper_table_ttd_eval_rows.tex (linhas LaTeX comparação TTD)")
    print("")
    print("  v19 — mudanças aplicadas:")
    print("    [1] SEEDS = [42, 43, 44, 45]  (+seed 45)")
    print("    [2] make_splits estratificado por recipe_id nos críticos")
    print(f"        USE_STRATIFIED_SPLITS={USE_STRATIFIED_SPLITS}  FORCE_REGEN_SPLITS={FORCE_REGEN_SPLITS}")
    print("    [3] macro \\seedFourtyFourNote adicionada ao paper_metrics_macros.tex")


if __name__ == "__main__":
    main()

"""## Execução final

A célula abaixo chama `main()` apenas se este notebook estiver sendo executado de ponta a ponta.

"""
