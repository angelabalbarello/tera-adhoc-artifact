# -*- coding: utf-8 -*-
"""
tera_pipeline/utils/tera_utils.py
Utilities do TERA Pipeline: determinismo, logging científico, I/O.
"""

import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class AfTkdConfig:
    onset_exp_alpha: float
    temp_fixed: float
    lambda_late: float
    theta_late: float
    pre_onset_push_weight: float
    hidden_kd_window: int
    warmup_epochs: int
    rampup_epochs: int

@dataclass
class PipelineConfig:
    # Esta classe espelha a estrutura do YAML de forma tipada
    dataset: dict
    model: dict
    training_meta: dict
    af_tkd: AfTkdConfig
    calibration: dict

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Faz o parsing seguro do bloco af_tkd
        af_tkd_raw = raw["training"]["af_tkd"]
        af_tkd_obj = AfTkdConfig(**af_tkd_raw)

        return cls(
            dataset=raw["dataset"],
            model=raw["model"],
            training_meta=raw["training"],
            af_tkd=af_tkd_obj,
            calibration=raw["calibration"]
        )

# Determinismo

def set_global_seed(seed: int) -> None:
    """
    Configura todas as fontes de aleatoriedade para reprodutibilidade total.
    Deve ser chamado ANTES de qualquer operação probabilística.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)


def get_device() -> torch.device:
    """Retorna dispositivo de computação disponível."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Configuração

def load_config(path: str) -> dict:
    """Carrega configuração YAML. Falha com mensagem clara se não encontrado."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {cfg_path}\n"
            f"Crie o arquivo em {cfg_path} antes de executar o pipeline."
        )
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_config_snapshot(cfg: dict, out_path: Path) -> None:
    """Salva snapshot da configuração usada no experimento."""
    out_path.write_text(
        yaml.dump(cfg, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )


# Logging científico

def setup_logging(
    name: str = "tera_pipeline",
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configura logging para console + arquivo.
    Formato inclui timestamp e nível para auditabilidade.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# Rastreabilidade

def get_git_commit() -> str:
    """Retorna hash do commit git atual, ou 'not_in_git_repo'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "not_in_git_repo"
    except Exception:
        return "not_in_git_repo"


def get_package_versions() -> Dict[str, str]:
    """Retorna versões dos pacotes críticos para reprodutibilidade."""
    packages = ["torch", "numpy", "pandas", "sklearn", "scipy"]
    versions = {}
    for pkg in packages:
        try:
            import importlib
            mod = importlib.import_module(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


def hash_file(path: Path) -> str:
    """Hash SHA-256 de um arquivo para rastreabilidade do gerador."""
    if not path.exists():
        return "not_found"
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()[:16]


# I/O

def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Salva dicionário como JSON formatado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False,
                   default=_json_default) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Tipo não serializável: {type(obj)!r}")


# Validação de integridade do pipeline

def assert_offline_only(*component_names: str) -> None:
    """
    Decorator/asserção conceitual: documenta que estes componentes
    NÃO devem ser instanciados no runtime embarcado.
    """
    for name in component_names:
        assert name in {"teacher", "af_tkd_protocol", "tera_gen", "calibration"}, \
            f"Componente '{name}' não é um componente offline reconhecido."


def assert_online_only(*component_names: str) -> None:
    """
    Asserção: documenta que estes componentes compõem o runtime embarcado.
    """
    for name in component_names:
        assert name in {"student_causal", "mheg", "hidden_state", "thresholds"}, \
            f"Componente '{name}' não é um componente online reconhecido."


def verify_seed_coverage(
    results_csv: Path,
    required_seeds: list = None,
    required_configs: list = None,
) -> bool:
    """
    Verifica se o CSV de resultados tem cobertura completa de seeds × configs.
    Retorna True se completo, imprime aviso e retorna False se incompleto.
    """
    import pandas as pd

    if required_seeds is None:
        required_seeds = [42, 43, 44, 45]
    if required_configs is None:
        required_configs = ["baseline_fixed", "baseline_hybrid",
                            "aftkd_fixed", "aftkd_hybrid"]

    if not results_csv.exists():
        print(f"aviso: CSV não encontrado: {results_csv}")
        return False

    df = pd.read_csv(results_csv)
    complete = True
    for config_id in required_configs:
        seeds_found = sorted(
            df[df.get("config_id", pd.Series()) == config_id]["Seed"]
            .unique().tolist()
        )
        missing = [s for s in required_seeds if s not in seeds_found]
        if missing:
            print(f"aviso: {config_id}: seeds faltantes {missing}")
            complete = False
        else:
            print(f"  {config_id}: seeds {seeds_found}")
    return complete

def aftkd_loss(student_fr, student_ep, teacher_fr, y_episode, onset_frames, config, lam_kd, device):
    """
    Função de perda AF-TKD adaptada para o conceito de Framework Modular.
    Implementa Pre-onset Push, Late Penalty e Hidden Memory Alignment de forma parametrizada.
    """
    alpha      = config.af_tkd.onset_exp_alpha
    temp       = config.af_tkd.temp_fixed
    pre_push   = config.af_tkd.pre_onset_push_weight
    lam_late   = config.af_tkd.lambda_late
    theta_late = config.af_tkd.theta_late

    batch_size, seq_len, _ = student_fr.shape

    # Probs convertidas para cálculo soft (KL Divergence)
    p_s = F.log_softmax(student_fr / temp, dim=-1)
    p_t = F.softmax(teacher_fr / temp, dim=-1)
    kl_loss = F.kl_div(p_s, p_t, reduction='none').sum(dim=-1) * (temp ** 2)

    # Criar tensor de pesos temporais dinâmicos por frame
    t_weights = torch.ones((batch_size, seq_len), device=device)

    for i in range(batch_size):
        t0 = int(onset_frames[i].item())
        if y_episode[i] > 0.5 and t0 > 0:  # Apenas episódios críticos com onset válido
            # Janela Pós-Onset: Decaimento exponencial clássico
            for t in range(t0, seq_len):
                t_weights[i, t] = math.exp(-alpha * (t - t0))

                # [MECANISMO LATE PENALTY]
                prob_estudante = torch.sigmoid(student_fr[i, t]).mean()
                if prob_estudante < theta_late:
                    t_weights[i, t] *= lam_late

            # [MECANISMO PRE-ONSET PUSH]
            win_start = max(0, t0 - config.af_tkd.hidden_kd_window)
            for t in range(win_start, t0):
                t_weights[i, t] = pre_push

    # Aplicação dos pesos construídos pelo framework sobre a perda Soft
    weighted_kl = (kl_loss * t_weights).mean()

    # Perda de classificação base (Hard labels)
    loss_hard = nn.BCEWithLogitsLoss()(student_ep, y_episode)

    # Combinação final orientada pelo agendador de fases
    total_loss = (1.0 - lam_kd) * loss_hard + lam_kd * weighted_kl
    return total_loss