# -*- coding: utf-8 -*-
"""
tera_pipeline/training/tera_train.py
Gerenciamento de treinamento do TERA Pipeline.

Modelos treinados:
  · BiLSTM teacher  — acesso bidirecional; NÃO reside no dispositivo
  · LSTM-Baseline   — aluno causal sem destilação
  · LSTM-AF-TKD     — aluno causal com protocolo AF-TKD (3 fases)

Protocolo AF-TKD (ablação A: M1 + M3):
  · Temperatura fixa T=2.0
  · Ponderação exponencial w(t) = exp(−α·t), α=0.15
  · Fase 1 warmup  (10 épocas): λ_KD = 0
  · Fase 2 rampup  (17 épocas): λ_KD ↑ linearmente
  · Fase 3 steady  (53 épocas): λ_KD fixo

SAÍDAS em exp_dir/models/:
  teacher_seed{N}.pt
  baseline_seed{N}.pt
  student_seed{N}.pt

Compatibilidade retroativa com modelos_salvos/abl_A/:
  Se os checkpoints já existirem em modelos_salvos/abl_A/ e --reuse-models
  for passado, o treinamento é pulado e os modelos são copiados para exp_dir.
"""

import math
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# Modelo canônico

class MultiTaskLSTM(nn.Module):
    """
    LSTM causal unidirecional — modelo de inferência embarcada.

    Runtime online:
      · ~58K parâmetros (H=64, D=31, 2 camadas, fc_fr + fc_ep)
      · ~228 KB FP32, ~57 KB INT8
      · Estado oculto: 512 bytes (buffer fixo)
      · Custo: O(4(DH+H²)) por passo

    Nota: artigos anteriores reportavam "~25K" e "~98 KB", valores
    correspondentes a uma arquitetura de 1 camada sem fc_ep — incorretos
    para este modelo de 2 camadas. Os valores acima são os corretos.

    O professor BiLSTM usa esta mesma classe com bi=True
    mas NÃO reside no dispositivo embarcado.

    Compatibilidade com run_v29:
      forward() retorna (fr, ep) onde fr.shape=(B,T,1), ep.shape=(B,1),
      idêntico ao MultiTaskLSTM do run_v29 (linha 420).
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.1,
                 bi: bool = False):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.bidirectional = bi
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bi,
        )
        factor = 2 if bi else 1
        # fc_fr: head frame-a-frame (usada na inferência embarcada)
        # fc_ep: head episódica com mean-pooling (usada na perda de treinamento)
        # Nomes compatíveis com checkpoints run_v29.
        self.fc_fr = nn.Linear(hidden_dim * factor, 1)
        self.fc_ep = nn.Linear(hidden_dim * factor, 1)

    def forward(
        self, x: torch.Tensor, h: Optional[Tuple] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq, input_dim)

        Retorna: (fr, ep)
          fr:  (batch, seq, 1) — logits frame-a-frame, NÃO squeezado
          ep:  (batch, 1)      — logit episódico via mean-pooling

        Indexação na inferência (idêntico ao run_v29):
          prob_t = sigmoid(fr[0, -1, 0])

        Se return_hidden=True, retorna (fr, ep, out_seq) para
        destilação de estado oculto (M5, opcional).
        """
        out, hc = self.lstm(x, h)
        fr = self.fc_fr(out)            # (batch, seq, 1)
        ep = self.fc_ep(out.mean(1))    # (batch, 1)   — mean-pooling temporal
        if return_hidden:
            return fr, ep, out
        return fr, ep


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Losses

# Losses Parametrizadas (Framework Modular)

def aftkd_loss(
    student_fr:   torch.Tensor,   # (B, T, 1) — saída de fc_fr do student
    student_ep:   torch.Tensor,   # (B, 1)    — saída de fc_ep do student
    teacher_fr:   torch.Tensor,   # (B, T, 1) — saída de fc_fr do professor
    y_episode:    torch.Tensor,   # (B,) ou (B, 1)
    onset_frames: torch.Tensor,   # (B,)
    config,                       # <--- Injeção do objeto de configuração do Framework
    lam_kd:       float = 0.5,
    window:       int   = 96,
) -> torch.Tensor:
    """
    Perda AF-TKD Avançada e Parametrizada integrada ao TERA Framework.

    Implementa de forma estritamente causal:
      1. Decaimento Exponencial Pós-Onset (assimétrico).
      2. [Ajuste 1] Pre-onset Push: Força o aprendizado de precursores antes do t0.
      3. [Ajuste 1] Late Penalty: Pune atrasos de subida de probabilidade do aluno.
    """
    # Squeeze para compatibilidade interna com as dimensões de lote
    student_logits = student_fr.squeeze(-1)   # (B, T)
    teacher_logits = teacher_fr.squeeze(-1)   # (B, T)
    ep_logit       = student_ep.squeeze(-1)   # (B,)

    batch = student_logits.shape[0]
    T     = student_logits.shape[1]

    # Extração de Hiperparâmetros do Framework (config.yaml)
    alpha       = config.af_tkd.onset_exp_alpha
    temp        = config.af_tkd.temp_fixed
    pre_push_w  = config.af_tkd.pre_onset_push_weight
    lam_late    = config.af_tkd.lambda_late
    theta_late  = config.af_tkd.theta_late
    kd_window   = config.af_tkd.hidden_kd_window


    # L_hard: BCE no head episódico dedicado (fc_ep)
    y_ep_float = y_episode.float().squeeze(-1) if y_episode.ndim > 1 else y_episode.float()
    l_hard = F.binary_cross_entropy_with_logits(ep_logit, y_ep_float)

    # L_soft: KL divergência com máscara temporal avançada
    p_teacher = torch.sigmoid(teacher_logits / temp)
    p_student = torch.sigmoid(student_logits / temp)
    kl = F.binary_cross_entropy(p_student, p_teacher, reduction="none")  # (B, T)

    # Criar matriz de pesos temporais dinâmicos por frame (B, T)
    weights = torch.ones_like(kl)

    crit_mask = (y_ep_float == 1)
    if crit_mask.sum() > 0:
        for i in range(batch):
            if crit_mask[i]:
                t0 = int(onset_frames[i].item())
                if t0 > 0:
                    # Inicializa vetor de pesos para o episódio corrente
                    w_ep = torch.ones(T, device=student_logits.device)

                    # 1. Janela Pós-Onset: Aplica o decaimento exponencial assimétrico
                    for t in range(t0, T):
                        w_ep[t] = torch.exp(torch.tensor(-alpha * (t - t0), device=student_logits.device))

                        # [MECANISMO LATE PENALTY]
                        # Se o aluno estiver reativo/atrasado (abaixo de theta_late), aplica punição severa
                        if torch.sigmoid(student_logits[i, t]) < theta_late:
                            w_ep[t] *= lam_late

                    # 2. [MECANISMO PRE-ONSET PUSH]
                    # Aplica peso fixo aumentado na janela de quadros precursores imediatamente anterior ao onset
                    win_start = max(0, t0 - kd_window)
                    for t in range(win_start, t0):
                        w_ep[t] = pre_push_w

                    # Normalização idêntica ao protocolo original para estabilidade de gradiente
                    w_ep = w_ep / (w_ep.sum() + 1e-8)
                    weights[i] = w_ep

        l_soft = (temp ** 2) * (kl * weights).sum(dim=1).mean()
    else:
        l_soft = torch.tensor(0.0, device=student_logits.device)

    return (1 - lam_kd) * l_hard + lam_kd * l_soft

# TrainingManager

class TrainingManager:
    """Gerencia o treinamento de todos os modelos para todas as seeds."""

    LEGACY_DIR = Path("modelos_salvos") / "abl_A"

    def __init__(self, cfg: dict, exp_dir: Path, seeds: List[int],
                 device: torch.device):
        self.cfg       = cfg
        self.exp_dir   = exp_dir
        self.seeds     = seeds
        self.device    = device
        self.train_cfg = cfg.get("training", {})
        self.model_cfg = cfg.get("model", {})
        self.out_dir   = exp_dir / "models"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.D = cfg["dataset"]["input_dim"]
        self.H = self.model_cfg["hidden_dim"]
        self.window = cfg["dataset"]["window"]

    def _ckpt(self, role: str, seed: int) -> Path:
        return self.out_dir / f"{role}_seed{seed}.pt"

    def _legacy_ckpt(self, role: str, seed: int) -> Path:
        return self.LEGACY_DIR / f"{role}_seed{seed}.pt"

    def _try_reuse(self, role: str, seed: int) -> bool:
        """Copia modelo do diretório legado se existir."""
        legacy = self._legacy_ckpt(role, seed)
        dest   = self._ckpt(role, seed)
        if dest.exists():
            return True
        if legacy.exists():
            shutil.copy(legacy, dest)
            print(f"    [cache] {role}_seed{seed}.pt copiado de legacy")
            return True
        return False

    # Carregamento de dados

    def _load_split(
        self, seed: int, split: str
    ) -> Tuple[torch.Tensor, ...]:
        d = self.exp_dir / "data"
        X   = torch.tensor(np.load(d / f"X_{split}_seed{seed}.npy"),
                           dtype=torch.float32)
        y_fr = torch.tensor(np.load(d / f"y_fr_{split}_seed{seed}.npy"),
                            dtype=torch.float32)
        y_ep = torch.tensor(np.load(d / f"y_ep_{split}_seed{seed}.npy"),
                            dtype=torch.long)
        prog = torch.tensor(np.load(d / f"progressive_{split}_seed{seed}.npy"),
                            dtype=torch.bool)
        return X, y_fr, y_ep, prog

    def _onset_frames(self, y_fr: torch.Tensor) -> torch.Tensor:
        """Retorna o frame de onset para cada episódio (primeiro frame crítico)."""
        onsets = []
        for yt in y_fr:
            idx = (yt > 0.5).nonzero(as_tuple=True)[0]
            onsets.append(idx[0].item() if len(idx) > 0 else self.window - 1)
        return torch.tensor(onsets, dtype=torch.long)

    # Teacher (BiLSTM)

    def train_teacher(self, seed: int) -> MultiTaskLSTM:
        if self._try_reuse("teacher", seed):
            return self._load_model("teacher", seed, bi=True)

        print(f"    Treinando teacher BiLSTM (seed={seed})...")
        t_cfg = self.train_cfg.get("teacher", {})
        X_tr, y_fr_tr, y_ep_tr, _ = self._load_split(seed, "tr")

        model = MultiTaskLSTM(self.D, self.H, bi=True).to(self.device)
        opt   = torch.optim.Adam(model.parameters(),
                                  lr=t_cfg.get("lr", 0.001))
        loader = DataLoader(
            TensorDataset(X_tr, y_fr_tr, y_ep_tr),
            batch_size=t_cfg.get("batch_size", 64), shuffle=True
        )
        for ep in range(t_cfg.get("epochs", 60)):
            model.train()
            total_loss = 0.0
            for xb, yf, ye in loader:
                xb, yf, ye = xb.to(self.device), yf.to(self.device), ye.to(self.device)
                fr, ep_out = model(xb)
                fr_logits  = fr.squeeze(-1)         # (B, T)
                ep_logit   = ep_out.squeeze(-1)     # (B,)
                # Perda combinada: frame-level + episode-level (idêntico ao run_v29)
                loss = (F.binary_cross_entropy_with_logits(fr_logits, yf.float()) +
                        F.binary_cross_entropy_with_logits(ep_logit, ye.float())) / 2.0
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item()
            if (ep + 1) % 20 == 0:
                print(f"      ep={ep+1:3d}  loss={total_loss/len(loader):.4f}")

        torch.save(model.state_dict(), self._ckpt("teacher", seed))
        print(f"    Teacher salvo: teacher_seed{seed}.pt")
        return model

    # Baseline

    def train_baseline(self, seed: int) -> MultiTaskLSTM:
        if self._try_reuse("baseline", seed):
            return self._load_model("baseline", seed)

        print(f"    Treinando Baseline (seed={seed})...")
        b_cfg = self.train_cfg.get("baseline", {})
        X_tr, y_fr_tr, y_ep_tr, _ = self._load_split(seed, "tr")

        model = MultiTaskLSTM(self.D, self.H).to(self.device)
        opt   = torch.optim.Adam(model.parameters(), lr=b_cfg.get("lr", 0.001))
        loader = DataLoader(
            TensorDataset(X_tr, y_fr_tr, y_ep_tr),
            batch_size=b_cfg.get("batch_size", 64), shuffle=True
        )
        for ep in range(b_cfg.get("epochs", 60)):
            model.train()
            total_loss = 0.0
            for xb, yf, ye in loader:
                xb, yf, ye = xb.to(self.device), yf.to(self.device), ye.to(self.device)
                fr, ep_out = model(xb)
                fr_logits  = fr.squeeze(-1)         # (B, T)
                ep_logit   = ep_out.squeeze(-1)     # (B,)
                # Perda combinada: frame-level + episode-level (idêntico ao run_v29)
                loss = (F.binary_cross_entropy_with_logits(fr_logits, yf.float()) +
                        F.binary_cross_entropy_with_logits(ep_logit, ye.float())) / 2.0
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item()
            if (ep + 1) % 20 == 0:
                print(f"      ep={ep+1:3d}  loss={total_loss/len(loader):.4f}")

        torch.save(model.state_dict(), self._ckpt("baseline", seed))
        print(f"    Baseline salvo: baseline_seed{seed}.pt")
        return model

    # AF-TKD Student (protocolo 3 fases)

    def train_aftkd(self, seed: int, teacher: MultiTaskLSTM) -> MultiTaskLSTM:
        if self._try_reuse("student", seed):
            return self._load_model("student", seed)

        print(f"    Treinando AF-TKD Student (seed={seed})...")
        s_cfg = self.train_cfg.get("student_aftkd", {})

        # Classe Adaptadora para manter o mapeamento de Framework do config.yaml
        class ConfigAdapter:
            def __init__(self, raw_cfg):
                class SubConfig:
                    def __init__(self, d):
                        self.onset_exp_alpha = d.get("onset_exp_alpha", 0.12)
                        self.temp_fixed = d.get("temp_fixed", 2.0)
                        self.pre_onset_push_weight = d.get("pre_onset_push_weight", 3.0)
                        self.lambda_late = d.get("lambda_late", 3.5)
                        self.theta_late = d.get("theta_late", 0.15)
                        self.hidden_kd_window = d.get("hidden_kd_window", 12)
                self.af_tkd = SubConfig(raw_cfg)

        # Instancia o adaptador consumindo os dados lidos do YAML
        adapter_config = ConfigAdapter(s_cfg)

        X_tr, y_fr_tr, y_ep_tr, _ = self._load_split(seed, "tr")
        onset_tr = self._onset_frames(y_fr_tr)

        student = MultiTaskLSTM(self.D, self.H).to(self.device)
        teacher = teacher.to(self.device)
        teacher.eval()

        opt = torch.optim.Adam(student.parameters(), lr=s_cfg.get("lr", 0.001))
        loader = DataLoader(
            TensorDataset(X_tr, y_fr_tr, y_ep_tr, onset_tr),
            batch_size=s_cfg.get("batch_size", 64), shuffle=True
        )

        alpha      = s_cfg.get("onset_exp_alpha",  0.15)
        temp       = s_cfg.get("temp_fixed",        2.0)
        lam_start  = s_cfg.get("lam_start",         0.20)
        lam_end    = s_cfg.get("lam_end",            1.00)
        warmup_ep  = s_cfg.get("warmup_epochs",      10)
        rampup_ep  = s_cfg.get("rampup_epochs",      17)
        total_ep   = s_cfg.get("epochs",             80)

        for ep in range(total_ep):
            # Protocolo 3 fases
            if ep < warmup_ep:
                lam_kd = 0.0
            elif ep < warmup_ep + rampup_ep:
                frac   = (ep - warmup_ep) / rampup_ep
                lam_kd = lam_start + frac * (lam_end - lam_start)
            else:
                lam_kd = lam_end

            student.train()
            total_loss = 0.0
            for xb, yf, ye, ons in loader:
                xb  = xb.to(self.device)
                yf  = yf.to(self.device)
                ye  = ye.to(self.device)
                ons = ons.to(self.device)

                s_fr, s_ep = student(xb)
                with torch.no_grad():
                    t_fr, _    = teacher(xb)

                loss = aftkd_loss(
                    student_fr=s_fr,
                    student_ep=s_ep,
                    teacher_fr=t_fr,
                    y_episode=ye,
                    onset_frames=ons,
                    config=adapter_config,  # <--- Injeção limpa e adaptada dos parâmetros do Ajuste 1
                    lam_kd=lam_kd,
                    window=self.window,
                )
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item()

            if (ep + 1) % 20 == 0:
                phase = ("warmup" if ep < warmup_ep
                         else "rampup" if ep < warmup_ep + rampup_ep
                         else "steady")
                print(f"      ep={ep+1:3d}  λ={lam_kd:.3f}  "
                      f"loss={total_loss/len(loader):.4f}  [{phase}]")

        torch.save(student.state_dict(), self._ckpt("student", seed))
        print(f"    Student salvo: student_seed{seed}.pt")
        return student

    # Carregamento

    def _load_model(self, role: str, seed: int, bi: bool = False) -> MultiTaskLSTM:
        model = MultiTaskLSTM(self.D, self.H, bi=bi).to(self.device)
        model.load_state_dict(
            torch.load(self._ckpt(role, seed), map_location=self.device)
        )
        model.eval()
        return model

    # Orquestrador

    def run(self, reuse: bool = True) -> None:
        """Treina teacher + baseline + student para todas as seeds."""
        for seed in self.seeds:
            print(f"\n  [Training] seed={seed}")
            from tera_pipeline.utils.tera_utils import set_global_seed
            set_global_seed(seed)

            teacher  = self.train_teacher(seed)
            _        = self.train_baseline(seed)
            _        = self.train_aftkd(seed, teacher)

            n_params = count_parameters(MultiTaskLSTM(self.D, self.H))
            print(f"  Parâmetros do student (fc_fr + fc_ep, 2 camadas): {n_params:,}"
                  f"  (~{n_params*4/1024:.0f} KB FP32)")
        print("\n  Treinamento concluído para todas as seeds")
