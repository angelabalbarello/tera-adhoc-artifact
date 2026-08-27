# -*- coding: utf-8 -*-
"""
tera_pipeline/inference/tera_infer.py
═══════════════════════════════════════════════════════════════════════════════
Módulo de inferência nativa do TERA Pipeline.

CORREÇÕES APLICADAS (vs versão anterior):
  [FIX-1] infer_stream_fixed e infer_stream_hybrid usam make_prefix_window
          frame-a-frame — protocolo causal idêntico ao run_v29.
          Versão anterior alimentava o episódio inteiro (1,T,D) de uma vez,
          permitindo que o LSTM visse frames futuros → TTDef≈0, FR≈0 trivial.

  [FIX-2] episode_probs = mean(frame_probs[:, -K_AGG:]) com K_AGG=6.
          Versão anterior usava probs.max(axis=1) → detecção trivialmente fácil.
          StreamEval agora carrega campo episode_probs separado.

  [FIX-3] InferenceManager.run() roda infer_stream_fixed no VAL e salva
          frame_probs_val_{role}_seed{N}.npy + y_ep_val_seed{N}.npy.
          Tera_eval usa esses arquivos para chamar select_thr_ep corretamente.

Referência: run_v29_ablacao_ttdef_ajuste_gatting.py, funções
  make_prefix_window (linha 571), episode_probs_from_frames (linha 584),
  infer_stream_fixed (linha 1020), infer_stream_hybrid (linha 1057).
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ── Constantes de protocolo ────────────────────────────────────────────────────
# [FIX-2] K_AGG=6: agregador causal — média dos últimos 6 frames para ep_probs.
# Fonte: run_v29 linha 189, "Seção 5.1: agregador causal k=6".
K_AGG = 6


@dataclass
class StreamEval:
    """Resultado de uma passagem de inferência streaming."""
    frame_probs:       np.ndarray   # (N_episodes, T)  probabilidades frame-a-frame
    episode_probs:     np.ndarray   # (N_episodes,)    mean(last K_AGG frames) [FIX-2]
    skip_mask:         np.ndarray   # (N_episodes, T)  True = suprimido pelo MHEG
    lat_ms:            float        # latência bruta por frame (ms)
    skip_pct:          float        # % de frames suprimidos
    cost_ms_per_frame: float        # custo efetivo = lat × (1 − skip%)


# ── Funções de protocolo causal ───────────────────────────────────────────────

def binary_entropy(p: float) -> float:
    """Entropia binária H(p) em bits."""
    p = float(np.clip(p, 1e-7, 1 - 1e-7))
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


# [FIX-1] make_prefix_window — idêntico ao run_v29 linha 571.
def make_prefix_window(x_seq: np.ndarray, t: int, window: int = 96) -> np.ndarray:
    """
    Janela causal até t, com pad à esquerda pelo primeiro frame.

    No instante t, alimenta sempre `window` frames:
      · t+1 < window → pad de (window−t−1) cópias do frame 0, seguido de x_0..x_t
      · t+1 ≥ window → últimos `window` frames até t (sem pad)

    Garante que o modelo nunca veja frames futuros — simula streaming
    real no dispositivo embarcado onde no instante t só existem x_0..x_t.
    """
    x_slice = x_seq[:t + 1, :]
    if x_slice.shape[0] < window:
        pad_len   = window - x_slice.shape[0]
        pad_frame = (x_slice[0:1, :] if x_slice.shape[0] > 0
                     else np.zeros((1, x_seq.shape[1]), dtype=x_seq.dtype))
        x_slice = np.concatenate(
            [np.repeat(pad_frame, pad_len, axis=0), x_slice], axis=0)
    else:
        x_slice = x_slice[-window:]
    return x_slice.astype(np.float32, copy=False)


# [FIX-2] episode_probs_from_frames — idêntico ao run_v29 linha 584.
def episode_probs_from_frames(frame_probs: np.ndarray,
                               k: int = K_AGG) -> np.ndarray:
    """
    Probabilidade episódica = média dos últimos k frames (k=6).

    Exige que o modelo sustente probabilidade alta nos últimos k frames.
    NÃO use probs.max(axis=1): um único frame espúrio classificaria
    qualquer episódio como crítico, tornando F1≈1 e FR≈0 trivialmente.
    """
    return np.mean(frame_probs[:, -k:], axis=1)


# ── Helpers internos ──────────────────────────────────────────────────────────

def _warmup(model: nn.Module, x_ref: np.ndarray,
            device: torch.device, steps: int = 100) -> None:
    """Warm-up de 100 passos antes da medição de latência."""
    model.eval()
    xt = torch.tensor(x_ref, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(steps):
            _ = model(xt)
    if device.type == "cuda":
        torch.cuda.synchronize()


# ── Funções de inferência ─────────────────────────────────────────────────────

@torch.no_grad()
def infer_stream_fixed(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    window: int = 96,
    warmup: int = 100,
) -> StreamEval:
    """
    Inferência streaming causal sem gating (política fixa).

    [FIX-1] Loop frame-a-frame com make_prefix_window:
      · Cada passo t → make_prefix_window(X[i], t, window) → (1, window, D)
      · Saída: sigmoid(out[0, -1]) — último token da janela causal
      · Latência medida por passo (ms/frame), batch=1, warmup=100

    Nota de desempenho: T forward passes por episódio (em vez de 1).
    Isso é correto para simulação de deployment — é o custo real de streaming.
    """
    model.eval()
    N, T, _ = X.shape

    # Warm-up com janela causal de referência (frame t=5)
    x_ref = make_prefix_window(X[0], min(5, T - 1), window)
    _warmup(model, x_ref, device, steps=warmup)

    all_probs = np.zeros((N, T), dtype=np.float32)
    lat_times = []

    for ep in range(N):
        for t in range(T):
            # [FIX-1] Janela causal: modelo vê apenas frames 0..t com pad
            x_slice = make_prefix_window(X[ep], t, window)
            xt = torch.tensor(x_slice, dtype=torch.float32,
                               device=device).unsqueeze(0)   # (1, window, D)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fr_log, _ = model(xt)         # fr_log: (1, window, 1) — run_v29 format
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat_times.append(time.perf_counter() - t0)
            # Último frame da janela causal = predição do instante t
            # Indexação idêntica ao run_v29 linha 1044: fr_log[0, -1, 0]
            all_probs[ep, t] = float(torch.sigmoid(fr_log[0, -1, 0]).item())

    lat_ms    = float(np.mean(lat_times)) * 1000   # ms/frame
    skip_mask = np.zeros((N, T), dtype=bool)

    return StreamEval(
        frame_probs=all_probs,
        episode_probs=episode_probs_from_frames(all_probs, K_AGG),  # [FIX-2]
        skip_mask=skip_mask,
        lat_ms=lat_ms,
        skip_pct=0.0,
        cost_ms_per_frame=lat_ms,
    )


@torch.no_grad()
def infer_stream_hybrid(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    tau_delta: float,
    tau_h: float,
    kin_indices: List[int],
    window: int = 96,
    warmup: int = 100,
) -> StreamEval:
    """
    Inferência streaming com gating MHEG (proteção híbrida de dois gatilhos).

    MHEG — dois gatilhos complementares:
      · Gatilho cinemático: max|Δx_kin| ≥ tau_delta
      · Gatilho entrópico:  H(p_{t-1}) ≥ tau_h

    [FIX-1] Quando ativado, alimenta make_prefix_window(X[i], t, window) —
    não o frame t isolado. Preserva o contexto temporal acumulado.
    Frame 0 sempre atualizado (inicialização obrigatória).

    Protocolo idêntico ao run_v29 infer_stream_hybrid (linha 1057).
    """
    model.eval()
    N, T, _ = X.shape

    x_ref = make_prefix_window(X[0], min(5, T - 1), window)
    _warmup(model, x_ref, device, steps=warmup)

    all_probs  = np.zeros((N, T), dtype=np.float32)
    skip_mask  = np.zeros((N, T), dtype=bool)
    lat_times  = []
    n_skipped  = 0

    for ep in range(N):
        last_p = None   # força atualização no frame 0

        for t in range(T):

            # ── Frame 0: inicialização obrigatória ────────────────────────────
            if t == 0 or last_p is None:
                x_slice = make_prefix_window(X[ep], t, window)
                xt = torch.tensor(x_slice, dtype=torch.float32,
                                   device=device).unsqueeze(0)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                fr_log, _ = model(xt)        # (1, window, 1)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                lat_times.append(time.perf_counter() - t0)
                last_p = float(torch.sigmoid(fr_log[0, -1, 0]).item())
                all_probs[ep, t] = last_p
                continue

            # ── Sinais de gating ───────────────────────────────────────────────
            dk = float(np.abs(
                X[ep, t, kin_indices] - X[ep, t - 1, kin_indices]
            ).max())
            H  = binary_entropy(last_p)

            # ── Decisão OR ────────────────────────────────────────────────────
            activate = (dk >= tau_delta) or (H >= tau_h)

            if not activate:
                # SKIP: propaga probabilidade anterior
                all_probs[ep, t] = last_p
                skip_mask[ep, t] = True
                n_skipped += 1
            else:
                # UPDATE: [FIX-1] janela causal completa (não frame isolado)
                x_slice = make_prefix_window(X[ep], t, window)
                xt = torch.tensor(x_slice, dtype=torch.float32,
                                   device=device).unsqueeze(0)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                fr_log, _ = model(xt)        # (1, window, 1)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                lat_times.append(time.perf_counter() - t0)
                last_p = float(torch.sigmoid(fr_log[0, -1, 0]).item())
                all_probs[ep, t] = last_p

    total_frames = N * T
    skip_pct     = float(n_skipped) / total_frames * 100.0
    lat_ms       = float(np.mean(lat_times)) * 1000.0 if lat_times else 0.0

    return StreamEval(
        frame_probs=all_probs,
        episode_probs=episode_probs_from_frames(all_probs, K_AGG),  # [FIX-2]
        skip_mask=skip_mask,
        lat_ms=lat_ms,
        skip_pct=skip_pct,
        cost_ms_per_frame=lat_ms * (1.0 - skip_pct / 100.0),
    )


def _measure_lat_fixed(
    model: nn.Module, X_sample: np.ndarray,
    device: torch.device, window: int = 96, warmup: int = 20,
) -> float:
    """
    Mede latência bruta por frame em modo batch=1.
    [FIX-1] Usa make_prefix_window para latência real de deployment.
    """
    model.eval()
    x_ref = make_prefix_window(
        X_sample[0], min(5, X_sample.shape[1] - 1), window)
    xt = torch.tensor(x_ref, dtype=torch.float32, device=device).unsqueeze(0)
    for _ in range(warmup):
        _ = model(xt)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(50):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(xt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)) * 1000.0


# ── InferenceManager ──────────────────────────────────────────────────────────

class InferenceManager:
    """
    Gerencia a inferência de todas as (seed, config) do fatorial 2×2.

    Garante:
      · [FIX-1] Inferência causal via make_prefix_window
      · [FIX-2] episode_probs via mean(last K_AGG=6 frames)
      · [FIX-3] VAL frame_probs salvas para select_thr_ep em tera_eval
      · Baseline Hybrid nativa (sem patch)
      · Mesmos tau_delta/tau_H para Baseline e AF-TKD Hybrid por seed
      · Cache de frame_probs para evitar re-inferência
    """

    OFFLINE_COMPONENTS = {"teacher", "af_tkd_protocol", "tera_gen", "calibration"}
    ONLINE_COMPONENTS  = {"student_causal", "mheg", "hidden_state", "thresholds"}

    def __init__(self, cfg: dict, exp_dir: Path, seeds: List[int],
                 device: torch.device):
        self.cfg        = cfg
        self.exp_dir    = exp_dir
        self.seeds      = seeds
        self.device     = device
        self.eval_cfg   = cfg.get("evaluation", {})
        self.model_cfg  = cfg.get("model", {})
        self.kin        = self.model_cfg.get("kinematic_indices", [12, 13, 14])
        self.window     = cfg.get("dataset", {}).get("window", 96)
        self.models_dir = Path(
            cfg.get("training", {}).get("models_dir", "modelos_salvos"))

    def _load_model(self, seed: int, role: str) -> nn.Module:
        from tera_pipeline.training.tera_train import MultiTaskLSTM
        D = self.cfg["dataset"]["input_dim"]
        H = self.model_cfg["hidden_dim"]
        ckpt = self.models_dir / "abl_A" / f"{role}_seed{seed}.pt"
        if not ckpt.exists():
            ckpt = self.exp_dir / "models" / f"{role}_seed{seed}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint não encontrado: {ckpt}\n"
                "Execute --stages training primeiro.")
        model = MultiTaskLSTM(D, H, bi=False).to(self.device)
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.eval()
        return model

    def _load_calibration(self, seed: int) -> dict:
        cal_path = self.exp_dir / "calibration" / "calibration_summary.csv"
        if cal_path.exists():
            import pandas as pd
            df = pd.read_csv(cal_path)
            row = df[df["seed"] == seed]
            if not row.empty:
                return row.iloc[0].to_dict()
        return {}

    def _load_data(self, seed: int,
                   split: str = "te") -> Tuple[np.ndarray, ...]:
        """Carrega X, y_fr, y_ep, prog para o split pedido (te | va | tr)."""
        d = self.exp_dir / "data"
        X    = np.load(d / f"X_{split}_seed{seed}.npy")
        y_fr = np.load(d / f"y_fr_{split}_seed{seed}.npy")
        y_ep = np.load(d / f"y_ep_{split}_seed{seed}.npy")
        prog = np.load(d / f"progressive_{split}_seed{seed}.npy")
        return X, y_fr, y_ep, prog

    def _cache_path(self, seed: int, config_id: str, key: str) -> Path:
        return self.exp_dir / "metrics" / f"{key}_{config_id}_seed{seed}.npy"

    def _is_cached(self, seed: int, config_id: str) -> bool:
        return self._cache_path(seed, config_id, "frame_probs").exists()

    def _val_probs_cached(self, seed: int) -> bool:
        base = self.exp_dir / "metrics"
        return (
            (base / f"frame_probs_val_baseline_seed{seed}.npy").exists() and
            (base / f"frame_probs_val_student_seed{seed}.npy").exists() and
            (base / f"y_ep_val_seed{seed}.npy").exists()
        )

    def run(self, configs: List[str]) -> None:
        """
        Executa inferência para todas as (seed, config).

        Para cada seed:
          1. Carrega baseline e student
          2. Carrega X_te (TEST) e X_va (VAL)
          3. Carrega calibração de gating (tau_delta, tau_H) — apenas gating
          4. [FIX-3] Roda infer_stream_fixed no VAL, salva VAL frame_probs
             → tera_eval usa esses arquivos em select_thr_ep
          5. Executa 4 configs do fatorial 2×2 no TEST
          6. Salva frame_probs, episode_probs, skip_mask por config
        """
        import pandas as pd

        for seed in self.seeds:
            print(f"\n  [Inference] Seed {seed}")
            X_te, y_fr_te, y_ep_te, prog_te = self._load_data(seed, "te")
            cal = self._load_calibration(seed)

            baseline = self._load_model(seed, "baseline")
            student  = self._load_model(seed, "student")

            tau_delta = float(cal.get("tau_delta", 0.02))
            tau_h     = float(cal.get("tau_h",     0.95))
            print(f"    Gating params: tau_δ={tau_delta:.4f}, tau_H={tau_h:.2f}")

            # ── [FIX-3] VAL inference para thr_ep calibration ─────────────────
            metrics_dir = self.exp_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)

            if not self._val_probs_cached(seed):
                print(f"    [FIX-3] Inferência VAL (thr_ep calibration)...")
                X_va, _, y_ep_va, _ = self._load_data(seed, "va")

                val_base = infer_stream_fixed(
                    baseline, X_va, device=self.device, window=self.window)
                np.save(metrics_dir / f"frame_probs_val_baseline_seed{seed}.npy",
                        val_base.frame_probs)

                val_stud = infer_stream_fixed(
                    student, X_va, device=self.device, window=self.window)
                np.save(metrics_dir / f"frame_probs_val_student_seed{seed}.npy",
                        val_stud.frame_probs)

                np.save(metrics_dir / f"y_ep_val_seed{seed}.npy", y_ep_va)
                print(f"    [FIX-3] VAL probs salvas (seed={seed})")
            else:
                print(f"    [FIX-3] VAL probs — cache OK (seed={seed})")

            # ── TEST inference por config ──────────────────────────────────────
            for config_id in configs:
                if self._is_cached(seed, config_id):
                    print(f"    [cache] {config_id} — pulando")
                    continue

                print(f"    → {config_id}")
                cfg_spec   = self.eval_cfg.get("configs", {}).get(config_id, {})
                model_role = cfg_spec.get("model", "baseline")
                use_gating = bool(cfg_spec.get("gating", False))
                model      = student if model_role == "student" else baseline

                if use_gating:
                    result = infer_stream_hybrid(
                        model=model, X=X_te, device=self.device,
                        tau_delta=tau_delta, tau_h=tau_h,
                        kin_indices=self.kin, window=self.window,
                    )
                else:
                    result = infer_stream_fixed(
                        model=model, X=X_te,
                        device=self.device, window=self.window,
                    )

                # Salva resultados por config
                np.save(self._cache_path(seed, config_id, "frame_probs"),
                        result.frame_probs)
                np.save(self._cache_path(seed, config_id, "episode_probs"),
                        result.episode_probs)   # [FIX-2]
                np.save(self._cache_path(seed, config_id, "skip_mask"),
                        result.skip_mask)

                # Ground truth compartilhado (idempotente)
                np.save(metrics_dir / f"y_true_fr_seed{seed}.npy",   y_fr_te)
                np.save(metrics_dir / f"y_episode_seed{seed}.npy",   y_ep_te)
                np.save(metrics_dir / f"progressive_mask_seed{seed}.npy", prog_te)

                print(f"      FR_skip={result.skip_pct:.1f}%  "
                      f"cost={result.cost_ms_per_frame:.3f}ms/q")

            # ── Latência de referência por role ───────────────────────────────
            # [FIX-LAT] Medida separada para baseline e student.
            # tera_eval usa lat_ms_baseline / lat_ms_student conforme o config.
            # O arquivo é sobrescrito a cada seed para manter a última medição.
            lat_baseline = _measure_lat_fixed(
                baseline, X_te[:1], self.device, window=self.window)
            lat_student  = _measure_lat_fixed(
                student,  X_te[:1], self.device, window=self.window)
            lat_ref = {
                "lat_ms_baseline": round(lat_baseline, 4),
                "lat_ms_student":  round(lat_student,  4),
                # Mantido para retrocompatibilidade com versões anteriores:
                "lat_ms_per_frame": round(lat_student, 4),
            }
            # [FIX-STD] Salva referência indexada por seed para que tera_eval
            # leia o valor correto por seed (elimina std=0.000 nas macros).
            lat_path_seed = metrics_dir / f"latency_reference_seed{seed}.json"
            lat_path_seed.write_text(json.dumps(lat_ref, indent=2))
            # Também mantém o arquivo genérico para retrocompatibilidade.
            lat_path = metrics_dir / "latency_reference.json"
            lat_path.write_text(json.dumps(lat_ref, indent=2))
