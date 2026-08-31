import numpy as np
import torch
import pandas as pd
from pathlib import Path

from synthetic_driver_risk_v7 import quick_generate
from run_v29_ablacao_ttdef_ajuste_gatting import (   # <- atualizado: v29
    MultiTaskLSTM,
    make_prefix_window,
    binary_entropy,
    infer_stream_fixed,           # necessário para calibrar tau_delta
)

# CONFIG
SEED       = 42
T          = 96
THETA_TTD  = 0.10    # limiar explícito, alinhado com o artigo
FRAME_THR  = 0.35    # limiar de onset igual ao protocolo principal
KIN        = [12, 13, 14]   # índices cinemáticos — igual ao run_v29

# Parâmetros de gating extraídos de exp_abl_A_results_all_seeds.csv
# Coluna Politica: "Hibrida (tau=0.0190, H=0.95)" para Seed=42
# Fonte: run_v29, select_hybrid_policy calibrado no VAL com FR=0 + max SkipPct
TAU_DELTA  = 0.0190   # percentil dos |Δx_KIN| no conjunto de validação
TAU_H      = 0.95     # entropia mínima para ativar o núcleo causal

# Critério quádruplo de seleção do episódio (alinhado com o artigo)
MIN_DELAY_BASE  = 2   # delay_base >= 2 frames
MIN_GAIN_KD     = 1   # gain_KD   >= 1 frame
MIN_SKIP_PCT    = 40  # skip%     >= 40%

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR  = Path(".")
MODEL_DIR = BASE_DIR / "modelos_salvos" / "abl_A"   # subpasta da ablação A
OUT_DIR   = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

# REFERÊNCIA DOS PARÂMETROS POR SEED (exp_abl_A_results_all_seeds.csv)
# Seed 42: tau_delta=0.0190, tau_H=0.95, SkipPct=40.1%, FR=0.000
# Seed 43: tau_delta=0.0460, tau_H=0.95, SkipPct=66.3%, FR=0.000
# Seed 44: tau_delta=0.0394, tau_H=0.95, SkipPct=65.4%, FR=0.000
# Seed 45: tau_delta=0.0192, tau_H=0.95, SkipPct=42.7%, FR=0.000

# 1. GERAR / REUTILIZAR DATASET
print("=" * 60)
print("ETAPA 1 — Dataset")

npz_path = OUT_DIR / f"dataset_seed{SEED}.npz"
csv_path = OUT_DIR / f"dataset_seed{SEED}.csv"

if npz_path.exists() and csv_path.exists():
    print(f"  Reutilizando dataset: {npz_path.name}")
else:
    print(f"  Gerando dataset (seed={SEED})...")
    quick_generate(
        out_npz=str(npz_path),
        out_csv=str(csv_path),
        seed=SEED,
        window=T,
        per_recipe=80,
    )

data    = np.load(npz_path)
X       = data["X"].astype(np.float32)
y_frame = data["y_frame_clean"].astype(np.float32) \
          if "y_frame_clean" in data.files \
          else data["y_frame"].astype(np.float32)

meta    = pd.read_csv(csv_path, sep=";")

# 2. SPLIT 70/15/15 — REPRODUÇÃO EXATA DO run_v29
print("\nETAPA 2 — Split")
from sklearn.model_selection import train_test_split

# Usar split estratificado por recipe_id, igual ao run_v29
class_col = next(c for c in ("class_name", "categoria", "class") if c in meta.columns)
strat_key = np.where(
    meta[class_col].astype(str).values == "Critico",
    meta["recipe_id"].astype(str).values,
    meta[class_col].astype(str).values,
)
idx = np.arange(len(X))
tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=strat_key)
va_idx, te_idx  = train_test_split(tmp_idx, test_size=0.50, random_state=SEED, stratify=strat_key[tmp_idx])

X_va,   y_va   = X[va_idx],  y_frame[va_idx]
X_te,   y_te   = X[te_idx],  y_frame[te_idx]

# y_episode: crítico se max(y_frame) > FRAME_THR
yep_te = (y_te.max(axis=1) > FRAME_THR).astype(int)

# flag progressivo do metadata
if "progressive" in meta.columns:
    prog_te = meta.iloc[te_idx]["progressive"].fillna(0).astype(int).values
else:
    prog_te = np.zeros(len(te_idx), dtype=int)

print(f"  Teste: {len(te_idx)} episódios | {yep_te.sum()} críticos")

# 3. CARREGAR MODELOS
print("\nETAPA 3 — Modelos")

baseline = MultiTaskLSTM(31, 64, bi=False).to(DEVICE)
student  = MultiTaskLSTM(31, 64, bi=False).to(DEVICE)

ckpt_base = MODEL_DIR / f"baseline_seed{SEED}.pt"
ckpt_stud = MODEL_DIR / f"student_seed{SEED}.pt"

if not ckpt_base.exists() or not ckpt_stud.exists():
    raise FileNotFoundError(
        f"Modelos não encontrados em {MODEL_DIR}.\n"
        f"Execute o run_v29 com ABLATION_MODE='A' e SEED={SEED} primeiro."
    )

baseline.load_state_dict(torch.load(ckpt_base, map_location=DEVICE))
student.load_state_dict( torch.load(ckpt_stud, map_location=DEVICE))
baseline.eval()
student.eval()
print(f"  baseline: {ckpt_base.name}")
print(f"  student : {ckpt_stud.name}")

# 4. SELEÇÃO DO EPISÓDIO — CRITÉRIO QUÁDRUPLO (alinhado com o artigo)
print("\nETAPA 4 — Seleção do episódio (critério quádruplo)")

m_detect  = 3       # m=3 consecutivos acima de THETA_TTD
dt        = 10.0 / T

def compute_ttd_frames(probs_ep, y_true_ep, theta, m=3):
    """
    TTD em frames para um único episódio.
    - onset via y_true_ep (ground truth), NÃO via probs_ep.
    BUG anterior: argmax(probs>0.35)=0 quando probs nunca passam 0.35,
    zerando delay para todos os episódios.
    """
    onset = int(np.argmax(y_true_ep > FRAME_THR))   # CORRETO: ground truth
    if y_true_ep[onset] <= FRAME_THR:
        return T   # episódio não é realmente crítico neste frame
    for t in range(onset, T - m + 1):
        if all(probs_ep[t + k] >= theta for k in range(m)):
            return t - onset
    return T

def compute_skip_pct(x_seq, tau_d, tau_h, y_prev_init=0.0):
    skips = 0
    y_prev = y_prev_init
    for t in range(T):
        delta = float(np.abs(x_seq[t, KIN] - x_seq[t-1, KIN]).max()) if t > 0 else 0.0
        H = binary_entropy(y_prev)
        update = (delta >= tau_d) or (H >= tau_h)
        if not update:
            skips += 1
        else:
            x_t = make_prefix_window(x_seq, t)
            xt  = torch.tensor(x_t, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                fr_s, _ = student(xt)
                y_prev  = torch.sigmoid(fr_s[0, -1]).item()
    return 100.0 * skips / T

# Pré-computa probabilidades frame a frame para todos os episódios críticos
crit_idx = np.where(yep_te == 1)[0]

EP = None
for i in crit_idx:
    # só progressivos (artigo especifica episódio progressivo)
    if prog_te[i] != 1:
        continue

    x_seq_i = X_te[i]
    y_ep_i  = y_te[i]

    # probabilidades frame a frame
    pb_i, ps_i = [], []
    for t in range(T):
        x_t = make_prefix_window(x_seq_i, t)
        xt  = torch.tensor(x_t, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            fb, _ = baseline(xt)
            fs, _ = student(xt)
            pb_i.append(torch.sigmoid(fb[0, -1]).item())
            ps_i.append(torch.sigmoid(fs[0, -1]).item())

    pb_i = np.array(pb_i)
    ps_i = np.array(ps_i)

    delay_base = compute_ttd_frames(pb_i, y_ep_i, THETA_TTD)
    delay_kd   = compute_ttd_frames(ps_i, y_ep_i, THETA_TTD)
    gain_kd    = delay_base - delay_kd
    skip_pct   = compute_skip_pct(x_seq_i, TAU_DELTA, TAU_H)

    print(f"  ep={i:3d} | delay_base={delay_base:3d} gain_kd={gain_kd:3d} "
          f"skip%={skip_pct:.1f}%", end="")

    # Critério quádruplo
    if (delay_base >= MIN_DELAY_BASE and
        gain_kd    >= MIN_GAIN_KD    and
        skip_pct   >= MIN_SKIP_PCT):
        print("  ok SELECIONADO")
        EP         = i
        p_base_ep  = pb_i
        p_afkd_ep  = ps_i
        break
    else:
        print()

if EP is None:
    raise RuntimeError(
        "Nenhum episódio satisfez o critério quádruplo.\n"
        "Verifique os valores de TAU_DELTA, TAU_H e os limiares MIN_*."
    )

x_seq = X_te[EP]
y_ep  = y_te[EP]

# onset via FRAME_THR (= protocolo principal)
onset_frame = int(np.argmax(y_ep > FRAME_THR))
onset_time  = onset_frame * dt

print(f"\n  Episódio selecionado: idx={EP}")
print(f"  onset_frame={onset_frame} | onset_time={onset_time:.2f}s")

# 5. INFERÊNCIA FRAME A FRAME COM GATING
print("\nETAPA 5 — Inferência com gating")
print(f"  tau_delta={TAU_DELTA} | tau_H={TAU_H} | theta_TTD={THETA_TTD}")

# gating usa subconjunto cinemático KIN (igual ao run_v29)
# delta = max sobre KIN (não mean sobre todos os features)
p_hibr  = []
gating  = []
y_prev  = 0.0

cost_base_cum = []
cost_afkd_cum = []
cost_hibr_cum = []
c_hibr = 0

for t in range(T):
    # gating decision (usa KIN e max — igual ao run_v29)
    delta  = float(np.abs(x_seq[t, KIN] - x_seq[t-1, KIN]).max()) if t > 0 else 0.0
    H      = binary_entropy(y_prev)
    update = (delta >= TAU_DELTA) or (H >= TAU_H)

    if update:
        y_curr  = p_afkd_ep[t]   # reutiliza prob já computada acima
        c_hibr += 1
    else:
        y_curr = y_prev

    p_hibr.append(y_curr)
    gating.append(int(update))
    y_prev = y_curr

    cost_base_cum.append(t + 1)         # sempre processa
    cost_afkd_cum.append(t + 1)         # sempre processa (sem gating)
    cost_hibr_cum.append(c_hibr)        # só frames ativos

skip_pct_final = 100.0 * gating.count(0) / T
print(f"  Gating ativo: {sum(gating)}/{T} frames ({100-skip_pct_final:.1f}% ativos)")
print(f"  Skip%: {skip_pct_final:.1f}%")

# 6. MÉTRICAS TTD DO EPISÓDIO
print("\nETAPA 6 — TTD do episódio")

def ttd_seconds(probs, theta, onset_f, m=3):
    """TTD em segundos a partir de um onset_f já calculado via ground truth."""
    for t in range(onset_f, T - m + 1):
        if all(probs[t + k] >= theta for k in range(m)):
            return (t - onset_f) * dt
    return (T - onset_f) * dt   # penalidade máxima (não detectado)

ttd_base_s = ttd_seconds(p_base_ep, THETA_TTD, onset_frame)
ttd_kd_s   = ttd_seconds(p_afkd_ep, THETA_TTD, onset_frame)
ttd_det_base = onset_time + ttd_base_s
ttd_det_kd   = onset_time + ttd_kd_s

print(f"  TTD Baseline : {ttd_base_s:.4f}s | t_det = {ttd_det_base:.2f}s")
print(f"  TTD AF-KD    : {ttd_kd_s:.4f}s   | t_det = {ttd_det_kd:.2f}s")
print(f"  Ganho        : {ttd_base_s - ttd_kd_s:.4f}s ({ttd_base_s - ttd_kd_s:.2f}s de antecipação)")

# custo acumulado em ms (usa Lat_ms do run_v29; ajuste se diferente)
LAT_BASE_MS = 0.885   # ms/frame — macro \resBaseFixoLat
LAT_KD_MS   = 0.915   # ms/frame — macro \resAlvoFixoLat

cost_base_ms = [c * LAT_BASE_MS for c in cost_base_cum]
cost_afkd_ms = [c * LAT_KD_MS   for c in cost_afkd_cum]
cost_hibr_ms = [c * LAT_KD_MS   for c in cost_hibr_cum]

pct_reducao = 100.0 * (1.0 - cost_hibr_ms[-1] / cost_afkd_ms[-1])
print(f"\n  Custo final Baseline : {cost_base_ms[-1]:.1f} ms")
print(f"  Custo final AF-KD    : {cost_afkd_ms[-1]:.1f} ms")
print(f"  Custo final Gating   : {cost_hibr_ms[-1]:.1f} ms")
print(f"  Redução de custo     : {pct_reducao:.1f}%")

# 7. EXPORTAR CSV
print("\nETAPA 7 — Exportação")

df = pd.DataFrame({
    "frame":          np.arange(T),
    "time_s":         np.arange(T) * dt,
    "p_base":         p_base_ep,
    "p_afkd":         p_afkd_ep,
    "p_hibrida":      p_hibr,
    "gating":         gating,
    "cost_base_ms":   cost_base_ms,
    "cost_afkd_ms":   cost_afkd_ms,
    "cost_hibr_ms":   cost_hibr_ms,
    "onset_frame":    onset_frame,
    "onset_time_s":   onset_time,
    "ttd_base_s":     ttd_base_s,
    "ttd_kd_s":       ttd_kd_s,
    "t_det_base_s":   ttd_det_base,
    "t_det_kd_s":     ttd_det_kd,
    "theta_ttd":      THETA_TTD,
    "tau_delta":      TAU_DELTA,
    "tau_H":          TAU_H,
    "skip_pct":       skip_pct_final,
    "ep_idx_te":      EP,
})

output_file = OUT_DIR / f"gating_temporal_seed{SEED}_corrigido.csv"
df.to_csv(output_file, index=False)
print(f"  CSV salvo: {output_file}")

# Resumo final
print("\n" + "=" * 60)
print("RESUMO — valores para a figura")
print(f"  onset_time   = {onset_time:.2f}s  (frame {onset_frame})")
print(f"  t_det_KD     = {ttd_det_kd:.2f}s  (TTD ≈ {ttd_kd_s:.2f}s)")
print(f"  t_det_Base   = {ttd_det_base:.2f}s  (TTD = {ttd_base_s:.2f}s)")
print(f"  skip%        = {skip_pct_final:.1f}%")
print(f"  custo_base   = {cost_base_ms[-1]:.0f}ms")
print(f"  custo_gating = {cost_hibr_ms[-1]:.0f}ms")
print(f"  reducao%     = {pct_reducao:.1f}%")
print("=" * 60)
