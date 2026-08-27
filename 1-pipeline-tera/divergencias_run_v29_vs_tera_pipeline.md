# Relatório de Divergências: run_v29 → TERA Pipeline
## Comparação run_v29 / synthetic_driver_risk_v7  vs  tera_gen / tera_train / tera_eval / tera_infer

---

## DIAGNÓSTICO PRINCIPAL

Os resultados impossíveis do novo pipeline (F1=1.000, FR=0.000, TTDef=0.000)
são explicados por TRÊS bugs estruturais. Não são problemas de configuração —
são diferenças funcionais de protocolo.

---

## BUG 1 — CRÍTICO: Inferência Não-Causal (tera_infer.py)

### run_v29 — correto: frame-a-frame com make_prefix_window
```python
for t in range(T):
    x_slice = make_prefix_window(X[i], t, window)  # pad até T, vê só t=0..t
    out, _ = model(x_slice_tensor)
    frame_probs[ep, t] = sigmoid(out[0, -1])        # só o último frame
```
O modelo no instante t vê APENAS os frames 0..t (com pad). Isso é streaming
causal real.

### tera_infer.py — errado: episódio inteiro de uma vez
```python
out, _ = model(x_ep)          # x_ep = (1, T, D) — episódio completo
probs = sigmoid(out[0])        # todos os T frames de uma vez
```
O LSTM no instante t pode fazer backpropagation implícita sobre frames
futuros t+1..T porque recebe a sequência completa. Isso quebra a
causalidade: o modelo "vê o futuro" durante a inferência.

### Consequência direta
Com a sequência completa, o LSTM produz probabilidades altas ANTES do onset
porque o estado oculto já processou os frames de risco. O resultado é
FR=0 e TTDef≈0 para TODAS as configurações, inclusive o baseline.

### Correção
```python
# tera_infer.py — substituir infer_stream_fixed por:
for ep in range(N):
    for t in range(T):
        x_slice = make_prefix_window(X[ep], t, window=T)  # importar do run_v29
        x_t = torch.tensor(x_slice, dtype=torch.float32,
                            device=device).unsqueeze(0)   # (1, T, D)
        out, _ = model(x_t)
        all_probs[ep, t] = torch.sigmoid(out[0, -1]).cpu().item()
```

---

## BUG 2 — CRÍTICO: Agregação de Probabilidade Episódica (tera_eval.py)

### run_v29 — correto: mean dos últimos K_AGG=6 frames
```python
K_AGG = 6
def episode_probs_from_frames(frame_probs, k=K_AGG):
    return np.mean(frame_probs[:, -k:], axis=1)  # média dos 6 últimos frames
```
Isso exige que o modelo sustente probabilidade alta nos últimos 6 frames
para classificar o episódio como crítico.

### tera_eval.py — errado: max sobre todos os frames
```python
ep_probs = probs.max(axis=1)   # máximo em qualquer frame
ep_hat   = (ep_probs >= thr_ep).astype(int)
```
Com max(), basta UM frame com p > thr_ep para classificar como crítico.
Combinado com thr_ep=0.05 (veja Bug 3), praticamente qualquer episódio
é classificado como crítico → F1≈1.0, FR≈0.

### Correção
```python
K_AGG = 6
ep_probs = np.mean(probs[:, -K_AGG:], axis=1)  # mean dos últimos 6 frames
ep_hat   = (ep_probs >= thr_ep).astype(int)
```

---

## BUG 3 — CRÍTICO: Threshold thr_ep=0.05 (calibração desconectada)

### run_v29 — correto: calibrado no VAL maximizando F1 com FR ≤ FAIL_BUDGET
```python
def select_thr_ep(frame_probs, y_true_ep):
    ep_probs = episode_probs_from_frames(frame_probs, k=K_AGG)  # mean k=6
    for thr in np.arange(0.01, 0.99, 0.01):
        pred = (ep_probs >= thr).astype(int)
        fr   = fail_rate(y_true_ep, pred)
        if fr <= FAIL_BUDGET:  # 0.05
            feasible.append((thr, f1_score(y_true_ep, pred)))
    # seleciona thr que maximiza F1 dentro do budget
```

### novo pipeline — errado: calibration_summary.csv tem thr_ep=0.050 fixo
```
thr_ep=0.050  theta_ttd=0.100  τΔ=0.0343  τH=0.95
```
thr_ep=0.05 é o valor do FAIL_BUDGET, não o threshold de classificação.
O tera_calib provavelmente confundiu as duas variáveis ou não implementou
o select_thr_ep. Com thr_ep=0.05 e ep_probs=max(), todo episódio
com qualquer frame acima de 5% é classificado como crítico.

### Correção
Implementar select_thr_ep em tera_calib usando episode_probs_from_frames
(mean k=6) sobre o VAL, não o limiar fixo.

---

## BUG 4 — IMPORTANTE: Assinatura Forward do MultiTaskLSTM

### run_v29
```python
def forward(self, x, return_hidden=False):
    h, _ = self.lstm(x)
    fr = self.fc_fr(h)           # (B, T, 1)
    ep = self.fc_ep(h.mean(1))   # (B, 1) — head episódico
    return fr, ep
```
Retorna DOIS tensores: logits por frame + logit episódico.

### tera_train.py
```python
def forward(self, x, h=None):
    out, hc = self.lstm(x, h)
    frame_logits = self.fc_fr(out).squeeze(-1)   # (batch, seq)
    return frame_logits, hc   # retorna estado oculto, NÃO logit episódico
```
Retorna frame_logits + hidden state tuple. O fc_ep existe mas nunca é
chamado no forward.

### Consequência
Os checkpoints do run_v29 têm pesos para fc_ep que são incompatíveis
com o forward do tera_train. Ao carregar `model.load_state_dict()` com
strict=True, pode silenciosamente aceitar (se as keys existem) mas
nunca usar fc_ep. Ao carregar com strict=False, os pesos fc_ep são
descartados sem aviso.

Verificar se os modelos legacy carregam sem KeyError. Se os pesos fc_ep
existem nos checkpoints mas não são usados, o modelo funciona (só
fc_fr é chamado), mas a semântica de episode-level é diferente.

### Ação
Adicionar fc_ep ao forward do tera_train, ou confirmar que o pipeline
inteiro usa apenas fc_fr. O run_v29 usa fc_ep implicitamente via
episode_probs_from_frames (que opera sobre os frame logits, não fc_ep
diretamente). Se o pipeline novo também usa apenas fc_fr para inferência,
a assinatura está funcionalmente OK mas semanticamente inconsistente.

---

## BUG 5 — MENOR: Cálculo de TTDef em tera_eval.py

### run_v29
```python
ttdef = ttd + fr * T_MAX_EF   # TTD_det (apenas detectados) + penalidade
```

### tera_eval.py
```python
ttd_det = mean([r.ttd_seconds for r in results if r.ttd_seconds is not None])
          if n_detected > 0 else 0.0
ttdef   = compute_ttdef(ttd_det, fail_rate, t_max)  # mesma fórmula
```
A fórmula está correta, mas com Bug 1+2+3, FR=0 e TTD_det=0 → TTDef=0.
Com os Bugs 1-3 corrigidos, o cálculo deve estar OK.

---

## BUG 6 — MENOR: Skip Rate 13.8% vs esperado ~50%

Os τ_delta calibrados pelo tera_calib (0.0267–0.0366) são DIFERENTES
dos valores do run_v29 (seed 42: 0.0190, seed 43: 0.0460). Isso
indica que a calibração de gating não está usando o mesmo protocolo
de seleção Pareto-ótima do run_v29.

run_v29 calibra τ_Δ para maximizar SkipPct sujeito a FR=0 no VAL.
O tera_calib parece usar um critério diferente (possivelmente um
percentil fixo dos |Δx_KIN|), resultando em τ_delta diferentes e
skip rates menores.

---

## TABELA-RESUMO

| # | Arquivo         | Função/Variável              | run_v29          | TERA Pipeline         | Impacto  |
|---|-----------------|------------------------------|------------------|-----------------------|----------|
| 1 | tera_infer.py   | infer_stream_fixed           | make_prefix_win  | episódio inteiro      | CRÍTICO  |
| 2 | tera_eval.py    | ep_probs                     | mean(last_k=6)   | max(all_frames)       | CRÍTICO  |
| 3 | calibração      | thr_ep                       | select_thr_ep()  | 0.050 fixo            | CRÍTICO  |
| 4 | tera_train.py   | MultiTaskLSTM.forward        | retorna fr, ep   | retorna fr, hc        | MÉDIO    |
| 5 | tera_eval.py    | TTDef                        | correto (fórmula)| correto (fórmula)     | OK*      |
| 6 | tera_calib      | τ_delta por seed             | Pareto-ótimo     | critério diferente    | MENOR    |

*OK desde que Bugs 1-3 sejam corrigidos.

---

## PLANO DE CORREÇÃO (ordem de prioridade)

1. **tera_infer.py** — substituir batch forward por make_prefix_window frame-a-frame
   (copiar lógica do run_v29 `infer_stream_fixed` e `infer_stream_hybrid`)

2. **tera_eval.py** — substituir `probs.max(axis=1)` por `mean(probs[:, -6:], axis=1)`

3. **tera_calib** — implementar `select_thr_ep` igual ao run_v29:
   calibra thr_ep no VAL por episode_probs_from_frames(k=6),
   maximizando F1 com FR ≤ 0.05

4. **Verificar** MultiTaskLSTM.forward: confirmar que os checkpoints
   carregam sem KeyError e que fc_ep não impacta os resultados

5. **Re-executar** calibração e inferência com modelos legacy para
   confirmar reprodução dos resultados do paper

---

## VERIFICAÇÃO RÁPIDA (antes de re-executar tudo)

```python
# Teste manual de causalidade — seed 42, episódio 0
import numpy as np, torch
from tera_pipeline.training.tera_train import MultiTaskLSTM

# Carregar modelo
model = MultiTaskLSTM(31, 64, bi=False)
model.load_state_dict(torch.load("modelos_salvos/abl_A/student_seed42.pt"))
model.eval()

# Carregar dados
X_te = np.load("results/exp_.../data/X_te_seed42.npy")

# Comparar: causal vs batch
from run_v29_ablacao_ttdef_ajuste_gatting import make_prefix_window
probs_causal = []
for t in range(96):
    x = torch.tensor(make_prefix_window(X_te[0], t, 96), dtype=torch.float32).unsqueeze(0)
    out, _ = model(x)
    probs_causal.append(torch.sigmoid(out[0, -1]).item())

probs_batch = torch.sigmoid(model(torch.tensor(X_te[0:1], dtype=torch.float32))[0][0]).detach().numpy()

print("Causal max:", max(probs_causal))
print("Batch  max:", probs_batch.max())
# Se forem diferentes → Bug 1 confirmado
```
