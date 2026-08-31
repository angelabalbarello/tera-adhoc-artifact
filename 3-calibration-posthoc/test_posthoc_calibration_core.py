"""Testa a lógica core de calibração sem dependências do pipeline."""
import numpy as np
import torch
import math
from scipy.optimize import minimize_scalar
from scipy.special import expit as scipy_sigmoid
from sklearn.isotonic import IsotonicRegression

EPSILON = 1e-7
T_MAX_EF = 10.0
TTD_M = 3
K_AGG = 6
FRAME_THR = 0.35
WINDOW = 96
DT = 10.0 / WINDOW


def binary_entropy(p):
    p = float(np.clip(p, EPSILON, 1.0 - EPSILON))
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def _logit(p):
    p = np.clip(p, EPSILON, 1.0 - EPSILON)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return scipy_sigmoid(z)


def expected_calibration_error(y_true, y_pred, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_pred >= lo) & (y_pred < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(float(y_true[mask].mean()) - float(y_pred[mask].mean()))
    return float(ece / max(len(y_true), 1))


class TemperatureScaler:
    def __init__(self):
        self.T = 1.0
        self._fitted = False

    def fit(self, probs_val, y_val):
        pf = probs_val.ravel().astype(np.float64)
        yf = y_val.ravel().astype(np.float64)
        logits = _logit(pf)

        def nll(T_val):
            T_val = max(T_val, EPSILON)
            p_cal = _sigmoid(logits / T_val)
            p_cal = np.clip(p_cal, EPSILON, 1.0 - EPSILON)
            return -float(np.mean(yf * np.log(p_cal) + (1.0 - yf) * np.log(1.0 - p_cal)))

        res = minimize_scalar(nll, bounds=(0.01, 20.0), method="bounded")
        self.T = float(res.x)
        self._fitted = True
        return self

    def transform(self, probs):
        if not self._fitted:
            return probs
        logits = _logit(probs.astype(np.float64))
        return _sigmoid(logits / self.T).astype(np.float32)


class PlattScaler:
    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self._fitted = False

    def fit(self, probs_val, y_val, lr=0.01, n_iter=1000):
        pf = probs_val.ravel().astype(np.float64)
        yf = y_val.ravel().astype(np.float64)
        logits = _logit(pf)
        a_t = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)
        b_t = torch.tensor([0.0], requires_grad=True, dtype=torch.float64)
        z_t = torch.tensor(logits, dtype=torch.float64)
        y_t = torch.tensor(yf, dtype=torch.float64)
        opt = torch.optim.LBFGS([a_t, b_t], lr=lr, max_iter=n_iter)

        def closure():
            opt.zero_grad()
            p_cal = torch.sigmoid(a_t * z_t + b_t)
            p_cal = torch.clamp(p_cal, EPSILON, 1.0 - EPSILON)
            loss = -torch.mean(y_t * torch.log(p_cal) + (1.0 - y_t) * torch.log(1.0 - p_cal))
            loss.backward()
            return loss

        opt.step(closure)
        self.a = float(a_t.detach())
        self.b = float(b_t.detach())
        self._fitted = True
        return self

    def transform(self, probs):
        if not self._fitted:
            return probs
        return _sigmoid(self.a * _logit(probs.astype(np.float64)) + self.b).astype(np.float32)


class IsotonicCalibrator:
    def __init__(self):
        self._iso = None
        self._fitted = False

    def fit(self, probs_val, y_val):
        pf = probs_val.ravel().astype(np.float64)
        yf = y_val.ravel().astype(np.float64)
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(pf, yf)
        self._fitted = True
        return self

    def transform(self, probs):
        if not self._fitted or self._iso is None:
            return probs
        return np.clip(self._iso.predict(probs.ravel().astype(np.float64)),
                       EPSILON, 1.0 - EPSILON).reshape(probs.shape).astype(np.float32)


def first_stable_detection(probs, onset, thr, m=TTD_M):
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


def compute_ttdef(ttd_det, fail_rate, t_max=T_MAX_EF):
    return ttd_det + fail_rate * t_max


def analyze_temporal_entropy(fp, y_fr, y_ep):
    h_stable, h_pre, h_post = [], [], []
    for ep_id in range(len(fp)):
        if y_ep[ep_id] != 1:
            continue
        onset_idx = np.where(y_fr[ep_id] > FRAME_THR)[0]
        if len(onset_idx) == 0:
            continue
        t0 = int(onset_idx[0])
        for t in range(fp.shape[1]):
            h = binary_entropy(float(fp[ep_id, t]))
            if t >= t0:
                h_post.append(h)
            elif t >= max(0, t0 - 10):
                h_pre.append(h)
            else:
                h_stable.append(h)
    return {
        "h_stable": float(np.mean(h_stable)) if h_stable else 0.0,
        "h_pre":    float(np.mean(h_pre))    if h_pre    else 0.0,
        "h_post":   float(np.mean(h_post))   if h_post   else 0.0,
    }


if __name__ == "__main__":
    np.random.seed(42)
    probs = np.random.rand(100, 96).astype("f")
    y_true = (np.random.rand(100, 96) > 0.5).astype("f")

    # Teste 1: Calibradores
    ts = TemperatureScaler().fit(probs, y_true)
    ps = PlattScaler().fit(probs, y_true)
    ic = IsotonicCalibrator().fit(probs, y_true)
    print(f"TS: T={ts.T:.4f} | PS: a={ps.a:.4f}, b={ps.b:.4f} | IC: fitted={ic._fitted}")

    p_ts = ts.transform(probs)
    p_ps = ps.transform(probs)
    p_ic = ic.transform(probs)

    yf = y_true.ravel()
    ece_b  = expected_calibration_error(yf, probs.ravel())
    ece_ts = expected_calibration_error(yf, p_ts.ravel())
    ece_ps = expected_calibration_error(yf, p_ps.ravel())
    ece_ic = expected_calibration_error(yf, p_ic.ravel())
    print(f"ECE antes={ece_b:.5f} | TS={ece_ts:.5f} | PS={ece_ps:.5f} | IC={ece_ic:.5f}")

    # Teste 2: Perfil temporal de entropia
    y_fr = np.zeros((100, 96), "f")
    y_ep = np.ones(100, "i")
    y_fr[:, 40:] = 0.8   # onset no frame 40

    prof_base = analyze_temporal_entropy(probs, y_fr, y_ep)
    prof_ts   = analyze_temporal_entropy(p_ts,  y_fr, y_ep)
    prof_ic   = analyze_temporal_entropy(p_ic,  y_fr, y_ep)

    print(f"\nPerfil H(t):")
    print(f"  Baseline: H_stable={prof_base['h_stable']:.4f}  "
          f"H_pre={prof_base['h_pre']:.4f}  H_post={prof_base['h_post']:.4f}")
    print(f"  TS:       H_stable={prof_ts['h_stable']:.4f}  "
          f"H_pre={prof_ts['h_pre']:.4f}  H_post={prof_ts['h_post']:.4f}")
    print(f"  IC:       H_stable={prof_ic['h_stable']:.4f}  "
          f"H_pre={prof_ic['h_pre']:.4f}  H_post={prof_ic['h_post']:.4f}")

    # Verifica a hipótese central:
    # ECE melhora mas separação temporal H(post) - H(stable) não aumenta
    strat_base = prof_base['h_post'] - prof_base['h_stable']
    strat_ts   = prof_ts['h_post']   - prof_ts['h_stable']
    strat_ic   = prof_ic['h_post']   - prof_ic['h_stable']
    print(f"\nEstratificacao temporal (H_post - H_stable):")
    print(f"  Baseline={strat_base:.4f}  TS={strat_ts:.4f}  IC={strat_ic:.4f}")
    print(f"  => TS mantém perfil similar ao Baseline: {abs(strat_ts - strat_base) < 0.05}")

    # Teste 3: TTDef
    ttds, n_fail = [], 0
    for ep in range(len(probs)):
        det = first_stable_detection(probs[ep], 40, thr=0.10, m=3)
        if det is not None:
            ttds.append((det - 40) * DT)
        else:
            n_fail += 1
    fr = n_fail / len(probs)
    ttd_det = float(np.mean(ttds)) if ttds else 0.0
    ttdef = compute_ttdef(ttd_det, fr)
    print(f"\nTTDef={ttdef:.4f}s (TTD_det={ttd_det:.4f}s, FR={fr:.4f})")

    # Teste 4: Verifica que transform produz probs em [0,1]
    for name, p_cal in [("TS", p_ts), ("PS", p_ps), ("IC", p_ic)]:
        assert p_cal.min() > 0.0 and p_cal.max() < 1.0, \
            f"{name}: probs fora de [0,1]: min={p_cal.min():.6f} max={p_cal.max():.6f}"
        assert p_cal.dtype == np.float32, f"{name}: dtype errado: {p_cal.dtype}"
    print("\nShape/range das probs calibradas: OK")

    print("\nTODOS OS TESTES CORE PASSARAM")
