"""Coeficientes do manifest correspondem ao codigo run_v29 (config A)."""
import json, re, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = (ROOT/"5-robustness-replay-runv29/Inferencia/run_v29_ablacao_ttdef_ajuste_gatting.py").read_text(errors="replace")
MAN = json.load(open(pathlib.Path(__file__).resolve().parents[1]/"results/aftoi_loss_manifest.json"))
def test_coefficients_in_code():
    checks = ["gamma_ep    = 0.25","0.60 * early_track + 1.60 * early_floor + 1.20 * early_mono + 1.50 * early_push",
              "+ 1.20 * tail_pen","+ 2.00 * noncrit_tail_pen","* 3.0",
              "lambda_late: float = 2.0","pre_pen / pre_count * 1.5",
              "warmup_epochs: int = 12, rampup_epochs: int = 16",
              "delta_end   = 2.80","max(alpha_curr, 0.50)","K           = 50","N_PRE_ONSET = 8",
              "theta_late: float = 0.10","if delta > 0.5","if delta > 1.0",
              "torch.linspace(0.45,"]
    for c in checks:
        assert c in SRC, c
    assert "epochs=80, beta_max=0.35, temp=2.0" in SRC
def test_manifest_values():
    assert MAN["coefficients"]["gamma_ep"]==0.25 and MAN["coefficients"]["lambda_late"]==2.0
    assert MAN["alignment"]["T"]==2.0 and MAN["schedules"]["alpha_floor"]==0.50
if __name__ == "__main__":
    test_coefficients_in_code(); test_manifest_values(); print("aftoi loss manifest OK")
