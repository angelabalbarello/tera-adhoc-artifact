"""Entropia canonica em BITS (Eq. 11): runtime, replays e figuras."""
import math, re, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
def Hbits(p): p=min(max(p,1e-12),1-1e-12); return -(p*math.log2(p)+(1-p)*math.log2(1-p))
def test_hbits_values():
    assert abs(Hbits(0.5)-1.0) < 1e-9
    assert Hbits(1e-9) < 1e-6 and Hbits(1-1e-9) < 1e-6
def test_operational_scripts_use_log2():
    for rel in ["5-robustness-replay-runv29/Inferencia/run_v29_ablacao_ttdef_ajuste_gatting.py",
                "6-revision-round1/baselines_novos/run_gating_baselines_v2.py",
                "6-revision-round1/baselines_novos/run_refresh_gate_v2.py",
                "6-revision-round1/skiprnn_baseline/preonset_occupancy.py",
                "6-revision-round1/temporal_scale_generalization_v2/common.py",
                "1-pipeline-tera/tera_pipeline/inference/tera_infer.py"]:
        s=(ROOT/rel).read_text(errors="replace"); assert "log2" in s, rel
def test_replay_gate_fixed_to_bits():
    s=(ROOT/"5-robustness-replay-runv29/Inferencia/generate_episode_frame_logs.py").read_text(errors="replace")
    assert "np.log2(p_clip)" in s  # gate em bits (fix 23/09)
    # coluna H_t do CSV permanece em nats DELIBERADAMENTE (figuras convertem)
if __name__ == "__main__":
    test_hbits_values(); test_operational_scripts_use_log2(); test_replay_gate_fixed_to_bits(); print("entropy OK")
