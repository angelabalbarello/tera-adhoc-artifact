"""A1 — NC1..NC6 existem, rodam, produzem métricas e não geram NaN nas principais."""
import os,sys; sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import article1_neg as A
NCS=["NC1_random_budget_matched","NC2_entropy_unstructured_aggressive","NC3_entropy_lagged",
     "NC4_entropy_miscalibrated","NC5_periodic_skip","NC6_cost_matched_unstructured"]
def _pool(sc="abrupt_onset_dominant",n=120):
    return [A.make_episode(i,onset=("abrupt" if i%3 else "progressive"),scenario=sc) for i in range(n)]
def test_nc_all_exist_and_run():
    eps=_pool(); tgt=A.aggregate(eps,"MHEG_hybrid")["cost"]
    for m in NCS:
        p={"p_compute":tgt} if m=="NC1_random_budget_matched" else ({"lag":3} if m=="NC3_entropy_lagged" else ({"N":3} if m=="NC5_periodic_skip" else {}))
        r=A.aggregate(eps,m,p); assert r is not None and "cost" in r
    return "6 controles negativos existem e executam"
def test_nc_metrics_no_nan():
    eps=_pool(); tgt=A.aggregate(eps,"MHEG_hybrid")["cost"]
    for m in NCS:
        p={"p_compute":tgt} if m=="NC1_random_budget_matched" else ({"lag":3} if m=="NC3_entropy_lagged" else ({"N":3} if m=="NC5_periodic_skip" else {}))
        r=A.aggregate(eps,m,p)
        for k in ("cost","fs_rate","fs_near","onset_window_failure"):
            assert r[k]==r[k], f"{m}.{k} é NaN"
    return "métricas principais sem NaN (cost/fs_rate/fs_near/onset_fail)"
def test_nc_actually_save_cost():
    """Diferença central vs controle antigo: NC devem REALMENTE economizar custo."""
    eps=_pool("rare_abrupt_critical_events"); tgt=A.aggregate(eps,"MHEG_hybrid")["cost"]
    for m in ["NC1_random_budget_matched","NC5_periodic_skip","NC6_cost_matched_unstructured"]:
        p={"p_compute":tgt} if m=="NC1_random_budget_matched" else ({"N":3} if m=="NC5_periodic_skip" else {"tau_h":0.5})
        c=A.aggregate(eps,m,p)["cost"]; assert c<0.6, f"{m} não economiza (cost={c:.2f})"
    return "NC1/NC5/NC6 realmente reduzem custo (não computam ~tudo)"
TESTS=[test_nc_all_exist_and_run,test_nc_metrics_no_nan,test_nc_actually_save_cost]
