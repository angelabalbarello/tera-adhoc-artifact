"""Artigo 1 — Reforço experimental: controles negativos plausíveis, comparação
cost-matched, false suppression near onset e sensibilidade a entropy lag.

Dados sintético-controlados. Nenhum resultado é forçado: os mecanismos são fiéis
e o que emergir (inclusive ausência de degradação) é reportado como está.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src import lab
from src import article1_neg as A
from config.config import SEEDS

# ---------------- Especificação dos 9 cenários --------------------------------
SCEN = {
 # scenario_id: (onset_weights {abrupt,progressive}, expected_failure_mode)
 "balanced_onsets":            (dict(abrupt=0.5,  progressive=0.5),  "none / cost saving"),
 "abrupt_onset_dominant":      (dict(abrupt=0.85, progressive=0.15), "false suppression at abrupt onset"),
 "progressive_onset_dominant": (dict(abrupt=0.15, progressive=0.85), "delayed detection (TTDef)"),
 "rare_abrupt_critical_events":(dict(abrupt=1.0,  progressive=0.0),  "missed transient critical event (coverage collapse)"),
 "high_noise_entropy":         (dict(abrupt=0.5,  progressive=0.5),  "entropy scheduler misfires (noise)"),
 "kinematic_signal_missing":   (dict(abrupt=0.7,  progressive=0.3),  "loss of kinematic protection at abrupt onset"),
 "entropy_lag":                (dict(abrupt=0.7,  progressive=0.3),  "lagged entropy misses abrupt onset"),
 "false_onset":                (dict(abrupt=0.5,  progressive=0.5),  "wasted compute / false trigger"),
 "no_onset_negative_control":  (dict(abrupt=0.5,  progressive=0.5),  "reference: false_suppression must be ~0"),
}
NB = 90   # episódios por (onset, seed)

def pool(scenario):
    ab, pr = SCEN[scenario][0]["abrupt"], SCEN[scenario][0]["progressive"]
    eps = []
    for s in SEEDS:
        r = lab.rng(s * 17 + 3)
        for i in range(NB):
            onset = "abrupt" if r.random() < ab / (ab + pr) else "progressive"
            eps.append(A.make_episode(s * 100000 + i, onset=onset, scenario=scenario))
    return eps

METHODS = ["always","MHEG_hybrid","MHEG_calibrated","NC1_random_budget_matched",
           "NC2_entropy_unstructured_aggressive","NC3_entropy_lagged",
           "NC4_entropy_miscalibrated","NC5_periodic_skip","NC6_cost_matched_unstructured"]

def method_params(method, target_cost, eps=None):
    if method == "NC1_random_budget_matched": return {"p_compute": target_cost}
    if method == "NC5_periodic_skip":         return {"N": 3}
    if method == "NC3_entropy_lagged":        return {"lag": 3}
    if method == "NC4_entropy_miscalibrated": return {"tau_h": 0.55}   # limiar de outra distribuição
    if method == "NC2_entropy_unstructured_aggressive":
        tau, _ = A.solve_threshold_for_cost(eps, "NC2_entropy_unstructured_aggressive", target_cost)
        return {"tau_h": tau}
    if method == "NC6_cost_matched_unstructured":
        tau, _ = A.solve_threshold_for_cost(eps, "NC6_cost_matched_unstructured", target_cost)
        return {"tau_h": tau}
    return {}

def interp(m):
    if m["method"] == "always": return "upper-cost reference"
    cr = 100 * (1 - m["cost"])
    if cr < 5: return "no cost saving (computes ~everything)"
    if m["method"].startswith("MHEG"):
        return "saves cost AND preserves onset (safe)" if m["fs_near"] < 0.1 else "saves cost but some near-onset suppression"
    if m["fs_near"] >= 0.3 or (m["coverage"] == m["coverage"] and m["coverage"] < 0.8):
        return "UNSAFE suppression to save cost (near-onset misses)"
    return "saves cost with limited near-onset harm"

def run():
    # ===== A1-T5 negative controls grid (Method × Scenario) =====
    t5 = []; per_scen_hybrid_cost = {}; pools = {}
    for sc in SCEN:
        eps = pool(sc); pools[sc] = eps
        hcost = A.aggregate(eps, "MHEG_hybrid")["cost"]; per_scen_hybrid_cost[sc] = hcost
        for meth in METHODS:
            pr = method_params(meth, hcost, eps)
            m = A.aggregate(eps, meth, pr)
            t5.append({"Method": meth, "Scenario": sc,
                       "Cost reduction↑": round(100 * (1 - m["cost"]), 1),
                       "Compute rate↓": round(100 * m["cost"], 1),
                       "Coverage↑": round(100 * m["coverage"], 1) if m["coverage"] == m["coverage"] else "n/a",
                       "Miss rate↓": round(100 * m["miss_rate"], 1) if m["miss_rate"] == m["miss_rate"] else "n/a",
                       "TTDef↓": round(m["ttdef"], 2) if m["ttdef"] == m["ttdef"] else "n/a",
                       "False suppression↓": round(100 * m["fs_rate"], 1),
                       "FS near onset↓": round(100 * m["fs_near"], 1) if m["fs_near"] == m["fs_near"] else "n/a",
                       "Interpretation": interp(m)})
    lab.write_table(pd.DataFrame(t5), "article1_A1_T5_negative_controls",
                    "Controles negativos (Método × Cenário) — sintético-controlado", to_results=True)

    # ===== A1-T6 cost-matched comparison =====
    t6 = []
    cm_methods = ["MHEG_hybrid","NC1_random_budget_matched","NC2_entropy_unstructured_aggressive","NC6_cost_matched_unstructured"]
    for sc in ["balanced_onsets","abrupt_onset_dominant","rare_abrupt_critical_events"]:
        eps = pools[sc]; tgt = per_scen_hybrid_cost[sc]
        for meth in cm_methods:
            pr = method_params(meth, tgt, eps); m = A.aggregate(eps, meth, pr)
            err = 100 * abs(m["cost"] - tgt) / max(1e-9, tgt)
            t6.append({"Method": meth, "Scenario": sc, "Target cost": round(tgt, 3),
                       "Observed cost": round(m["cost"], 3), "Cost match error %": round(err, 1),
                       "Coverage↑": round(100 * m["coverage"], 1) if m["coverage"] == m["coverage"] else "n/a",
                       "Miss rate↓": round(100 * m["miss_rate"], 1) if m["miss_rate"] == m["miss_rate"] else "n/a",
                       "TTDef↓": round(m["ttdef"], 2) if m["ttdef"] == m["ttdef"] else "n/a",
                       "FS near onset↓": round(100 * m["fs_near"], 1) if m["fs_near"] == m["fs_near"] else "n/a",
                       "Is cost-matched?": "yes" if err <= 2.0 else ("~ (%.1f%%)" % err)})
    lab.write_table(pd.DataFrame(t6), "article1_A1_T6_cost_matched_comparison",
                    "Comparação sob orçamento computacional equivalente (±2%)", to_results=True)

    # ===== A1-T7 onset-window failures (por onset_type) =====
    t7 = []
    key_methods = ["MHEG_hybrid","MHEG_kinematic_only" if False else "MHEG_calibrated",
                   "NC1_random_budget_matched","NC3_entropy_lagged","NC6_cost_matched_unstructured"]
    for sc in ["abrupt_onset_dominant","progressive_onset_dominant","rare_abrupt_critical_events"]:
        eps = pools[sc]; tgt = per_scen_hybrid_cost[sc]
        for onset in ["abrupt","progressive"]:
            sub = [e for e in eps if e["onset"] == onset and e["has_onset"]]
            if len(sub) < 5: continue
            for meth in key_methods:
                pr = method_params(meth, tgt, eps); m = A.aggregate(sub, meth, pr)
                t7.append({"Method": meth, "Scenario": sc, "Onset type": onset,
                           "n_onsets": m["n_onsets"], "n_onset_window_suppressions": m["n_win_supp"],
                           "onset_window_failure_rate": round(m["onset_window_failure"], 3),
                           "coverage": round(m["coverage"], 3) if m["coverage"] == m["coverage"] else "n/a",
                           "TTDef": round(m["ttdef"], 2) if m["ttdef"] == m["ttdef"] else "n/a"})
    lab.write_table(pd.DataFrame(t7), "article1_A1_T7_onset_window_failures",
                    "Falhas na janela de onset (abrupt vs progressive)", to_results=True)

    # ===== A1-T8 entropy-lag sensitivity =====
    t8 = []
    for lag in [1, 3, 5, 10]:
        for sc in ["abrupt_onset_dominant","progressive_onset_dominant","rare_abrupt_critical_events"]:
            eps = pools[sc]; m = A.aggregate(eps, "NC3_entropy_lagged", {"lag": lag})
            t8.append({"Lag frames": lag, "Scenario": sc,
                       "Coverage↑": round(100 * m["coverage"], 1) if m["coverage"] == m["coverage"] else "n/a",
                       "Miss rate↓": round(100 * m["miss_rate"], 1) if m["miss_rate"] == m["miss_rate"] else "n/a",
                       "TTDef↓": round(m["ttdef"], 2) if m["ttdef"] == m["ttdef"] else "n/a",
                       "FS near onset↓": round(100 * m["fs_near"], 1) if m["fs_near"] == m["fs_near"] else "n/a",
                       "Cost reduction↑": round(100 * (1 - m["cost"]), 1)})
    lab.write_table(pd.DataFrame(t8), "article1_A1_T8_entropy_lag_sensitivity",
                    "Sensibilidade a entropy lag (por onset)", to_results=True)

    # ===== scenarios metadata =====
    meta = []
    for sc, (ow, efm) in SCEN.items():
        eps = pools[sc]; kav = np.mean([e.get("kinematic_available", True) for e in eps])
        meta.append({"scenario_id": sc, "seed": ",".join(map(str, SEEDS)),
                     "onset_type": ("abrupt" if ow["abrupt"] > 0.6 else "progressive" if ow["progressive"] > 0.6 else "mixed"),
                     "noise_level": "high" if sc == "high_noise_entropy" else "base",
                     "critical_event_rate": "transient" if sc == "rare_abrupt_critical_events" else "persistent",
                     "kinematic_available": round(float(kav), 2),
                     "entropy_quality": "degraded" if sc in ("high_noise_entropy","entropy_lag") else "organized",
                     "expected_failure_mode": efm, "n_episodes": len(eps)})
    lab.write_table(pd.DataFrame(meta), "article1_scenarios_metadata", "Cenários A1 (metadados)", to_results=True)

    _figures(pools, per_scen_hybrid_cost, t8)
    print("[Artigo1/NC] A1-T5..T8 + metadata + figuras A1-F7..F10 geradas.")
    return t5, t6, t7, t8

def _figures(pools, hcost, t8):
    import pandas as pd
    plt = lab.plt
    # ---- A1-F7 negative controls cost × coverage (rare_abrupt_critical) ----
    sc = "rare_abrupt_critical_events"; eps = pools[sc]; tgt = hcost[sc]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for meth in METHODS + ["MHEG_kinematic_only","MHEG_entropy_only"]:
        pr = method_params(meth, tgt, eps); m = A.aggregate(eps, meth, pr)
        cr = 100 * (1 - m["cost"]); cov = 100 * m["coverage"] if m["coverage"] == m["coverage"] else np.nan
        col = ("#0A4E6E" if meth.startswith("MHEG") else "#B23A3A" if meth.startswith("NC") else "#64748B")
        mk = "o" if meth.startswith("MHEG") else "^" if meth.startswith("NC") else "s"
        ax.scatter(cr, cov, s=140, color=col, marker=mk, edgecolor="white", zorder=3)
        ax.annotate(meth.replace("MHEG_", "").replace("_budget_matched", "").replace("_unstructured", ""),
                    (cr, cov), fontsize=6.5, xytext=(5, -3), textcoords="offset points")
    lab.style_ax(ax, "A1-F7 — Negative controls: cost reduction × coverage (rare critical)",
                 "cost reduction (%)", "coverage (%)"); ax.set_ylim(-5, 105)
    ax.scatter([], [], marker="o", color="#0A4E6E", label="MHEG (proposed)")
    ax.scatter([], [], marker="^", color="#B23A3A", label="negative controls")
    ax.scatter([], [], marker="s", color="#64748B", label="always compute")
    ax.legend(fontsize=7, loc="lower left"); lab.savefig(fig, "A1_F7_negative_controls_cost_coverage")

    # ---- A1-F8 false suppression near onset (grouped bars) ----
    groups = {"abrupt": "abrupt_onset_dominant", "progressive": "progressive_onset_dominant",
              "rare_critical": "rare_abrupt_critical_events"}
    bm = ["MHEG_hybrid","NC1_random_budget_matched","NC3_entropy_lagged","NC5_periodic_skip","NC6_cost_matched_unstructured"]
    data = {g: [] for g in groups}
    for g, sc in groups.items():
        eps = pools[sc]; tgt = hcost[sc]
        for meth in bm:
            m = A.aggregate(eps, meth, method_params(meth, tgt, eps))
            data[g].append(100 * m["fs_near"] if m["fs_near"] == m["fs_near"] else 0)
    x = np.arange(len(bm)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for i, (g, c) in enumerate(zip(groups, ["#B23A3A","#E8A33D","#1C7293"])):
        ax.bar(x + (i - 1) * w, data[g], w, label=g, color=c)
    ax.set_xticks(x); ax.set_xticklabels([b.replace("MHEG_","").replace("_budget_matched","").replace("_unstructured","") for b in bm], rotation=12, fontsize=8)
    ax.legend(fontsize=8); lab.style_ax(ax, "A1-F8 — False suppression near onset (by onset type)", "", "FS near onset (%)")
    lab.savefig(fig, "A1_F8_onset_window_false_suppression")

    # ---- A1-F9 entropy-lag sensitivity ----
    df8 = pd.DataFrame(t8); fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for sc, c in zip(["abrupt_onset_dominant","progressive_onset_dominant","rare_abrupt_critical_events"],
                     ["#B23A3A","#2F855A","#1C7293"]):
        d = df8[df8.Scenario == sc]
        ax.plot(d["Lag frames"], d["Coverage↑"], marker="o", label=sc.replace("_onset_dominant","").replace("_events",""), color=c, lw=1.8)
    lab.style_ax(ax, "A1-F9 — Entropy lag sensitivity", "lag (frames)", "coverage (%)"); ax.legend(fontsize=8)
    lab.savefig(fig, "A1_F9_entropy_lag_sensitivity")

    # ---- A1-F10 cost-matched frontier (observed cost × coverage) ----
    sc = "rare_abrupt_critical_events"; eps = pools[sc]; tgt = hcost[sc]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    marks = {"MHEG_hybrid": ("#0A4E6E","*","proposed"),
             "NC1_random_budget_matched": ("#B23A3A","^","cost-matched NC"),
             "NC6_cost_matched_unstructured": ("#8B3A62","^","cost-matched NC"),
             "NC2_entropy_unstructured_aggressive": ("#E8A33D","v","aggressive NC"),
             "NC5_periodic_skip": ("#64748B","s","naive NC"), "always": ("#334155","P","reference")}
    ax.axvline(tgt, color="#94a3b8", ls="--", lw=1, label="hybrid budget")
    for meth, (c, mk, lbl) in marks.items():
        m = A.aggregate(eps, meth, method_params(meth, tgt, eps))
        cov = 100 * m["coverage"] if m["coverage"] == m["coverage"] else np.nan
        ax.scatter(m["cost"], cov, s=170, color=c, marker=mk, edgecolor="white", zorder=3)
        ax.annotate(meth.replace("MHEG_","").replace("_budget_matched","").replace("_unstructured","").replace("entropy_",""),
                    (m["cost"], cov), fontsize=6.5, xytext=(5, -4), textcoords="offset points")
    lab.style_ax(ax, "A1-F10 — Cost-matched frontier (observed cost × coverage)", "observed computational cost", "coverage (%)")
    ax.set_ylim(-5, 105); ax.legend(fontsize=7, loc="center right"); lab.savefig(fig, "A1_F10_cost_matched_frontier")

if __name__ == "__main__":
    run()
