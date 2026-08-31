"""Artigo 1 / E1 — Computacao seletiva confiavel na edge (sintetico-controlado)."""
import os,sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src import lab; from config.config import SEEDS, GATING

def run():
    pol=["always","entropy","kinematic","hybrid"]; Nb=GATING["n_episodes"]
    per={p:[] for p in pol}; neg=[]
    for s in SEEDS:
        eps=[lab.gen_gating_episode(s*1000+i,onset=("abrupt" if i%2==0 else "progressive"),organized=True) for i in range(Nb)]
        for p in pol: per[p].append(lab.aggregate_gating(eps,p,tau_h=GATING["tau_h"],tau_d=GATING["tau_delta"],theta=GATING["theta"]))
        epn=[lab.gen_gating_episode(90000+s*1000+i,onset=("abrupt" if i%2==0 else "progressive"),organized=False) for i in range(Nb)]
        neg.append(lab.aggregate_gating(epn,"entropy"))
    # A1-T1 baselines
    t1=pd.DataFrame([
        ["always","Always compute","No","No","No","No","No","upper-cost baseline"],
        ["unstructured","Unstructured + scheduler","No","Yes","No","No","Yes","negative control"],
        ["entropy","MHEG entropy-only","Yes","Yes","No","No","Yes","ablation"],
        ["kinematic","MHEG kinematic-only","Yes","No","Yes","No","Yes","ablation"],
        ["hybrid","MHEG entropy+kinematic","Yes","Yes","Yes","No","Yes","proposed"],
        ["hybrid_cal","MHEG hybrid + calibrated thresholds","Yes","Yes","Yes","Yes","Yes","proposed calibrated"]],
        columns=["method_id","method_name","temporal_org","entropy","kinematics","calibration","selective_suppression","role"])
    lab.write_table(t1,"article1_A1_T1_baselines","Braços experimentais do Artigo 1")
    # A1-T2 global results (mean ± std)
    def row(name,rs):
        c=lab.ms([r["cost"] for r in rs]); cov=lab.ms([r["coverage"] for r in rs]); mi=lab.ms([r["miss_rate"] for r in rs])
        td=lab.ms([r["ttdef"] for r in rs]); fsr=lab.ms([r["false_suppression"] for r in rs])
        return {"method":name,"cost↓":lab.fmt(c[0],c[1]),"coverage↑":lab.fmt(cov[0],cov[1]),
                "miss_rate↓":lab.fmt(mi[0],mi[1]),"TTDef↓":lab.fmt(td[0],td[1]),"false_suppression↓":lab.fmt(fsr[0],fsr[1])}
    t2=pd.DataFrame([row(p,per[p]) for p in pol]+[row("entropy_unstructured(neg)",neg)])
    lab.write_table(t2,"article1_A1_T2_global_results","Resultados globais E1 (média ± desvio; 4 seeds)",to_results=True)
    # A1-T4 stats (per-episode, 1 batch grande)
    big=[lab.gen_gating_episode(70000+i,onset=("abrupt" if i%2==0 else "progressive"),organized=True) for i in range(600)]
    def per_ep(p):
        o=[lab.simulate_gating(e,p) for e in big]
        return (np.array([(r["ttd"] if r["detected"] else r["Tmax"]) for r in o],float),
                np.array([r["false_supp"] for r in o]))
    tt={p:per_ep(p) for p in pol}
    def strow(comp,metric,a,b):
        p,e,sig,it=lab.wtest(a,b); return {"comparison":comp,"metric":metric,"test":"Wilcoxon","p_value":p,"effect_size(Cliff)":e,"significant":sig,"interpretation":it}
    t4=pd.DataFrame([strow("hybrid vs entropy","false_suppression",tt["hybrid"][1],tt["entropy"][1]),
                     strow("hybrid vs kinematic","TTDef",tt["hybrid"][0],tt["kinematic"][0]),
                     strow("hybrid vs always","TTDef",tt["hybrid"][0],tt["always"][0])])
    lab.write_table(t4,"article1_A1_T4_statistical_tests","Testes pareados E1")
    # figuras
    lab.flow_diagram(["Input multimodal","AF-TOI temporal organization","Causal compact model","entropy / Δx / kinematics","MHEG","compute or reuse state","risk decision"],
                     "A1_F1_architecture","A1-F1 — AF-TOI + MHEG architecture")
    # A1-F2 entropy curves
    aborg=lab.gen_gating_episode(1,onset="abrupt",organized=True); abuns=lab.gen_gating_episode(1,onset="abrupt",organized=False)
    fig,ax=lab.plt.subplots(figsize=(6.4,4)); rel=np.arange(aborg["L"])-aborg["t0"]
    ax.plot(rel,aborg["H"],label="AF-TOI (organized)",color=lab.C["entropy"],lw=1.8)
    ax.plot(rel,abuns["H"],label="unstructured",color=lab.C["unstruct"],lw=1.8,ls="--")
    ax.axvline(0,color="#999",lw=0.8); lab.style_ax(ax,"A1-F2 — Entropy vs. time relative to onset","t - t_onset (frames)","entropy"); ax.legend(fontsize=8)
    lab.savefig(fig,"A1_F2_entropy_curves")
    fig,ax=lab.plt.subplots(figsize=(6.4,4)); ax.boxplot([tt[p][0] for p in pol],labels=pol,showfliers=False)
    lab.style_ax(ax,"A1-F3 — TTDef distribution","","TTDef (frames)"); lab.savefig(fig,"A1_F3_ttdef_distribution")
    fig,ax=lab.plt.subplots(figsize=(6.4,4))
    for p,c in zip(pol,[lab.C["always"],lab.C["entropy"],lab.C["kinematic"],lab.C["hybrid"]]):
        x=np.sort(tt[p][0]); ax.plot(x,np.arange(1,len(x)+1)/len(x),label=p,color=c,lw=1.8)
    lab.style_ax(ax,"A1-F4 — TTDef CDF","TTDef (frames)","cumulative proportion"); ax.legend(fontsize=8); lab.savefig(fig,"A1_F4_ttdef_cdf")
    fig,ax=lab.plt.subplots(figsize=(6.2,4.2))
    for p in pol+["neg"]:
        rs = neg if p=="neg" else per[p]; cost=np.mean([r["cost"] for r in rs]); cov=np.mean([r["coverage"] for r in rs])
        key=("unstruct" if p=="neg" else p); ax.scatter(cost,cov,s=150,color=lab.C[key],edgecolor="white",zorder=3)
        ax.annotate(("neg.control" if p=="neg" else p),(cost,cov),fontsize=8,xytext=(6,-4),textcoords="offset points")
    lab.style_ax(ax,"A1-F5 — Cost × coverage frontier","computational cost","coverage"); ax.set_ylim(0,1.05); lab.savefig(fig,"A1_F5_cost_coverage_frontier")
    print("[Artigo1/E1] tabelas A1-T1/T2/T4 e figuras A1-F1..F5 geradas.")

if __name__=="__main__": run()
