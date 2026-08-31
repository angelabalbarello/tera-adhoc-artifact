"""Artigo 1 / E1b — Ablacao da cinematica no gating (estratificado por onset)."""
import os,sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src import lab; from config.config import GATING

def run():
    arms=["always","entropy","kinematic","hybrid"]; rows=[]; store={}
    for onset,s0 in [("abrupt",2000),("progressive",3000)]:
        eps=[lab.gen_gating_episode(s0+i,onset=onset,organized=True) for i in range(300)]; store[onset]=eps
        for p in arms:
            m=lab.aggregate_gating(eps,p,tau_h=GATING["tau_h"],tau_d=GATING["tau_delta"],theta=GATING["theta"])
            rows.append({"method":p,"onset_type":onset,"coverage↑":round(m["coverage"],3),
                         "false_suppression↓":round(m["false_suppression"],3),"TTDef↓":round(m["ttdef"],3),
                         "cost↓":round(m["cost"],3),"n_episodes":300})
    t3=pd.DataFrame(rows); lab.write_table(t3,"article1_A1_T3_onset_stratified","E1b — resultados estratificados por onset",to_results=True)
    ab=t3[t3.onset_type=="abrupt"].set_index("method").loc[arms]; pr=t3[t3.onset_type=="progressive"].set_index("method").loc[arms]
    fig,ax=lab.plt.subplots(figsize=(6.8,4)); x=np.arange(len(arms)); w=0.38
    ax.bar(x-w/2,ab["false_suppression↓"],w,label="abrupt",color=lab.C["unstruct"])
    ax.bar(x+w/2,pr["false_suppression↓"],w,label="progressive",color=lab.C["kinematic"])
    ax.set_xticks(x); ax.set_xticklabels(arms); ax.legend(fontsize=9)
    lab.style_ax(ax,"A1-F6 — False suppression ablation (by onset)","","false suppression rate"); lab.savefig(fig,"A1_F6_false_suppression_ablation")
    print("[Artigo1/E1b] tabela A1-T3 e figura A1-F6 geradas.")

if __name__=="__main__": run()
