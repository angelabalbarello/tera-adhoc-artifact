"""Motor comum da suite experimental Safe-Drive (dados sintetico-controlados).

Fornece: geradores determinsticos, metricas, testes estatisticos, PCA(numpy),
e escritores de tabela (.md/.csv/.tex) e figura (.pdf/.png 300dpi).
"""
from __future__ import annotations
import os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("results","figures","tables","reports","logs","data"):
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)
RES, FIG, TAB, REP, LOG = (os.path.join(ROOT,x) for x in ("results","figures","tables","reports","logs"))

sys.path.insert(0, os.path.dirname(ROOT))   # p/ importar 'app' (modulos reais da tese)

C = {"hybrid":"#0A4E6E","entropy":"#1C7293","kinematic":"#E8A33D","always":"#64748B",
     "unstruct":"#B23A3A","calib":"#2F855A","acc":"#6b46c1"}
try:
    from scipy.stats import wilcoxon; _SCIPY=True
except Exception:
    _SCIPY=False

def rng(seed): return np.random.default_rng(seed)

def style_ax(ax, title="", xl="", yl=""):
    ax.set_title(title, fontsize=12, fontweight="bold", color="#0f172a")
    ax.set_xlabel(xl, fontsize=10, color="#334155"); ax.set_ylabel(yl, fontsize=10, color="#334155")
    ax.grid(True, alpha=0.25, linewidth=0.6); ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(colors="#475569", labelsize=9)

def savefig(fig, name):
    """Salva figura de submissao: PDF vetorial + PNG 300 dpi."""
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return name

def _tex(df, caption, label):
    cols = list(df.columns)
    out = ["\\begin{table}[t]","\\centering","\\small",
           "\\caption{"+caption+"}","\\label{tab:"+label+"}",
           "\\begin{tabular}{"+ "l"*len(cols) +"}","\\hline",
           " & ".join(str(c).replace("_","\\_").replace("%","\\%") for c in cols)+" \\\\","\\hline"]
    for _,r in df.iterrows():
        out.append(" & ".join(str(v).replace("_","\\_").replace("%","\\%") for v in r.values)+" \\\\")
    out += ["\\hline","\\end{tabular}","\\end{table}"]
    return "\n".join(out)

def write_table(df, name, caption="", to_results=False):
    """Escreve tabela em .md, .csv e .tex (pasta tables/). Opcional: copia CSV p/ results/."""
    import pandas as pd
    df.to_csv(os.path.join(TAB, f"{name}.csv"), index=False)
    with open(os.path.join(TAB, f"{name}.md"), "w", encoding="utf-8") as f: f.write(df.to_markdown(index=False))
    with open(os.path.join(TAB, f"{name}.tex"), "w", encoding="utf-8") as f: f.write(_tex(df, caption or name, name))
    if to_results: df.to_csv(os.path.join(RES, f"{name}.csv"), index=False)
    return name

def save_result_csv(df, name):
    df.to_csv(os.path.join(RES, f"{name}.csv"), index=False); return name

def pca2(X):
    Xc = X - X.mean(0); U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T

def wtest(a, b):
    """Wilcoxon pareado + Cliff's delta. Retorna (p, effect, sig, interp)."""
    a=np.asarray(a,float); b=np.asarray(b,float)
    if not _SCIPY: return ("n/a","n/a","n/a","scipy ausente")
    try:
        d=a-b
        if np.allclose(d,0): p=1.0
        else: _,p=wilcoxon(a,b,zero_method="wilcox")
        gt=np.sum(a[:,None]>b[None,:]); lt=np.sum(a[:,None]<b[None,:]); cd=(gt-lt)/(len(a)*len(b))
        ac=abs(cd); interp=("negligivel" if ac<0.147 else "pequeno" if ac<0.33 else "medio" if ac<0.474 else "grande")
        return (round(float(p),4), round(float(cd),3), "sim" if p<0.05 else "nao", interp)
    except Exception as e:
        return ("n/a","n/a","n/a",str(e)[:20])

def ms(vals): 
    vals=np.asarray(vals,float); m=vals.mean(); s=vals.std(); 
    ci=1.96*s/math.sqrt(len(vals)) if len(vals)>1 else 0.0
    return round(m,3), round(s,3), round(ci,3)

def fmt(m,s): return f"{m:.3f} ± {s:.3f}"

# ---------------- Gerador de gating (E1/E1b) — mecanismo FIEL ----------------
def gen_gating_episode(seed, onset="abrupt", organized=True, L=60):
    r=rng(seed); t0=int(r.integers(21,39)); risk=np.zeros(L)
    if onset=="abrupt":
        risk[t0:]=1.0; prob=np.clip(risk*0.9+r.normal(0,0.04,L),0,1)
    else:
        risk=np.clip((np.arange(L)-t0)/12.0,0,1); prob=np.clip(risk*0.9+r.normal(0,0.04,L),0,1)
    H=np.full(L,0.12)+r.normal(0,0.03,L)
    if organized:
        w0,w1=(t0+1,min(L,t0+11)) if onset=="abrupt" else (t0,min(L,t0+15))
        H[w0:w1]=0.9+r.normal(0,0.03,w1-w0)
    else:
        H=np.full(L,0.5)+r.normal(0,0.05,L)
    H=np.clip(H,0,1); dx=np.abs(r.normal(0,0.04,L))
    if onset=="abrupt": dx[t0:t0+2]+=r.uniform(0.55,0.8)
    else: dx+=np.clip((np.arange(L)-t0)/60.0,0,0.12)
    dx=np.clip(dx,0,1)
    return {"t0":t0,"risk_true":risk,"prob":prob,"H":H,"dx":dx,"L":L,"onset":onset,"organized":organized}

def simulate_gating(ep, policy, tau_h=0.35, tau_d=0.30, theta=0.5):
    L=ep["L"]; t0=ep["t0"]; computed=np.zeros(L,bool); avail=ep["prob"].copy(); last=ep["prob"][0]; fs=0
    for t in range(L):
        trig = (True if policy=="always" else ep["H"][t]>=tau_h if policy=="entropy"
                else ep["dx"][t]>=tau_d if policy=="kinematic"
                else (ep["H"][t]>=tau_h or ep["dx"][t]>=tau_d))
        if trig: computed[t]=True; last=ep["prob"][t]
        else:
            avail[t]=last
            if ep["risk_true"][t]>=0.5 and last<theta: fs+=1
    det=None
    for t in range(t0,L):
        if avail[t]>=theta: det=t; break
    detected=det is not None; ttd=(det-t0) if detected else None
    return {"cost":computed.mean(),"detected":detected,"ttd":ttd,
            "false_supp":fs/max(1,(ep["risk_true"]>=0.5).sum()),"Tmax":L}

def aggregate_gating(eps, policy, **kw):
    rows=[simulate_gating(e,policy,**kw) for e in eps]; n=len(rows)
    miss=sum(1 for r in rows if not r["detected"])/n
    ttds=[r["ttd"] for r in rows if r["detected"]]; Tmax=rows[0]["Tmax"]
    return {"policy":policy,"cost":float(np.mean([r["cost"] for r in rows])),"coverage":1-miss,"miss_rate":miss,
            "ttd_det":float(np.mean(ttds)) if ttds else float("nan"),
            "ttdef":float(np.mean([(r["ttd"] if r["detected"] else Tmax) for r in rows])),
            "false_suppression":float(np.mean([r["false_supp"] for r in rows]))}

def flow_diagram(steps, name, title, color="#0A4E6E", horizontal=True):
    """Diagrama simples de fluxo (caixas + setas) p/ arquiteturas/lifecycles."""
    n=len(steps); fig,ax=plt.subplots(figsize=(min(2.2*n,13),2.6) if horizontal else (4.4,1.4*n))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    if horizontal:
        w=1.0/n
        for i,s in enumerate(steps):
            x=i*w+0.01
            ax.add_patch(plt.Rectangle((x,0.32),w-0.02,0.36,fc="#eef4fb",ec=color,lw=1.6))
            ax.text(x+(w-0.02)/2,0.5,s,ha="center",va="center",fontsize=8.5,wrap=True,color="#12344d")
            if i<n-1: ax.annotate("",xy=(x+w-0.01,0.5),xytext=(x+w-0.02,0.5),arrowprops=dict(arrowstyle="->",color=color,lw=1.6))
    else:
        h=1.0/n
        for i,s in enumerate(steps):
            y=1-(i+1)*h+0.01
            ax.add_patch(plt.Rectangle((0.1,y),0.8,h-0.02,fc="#eef4fb",ec=color,lw=1.6))
            ax.text(0.5,y+(h-0.02)/2,s,ha="center",va="center",fontsize=8.5,color="#12344d")
            if i<n-1: ax.annotate("",xy=(0.5,y),xytext=(0.5,y-0.005),arrowprops=dict(arrowstyle="->",color=color,lw=1.6))
    ax.set_title(title,fontsize=11,fontweight="bold",color="#0f172a")
    savefig(fig,name)
