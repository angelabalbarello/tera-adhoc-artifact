"""Equivalencia entre semanticas de inferencia (seed 42, checkpoints congelados).

Compara, com os MESMOS pesos e limiares calibrados:
  (A) prefix-window recompute-on-update  — protocolo publicado (tera_infer);
  (B) passo incremental com estado persistente e skip real — Algoritmo 1 literal.

Saida: results/inference_semantics_seed42.csv + resumo no stdout.
Resultado de referencia (2026-10-11):
  FIXED : prefix FR=0 TTD=0.0156 | incremental FR=0 TTD=0.0095 (max|dp|=0.467 em frames iniciais, pad)
  MHEG  : prefix FR=0 TTDef=0.0234 Sup=44.7%  (== publicado: 0.0234 / 44.686%)
          incremental FR=0 TTDef=0.1701 Sup=44.4%  (NAO equivalente)
Implicacao textual: os resultados agendados publicados referem-se a semantica
recompute-on-update; a variante single-step do Algoritmo 1 requer calibracao e
avaliacao proprias (declarado na Secao 3.4 do artigo).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from nplstm_numpy_reader import load_pt, sigmoid
import numpy as np, pandas as pd

HERE=pathlib.Path(__file__).resolve()
ART=HERE.parents[3]
ROOT=ART/"1-pipeline-tera/results/exp_seed12_round1"
X=np.load(ROOT/"data/X_te_seed42.npy").astype(np.float32)
yfr=np.load(ROOT/"data/y_fr_te_seed42.npy").astype(np.float32)
yep=np.load(ROOT/"data/y_ep_te_seed42.npy").astype(int)
N,T,D=X.shape; H=64; KIN=[12,13,14]
cal=pd.read_csv(ROOT/"calibration/calibration_summary.csv"); row=cal[cal.seed==42].iloc[0]
tau_d,tau_h=float(row.tau_delta),float(row.tau_h)

class Np2:
    def __init__(self,sd):
        self.L=[(sd[f'lstm.weight_ih_l{l}'].T.copy(),sd[f'lstm.weight_hh_l{l}'].T.copy(),(sd[f'lstm.bias_ih_l{l}']+sd[f'lstm.bias_hh_l{l}']).copy()) for l in (0,1)]
        self.wf=sd['fc_fr.weight'].reshape(-1); self.bf=float(sd['fc_fr.bias'].reshape(-1)[0])
    def step(self,x,h,c):
        inp=x
        for l,(Wi,Wh,b) in enumerate(self.L):
            g=inp@Wi+h[l]@Wh+b
            i,f,gg,o=g[:,:H],g[:,H:2*H],g[:,2*H:3*H],g[:,3*H:]
            c[l]=sigmoid(f)*c[l]+sigmoid(i)*np.tanh(gg)
            h[l]=sigmoid(o)*np.tanh(c[l])
            inp=h[l]
        return sigmoid(h[1]@self.wf+self.bf),h,c
    def run_window(self,W):
        h=[np.zeros((W.shape[0],H),np.float32) for _ in range(2)]
        c=[np.zeros((W.shape[0],H),np.float32) for _ in range(2)]
        p=None
        for t in range(W.shape[1]):
            p,h,c=self.step(W[:,t,:],h,c)
        return p

def prefix_stream(m):
    P=np.zeros((N,T),np.float32)
    for t in range(T):
        pad=np.repeat(X[:,0:1,:],T-t-1,axis=1) if t+1<T else X[:,:0,:]
        W=np.concatenate([pad,X[:,:t+1,:]],axis=1) if t+1<T else X
        P[:,t]=m.run_window(W)
    return P

def metrics(P):
    def det(p,onset,thr=0.10,mm=3):
        cnt=0
        for t in range(onset,len(p)):
            cnt=cnt+1 if p[t]>=thr else 0
            if cnt>=mm: return t-mm+1
        return None
    miss=0; tts=[]
    for i in range(N):
        if not yep[i]: continue
        onset=int(np.where(yfr[i]>0.35)[0][0])
        dd=det(P[i],onset)
        if dd is None: miss+=1
        else: tts.append((dd-onset)*10/96)
    fr=miss/int(yep.sum()); ttd=float(np.mean(tts)) if tts else 0.0
    return fr,ttd,ttd+fr*10.0

def gates(last):
    pc=np.clip(last,1e-7,1-1e-7)
    return -(pc*np.log2(pc)+(1-pc)*np.log2(1-pc))

def hybrid_prefix(m,Pfix):
    P=np.zeros((N,T),np.float32); sup=np.zeros((N,T),bool); last=np.zeros(N,np.float32)
    for t in range(T):
        if t==0: act=np.ones(N,bool)
        else:
            dk=np.abs(X[:,t,KIN]-X[:,t-1,KIN]).max(axis=1)
            act=(dk>=tau_d)|(gates(last)>=tau_h)
        P[:,t]=np.where(act,Pfix[:,t],last); sup[:,t]=~act; last=P[:,t]
    return P,sup

def hybrid_incremental(m):
    P=np.zeros((N,T),np.float32); sup=np.zeros((N,T),bool)
    h=[np.zeros((N,H),np.float32) for _ in range(2)]; c=[np.zeros((N,H),np.float32) for _ in range(2)]
    last=np.zeros(N,np.float32)
    for t in range(T):
        if t==0: act=np.ones(N,bool)
        else:
            dk=np.abs(X[:,t,KIN]-X[:,t-1,KIN]).max(axis=1)
            act=(dk>=tau_d)|(gates(last)>=tau_h)
        hp=[a.copy() for a in h]; cp=[a.copy() for a in c]
        p,hn,cn=m.step(X[:,t,:],[a.copy() for a in h],[a.copy() for a in c])
        for l in range(2):
            h[l]=np.where(act[:,None],hn[l],hp[l]); c[l]=np.where(act[:,None],cn[l],cp[l])
        P[:,t]=np.where(act,p,last); sup[:,t]=~act; last=P[:,t]
    return P,sup

def main():
    m=Np2(load_pt(ROOT/"models/student_seed42.pt"))
    Pfix=prefix_stream(m)
    hinc=[np.zeros((N,H),np.float32) for _ in range(2)]; cinc=[np.zeros((N,H),np.float32) for _ in range(2)]
    Pinc=np.zeros((N,T),np.float32)
    for t in range(T):
        p,hinc,cinc=m.step(X[:,t,:],hinc,cinc); Pinc[:,t]=p
    rows=[]
    for name,P,sup in [("fixed_prefix",Pfix,None),("fixed_incremental",Pinc,None)]:
        fr,ttd,te=metrics(P); rows.append(dict(variant=name,FR=fr,TTD=ttd,TTDef=te,SupPct=0.0))
    Pa,sa=hybrid_prefix(m,Pfix); Pb,sb=hybrid_incremental(m)
    for name,P,sup in [("mheg_prefix_recompute",Pa,sa),("mheg_incremental_skip",Pb,sb)]:
        fr,ttd,te=metrics(P); rows.append(dict(variant=name,FR=fr,TTD=ttd,TTDef=te,SupPct=100*sup.mean()))
    out=pd.DataFrame(rows)
    out.to_csv(HERE.parents[1]/"results/inference_semantics_seed42.csv",index=False)
    print(out.to_string(index=False))
    pub=pd.read_csv(ART/"6-revision-round1/seed-expansion-tera/expanded_per_seed.csv")
    pub42=pub[(pub.Seed==42)&(pub.config_id=="aftkd_hybrid")]
    assert abs(out[out.variant=="mheg_prefix_recompute"].TTDef.iloc[0]-float(pub42.TTDef.iloc[0]))<5e-4
    print("OK: replay prefix reproduz o publicado (TTDef seed42)")

if __name__=="__main__": main()
