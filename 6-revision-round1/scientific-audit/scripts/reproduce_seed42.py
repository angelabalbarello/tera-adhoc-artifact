#!/usr/bin/env python3
"""Reproducao independente (seed 42) — ADHOC-D-26-02099, revisao round 1.

Reproduz, a partir dos checkpoints congelados e da particao de teste:
  1. os quatro resultados de semantica de inferencia (fixed/MHEG x janela/incremental);
  2. a reconciliacao da Tabela 13 (p_max por episodio critico, fracao >= theta, FR);
  3. a verificacao semantica FR >= frac(p_max < theta) no mesmo stream;
e compara com os valores publicados (expanded_per_seed.csv).

Uso (todos os arquivos no diretorio atual, como neste pacote):
    python3 reproduce_seed42.py
ou com caminhos explicitos:
    python3 reproduce_seed42.py --data DIR --models DIR --published expanded_per_seed.csv

Dependencias: numpy, pandas (python >= 3.9). Sem torch: o checkpoint .pt e lido
por um unpickler minimo e o LSTM de 2 camadas e executado em numpy.
"""
import argparse, zipfile, pickle, io, pathlib, hashlib
import numpy as np, pandas as pd

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for ch in iter(lambda:f.read(1<<20),b''): h.update(ch)
    return h.hexdigest()

DTYPES={"FloatStorage":np.float32,"DoubleStorage":np.float64,"LongStorage":np.int64,
        "IntStorage":np.int32,"HalfStorage":np.float16,"BoolStorage":np.bool_}
def load_pt(path):
    zf=zipfile.ZipFile(path); prefix=zf.namelist()[0].split("/")[0]; storages={}
    class Stor:
        def __init__(self,key,dtype): self.key=key; self.dtype=dtype
    class Unp(pickle.Unpickler):
        def find_class(self,mod,name):
            if name=="_rebuild_tensor_v2":
                def rt(storage,offset,size,stride,*a):
                    data=storages[storage.key]
                    if not size: return data[offset:offset+1].copy()
                    arr=data[offset:offset+int(np.prod(size))]
                    return np.lib.stride_tricks.as_strided(
                        arr, shape=size, strides=[st*arr.itemsize for st in stride]).copy()
                return rt
            if name in DTYPES: return DTYPES[name]
            if name=="OrderedDict":
                from collections import OrderedDict; return OrderedDict
            return lambda *a,**k: None
        def persistent_load(self,pid):
            dtype_cls,key=pid[1],pid[2]
            dt=dtype_cls if isinstance(dtype_cls,type) else np.float32
            if key not in storages:
                storages[key]=np.frombuffer(zf.read(f"{prefix}/data/{key}"),dtype=dt)
            return Stor(key,dt)
    sd=Unp(io.BytesIO(zf.read(f"{prefix}/data.pkl"))).load()
    return {k:np.asarray(v,dtype=np.float32) for k,v in sd.items()}

def sigmoid(x): return 1/(1+np.exp(-x))
H=64
class Np2:
    def __init__(self,sd):
        self.L=[(sd[f'lstm.weight_ih_l{l}'].T.copy(),sd[f'lstm.weight_hh_l{l}'].T.copy(),
                 (sd[f'lstm.bias_ih_l{l}']+sd[f'lstm.bias_hh_l{l}']).copy()) for l in (0,1)]
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
    def run_window(self,Wn):
        n=Wn.shape[0]
        h=[np.zeros((n,H),np.float32) for _ in range(2)]
        c=[np.zeros((n,H),np.float32) for _ in range(2)]
        p=None
        for t in range(Wn.shape[1]): p,h,c=self.step(Wn[:,t,:],h,c)
        return p

def main():
    ap=argparse.ArgumentParser()
    here=pathlib.Path(__file__).parent
    ap.add_argument("--data",default=str(here)); ap.add_argument("--models",default=str(here))
    ap.add_argument("--published",default=str(here/"expanded_per_seed.csv"))
    ap.add_argument("--calib",default=str(here/"calibration_summary.csv"))
    ap.add_argument("--out",default=str(here))
    a=ap.parse_args(); D=pathlib.Path(a.data); M=pathlib.Path(a.models); O=pathlib.Path(a.out)

    print("== hashes dos insumos ==")
    for f in ["X_te_seed42.npy","y_fr_te_seed42.npy","y_ep_te_seed42.npy"]:
        print(f"  {f}  sha256={sha256(D/f)}")
    for f in ["student_seed42.pt","baseline_seed42.pt"]:
        print(f"  {f}  sha256={sha256(M/f)}")

    X=np.load(D/"X_te_seed42.npy").astype(np.float32)
    yfr=np.load(D/"y_fr_te_seed42.npy").astype(np.float32)
    yep=np.load(D/"y_ep_te_seed42.npy").astype(int)
    N,T,_=X.shape; KIN=[12,13,14]
    cal=pd.read_csv(a.calib); row=cal[cal.seed==42].iloc[0]
    tau_d,tau_h=float(row.tau_delta),float(row.tau_h)

    def prefix_stream(m):
        P=np.zeros((N,T),np.float32)
        for t in range(T):
            pad=np.repeat(X[:,0:1,:],T-t-1,axis=1) if t+1<T else X[:,:0,:]
            Wn=np.concatenate([pad,X[:,:t+1,:]],axis=1) if t+1<T else X
            P[:,t]=m.run_window(Wn)
        return P
    def det(p,onset,thr=0.10,mm=3):
        cnt=0
        for t in range(onset,len(p)):
            cnt=cnt+1 if p[t]>=thr else 0
            if cnt>=mm: return t-mm+1
        return None
    def metrics(P):
        miss=0; tts=[]
        for i in range(N):
            if not yep[i]: continue
            onset=int(np.where(yfr[i]>0.35)[0][0]); dd=det(P[i],onset)
            if dd is None: miss+=1
            else: tts.append((dd-onset)*10/96)
        fr=miss/int(yep.sum()); ttd=float(np.mean(tts)) if tts else 0.0
        return fr,ttd,ttd+fr*10.0
    def gateH(last):
        pc=np.clip(last,1e-7,1-1e-7)
        return -(pc*np.log2(pc)+(1-pc)*np.log2(1-pc))
    def hybrid_prefix(Pfix):
        P=np.zeros((N,T),np.float32); sup=np.zeros((N,T),bool); last=np.zeros(N,np.float32)
        for t in range(T):
            act=np.ones(N,bool) if t==0 else \
                (np.abs(X[:,t,KIN]-X[:,t-1,KIN]).max(axis=1)>=tau_d)|(gateH(last)>=tau_h)
            P[:,t]=np.where(act,Pfix[:,t],last); sup[:,t]=~act; last=P[:,t]
        return P,sup
    def hybrid_inc(m):
        P=np.zeros((N,T),np.float32); sup=np.zeros((N,T),bool)
        h=[np.zeros((N,H),np.float32) for _ in range(2)]; c=[np.zeros((N,H),np.float32) for _ in range(2)]
        last=np.zeros(N,np.float32)
        for t in range(T):
            act=np.ones(N,bool) if t==0 else \
                (np.abs(X[:,t,KIN]-X[:,t-1,KIN]).max(axis=1)>=tau_d)|(gateH(last)>=tau_h)
            hp=[x.copy() for x in h]; cp=[x.copy() for x in c]
            p,hn,cn=m.step(X[:,t,:],[x.copy() for x in h],[x.copy() for x in c])
            for l in range(2):
                h[l]=np.where(act[:,None],hn[l],hp[l]); c[l]=np.where(act[:,None],cn[l],cp[l])
            P[:,t]=np.where(act,p,last); sup[:,t]=~act; last=P[:,t]
        return P,sup

    m=Np2(load_pt(M/"student_seed42.pt")); mb=Np2(load_pt(M/"baseline_seed42.pt"))
    Pfix=prefix_stream(m)
    h=[np.zeros((N,H),np.float32) for _ in range(2)]; c=[np.zeros((N,H),np.float32) for _ in range(2)]
    Pinc=np.zeros((N,T),np.float32)
    for t in range(T):
        p,h,c=m.step(X[:,t,:],h,c); Pinc[:,t]=p
    Pa,sa=hybrid_prefix(Pfix); Pb,sb=hybrid_inc(m)

    rows=[]
    for name,P,sup in [("fixed_prefix",Pfix,None),("fixed_incremental",Pinc,None),
                       ("mheg_prefix_recompute",Pa,sa),("mheg_incremental_skip",Pb,sb)]:
        fr,ttd,te=metrics(P)
        rows.append(dict(variant=name,FR=fr,TTD=ttd,TTDef=te,SupPct=0.0 if sup is None else 100*sup.mean()))
    out=pd.DataFrame(rows); print("\n== semanticas de inferencia (student, seed 42) =="); print(out.to_string(index=False))
    out.to_csv(O/"repro_inference_semantics_seed42.csv",index=False)

    pub=pd.read_csv(a.published); p42=pub[(pub.Seed==42)]
    ref=float(p42[p42.config_id=="aftkd_hybrid"].TTDef.iloc[0])
    assert abs(out[out.variant=="mheg_prefix_recompute"].TTDef.iloc[0]-ref)<5e-4, "nao reproduz o publicado"
    print(f"\nOK: mheg_prefix_recompute TTDef == publicado ({ref})")

    print("\n== reconciliacao Tabela 13 (stream prefix-window) ==")
    for nm,P in [("baseline_fixed",prefix_stream(mb)),("aftkd_fixed",Pfix)]:
        crit=[i for i in range(N) if yep[i]]
        pmax=np.array([P[i].max() for i in crit])
        miss=sum(1 for i in crit if det(P[i],int(np.where(yfr[i]>0.35)[0][0])) is None)
        fr=miss/len(crit)
        assert fr >= float((pmax<0.10).mean())-1e-9, "verificacao semantica violada"
        pubfr=float(p42[p42.config_id==nm].FailRate.iloc[0])
        assert abs(fr-pubfr)<1e-9, f"FR difere do publicado em {nm}"
        print(f"  {nm}: n={len(crit)} FR={fr:.4f} (=publicado) pbar_max={pmax.mean():.4f}±{pmax.std(ddof=1):.4f} frac>=0.10={(pmax>=0.10).mean()*100:.1f}%  [FR >= frac(p_max<theta): OK]")
    print("\nTudo reproduzido.")

if __name__=="__main__": main()
