"""Engine da campanha incremental (protocolo: PROTOCOL_FROZEN.md).

Execucao single-step com estado persistente, em numpy (leitor de checkpoint
sem torch, validado contra os valores publicados — ver scientific-audit).
"""
import zipfile, pickle, io, pathlib
import numpy as np

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
                    return np.lib.stride_tricks.as_strided(arr,shape=size,
                        strides=[st*arr.itemsize for st in stride]).copy()
                return rt
            if name in DTYPES: return DTYPES[name]
            if name=="OrderedDict":
                from collections import OrderedDict; return OrderedDict
            return lambda *a,**k: None
        def persistent_load(self,pid):
            dt=pid[1] if isinstance(pid[1],type) else np.float32
            key=pid[2]
            if key not in storages:
                storages[key]=np.frombuffer(zf.read(f"{prefix}/data/{key}"),dtype=dt)
            return Stor(key,dt)
    sd=Unp(io.BytesIO(zf.read(f"{prefix}/data.pkl"))).load()
    return {k:np.asarray(v,dtype=np.float32) for k,v in sd.items()}

def sigmoid(x): return 1/(1+np.exp(-x))
H=64; KIN=[12,13,14]; DT=10.0/96

class Model:
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

def binH(p):
    pc=np.clip(p,1e-7,1-1e-7)
    return -(pc*np.log2(pc)+(1-pc)*np.log2(1-pc))

def fixed_incremental(m,X):
    """Politica fixa single-step: uma passada com estado (equivale a full sequence)."""
    N,T,_=X.shape
    h=[np.zeros((N,H),np.float32) for _ in range(2)]
    c=[np.zeros((N,H),np.float32) for _ in range(2)]
    P=np.zeros((N,T),np.float32)
    for t in range(T):
        p,h,c=m.step(X[:,t,:],h,c); P[:,t]=p
    return P

def gated_incremental_multi(m,X,tauds,tauhs,calib=None,ent_lag=0,kin_zero=False,kin_lag=0,check_frozen=False):
    """Execucao incremental COM gate, para C combos (tau_d,tau_h) em paralelo.
    Estado shape (C*N,H). Retorna P (C,N,T) e sup (C,N,T).
    calib: funcao prob->prob aplicada na emissao de cada update (IC)."""
    C=len(tauds); N,T,_=X.shape
    td=np.repeat(np.asarray(tauds,np.float32),N)   # (C*N,)
    th=np.repeat(np.asarray(tauhs,np.float32),N)
    Xr=np.tile(X,(C,1,1))                           # (C*N,T,D)
    h=[np.zeros((C*N,H),np.float32) for _ in range(2)]
    c=[np.zeros((C*N,H),np.float32) for _ in range(2)]
    last=np.zeros(C*N,np.float32)
    P=np.zeros((C*N,T),np.float32); sup=np.zeros((C*N,T),bool)
    # historico de p para lag entropico
    hist=[last.copy()]
    for t in range(T):
        if t==0:
            act=np.ones(C*N,bool)
        else:
            if kin_zero: dk=np.zeros(C*N,np.float32)
            else:
                tk=max(1,t-kin_lag)
                dk=np.abs(Xr[:,tk,KIN]-Xr[:,tk-1,KIN]).max(axis=1)
            pl=hist[max(0,len(hist)-1-ent_lag)]
            act=(dk>=td)|(binH(pl)>=th)
        hp=[a.copy() for a in h]; cp=[a.copy() for a in c]
        pn,hn,cn=m.step(Xr[:,t,:],[a.copy() for a in h],[a.copy() for a in c])
        if calib is not None: pn=calib(pn)
        for l in range(2):
            h[l]=np.where(act[:,None],hn[l],hp[l]); c[l]=np.where(act[:,None],cn[l],cp[l])
        if check_frozen and t>0:
            # Emenda 2 item 5: TODAS as 4 matrizes de estado congeladas (bit a bit)
            for _l in range(2):
                assert np.array_equal(h[_l][~act],hp[_l][~act]), f"h{_l} nao congelado t={t}"
                assert np.array_equal(c[_l][~act],cp[_l][~act]), f"c{_l} nao congelado t={t}"
        P[:,t]=np.where(act,pn,last); sup[:,t]=~act; last=P[:,t]
        hist.append(last.copy())
    return P.reshape(C,N,T),sup.reshape(C,N,T)

def stable_det(p,onset,thr,m=3):
    c=0
    for t in range(onset,len(p)):
        c=c+1 if p[t]>=thr else 0
        if c>=m: return t-m+1
    return None

def det_full(p,thr,m=3):
    c=0
    for t in range(len(p)):
        c=c+1 if p[t]>=thr else 0
        if c>=m: return t-m+1
    return None

def metrics(P,yfr,yep,prog,theta,thr_ep,sup=None):
    """Metricas do protocolo (teste). Retorna dict + logs por episodio."""
    N,T=P.shape; logs=[]
    miss=0; tts=[]; causal=[]; ants=[]; pmaxs=[]
    for i in range(N):
        crit=int(yep[i]); pmax=float(P[i].max())
        onset=int(np.where(yfr[i]>0.35)[0][0]) if crit and (yfr[i]>0.35).any() else -1
        d=stable_det(P[i],onset,theta) if crit and onset>=0 else None
        if crit:
            pmaxs.append(pmax)
            if d is None: miss+=1
            else:
                tts.append((d-onset)*DT); causal.append((d+2-onset)*DT)
            if prog[i]:
                df=det_full(P[i],theta)
                ants.append(max(0.0,(onset-df)*DT) if df is not None else 0.0)
        logs.append(dict(episode_id=i,is_critical=crit,onset_frame=onset,p_max=round(pmax,5),
            t_det_retro=-1 if (not crit or d is None) else d,
            t_det_causal=-1 if (not crit or d is None) else d+2,
            detected=int(bool(crit and d is not None)),
            suppressed_frames=int(sup[i].sum()) if sup is not None else 0,
            steps_executed=T-(int(sup[i].sum()) if sup is not None else 0)))
    ncrit=int(yep.sum()); fr=miss/ncrit
    ep_hat=(P.max(axis=1)>=thr_ep).astype(int)
    fr_cls=float(((yep==1)&(ep_hat==0)).sum()/max(ncrit,1))
    tp=int(((yep==1)&(ep_hat==1)).sum()); fp=int(((yep==0)&(ep_hat==1)).sum()); fn=ncrit-tp
    f1=2*tp/max(2*tp+fp+fn,1)
    ttd=float(np.mean(tts)) if tts else 0.0
    out=dict(FR=fr,n_det=ncrit-miss,n_miss=miss,FR_cls=fr_cls,
        TTD_b=ttd,TTDef=ttd+fr*10.0,
        causal_delay=float(np.mean(causal)) if causal else 0.0,
        A_p=float(np.mean(ants)) if ants else float('nan'),
        F1_maxrule=f1,
        SupPct=0.0 if sup is None else 100*float(sup.mean()),
        steps_per_frame=1.0 if sup is None else float(1-sup.mean()),
        frac_pmax_below=float((np.array(pmaxs)<theta).mean()))
    assert out["FR"] >= out["frac_pmax_below"]-1e-9
    return out,logs
