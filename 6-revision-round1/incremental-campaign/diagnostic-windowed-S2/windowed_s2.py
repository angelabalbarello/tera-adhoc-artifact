"""Diagnostico exploratorio S2 na semantica de JANELA (DIAGNOSTIC_PREREG.md).
Implementa fielmente tera_infer.py::infer_stream_hybrid em numpy:
janela causal W=96 (pad esquerda frame 0), recomputacao na atualizacao, supressao
reutiliza ultima p emitida, 1a atualizacao forcada (t=0). Sem refresh/timeout.
Uso: python3 windowed_s2.py SEED [SEED...]"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0,str(pathlib.Path(__file__).parents[1]))
import incremental_engine as E
HERE=pathlib.Path(__file__).parent; CAMP=HERE.parent
R=CAMP.parents[1]/"1-pipeline-tera/results/exp_seed12_round1"
cal=pd.read_csv(R/"calibration/calibration_summary.csv")
W=96

def window_probs_batch(m,Xw):
    """Roda a LSTM sobre janelas (B,W,D) do zero; retorna p do ultimo passo."""
    B=Xw.shape[0]
    h=[np.zeros((B,E.H),np.float32) for _ in range(2)]
    c=[np.zeros((B,E.H),np.float32) for _ in range(2)]
    for t in range(W):
        p,h,c=m.step(Xw[:,t,:],h,c)
    return p

def prefix_windows(X,eps,t):
    """make_prefix_window vetorizada p/ episodios eps no instante t."""
    xs=X[eps,:t+1,:]
    if t+1<W:
        pad=np.repeat(xs[:,0:1,:],W-(t+1),axis=1)
        return np.concatenate([pad,xs],axis=1)
    return X[eps,t+1-W:t+1,:]

def run_hybrid(m,X,tau_d,tau_h,kin_zero_gate=False):
    N,T,_=X.shape
    P=np.zeros((N,T),np.float32); sup=np.zeros((N,T),bool)
    last=np.zeros(N,np.float32)
    gate_log=[]  # (ep,t,dk,H,activated)
    for t in range(T):
        if t==0:
            act=np.ones(N,bool); dk=np.full(N,np.nan); Hl=np.full(N,np.nan)
        else:
            dk_real=np.abs(X[:,t,:][:,E.KIN]-X[:,t-1,:][:,E.KIN]).max(axis=1)
            dk=np.zeros(N,np.float32) if kin_zero_gate else dk_real
            Hl=E.binH(last)
            act=(dk>=tau_d)|(Hl>=tau_h)
        eps=np.where(act)[0]
        if len(eps):
            P[eps,t]=window_probs_batch(m,prefix_windows(X,eps,t))
        P[~act,t]=last[~act]; sup[:,t]=~act; last=P[:,t]
        for i in range(N):
            gate_log.append((i,t,float(dk[i]) if t>0 else -1.0,
                             float(Hl[i]) if t>0 else -1.0,bool(act[i])))
    return P,sup,gate_log

def eval_cond(P,sup,yfr,yep,theta):
    N,T=P.shape; miss=0; tts=[]; rows=[]
    longest=0; frozen_to_end=0
    for i in range(N):
        s=sup[i]
        # maior sequencia consecutiva sem atualizacao
        run=0; mx=0
        for v in s:
            run=run+1 if v else 0; mx=max(mx,run)
        longest=max(longest,mx)
        fte=bool(s[-1] and s[np.where(~s)[0][-1]+1:].all()) if (~s).any() else False
        if s.any() and fte: frozen_to_end+=1
        crit=int(yep[i])
        onset=int(np.where(yfr[i]>0.35)[0][0]) if crit and (yfr[i]>0.35).any() else -1
        d=E.stable_det(P[i],onset,theta) if crit and onset>=0 else None
        if crit:
            if d is None: miss+=1
            else: tts.append((d-onset)*E.DT)
        upd=np.where(~s)[0]
        rows.append(dict(episode_id=i,is_critical=crit,onset_frame=onset,
            detected=int(bool(crit and d is not None)),
            t_det=-1 if (not crit or d is None) else int(d),
            n_updates=int((~s).sum()),longest_no_update=int(mx),
            frozen_to_end=int(fte),last_update=int(upd[-1]) if len(upd) else -1,
            p_final=round(float(P[i,-1]),5)))
    ncrit=int(yep.sum()); fr=miss/ncrit
    ttd=float(np.mean(tts)) if tts else 0.0
    met=dict(FR=round(fr,4),n_crit=ncrit,n_miss=miss,TTD=round(ttd,4),
             TTDef=round(ttd+fr*10.0,4),SupPct=round(100*float(sup.mean()),4),
             longest_no_update=int(longest),episodes_frozen_to_end=frozen_to_end)
    return met,rows

def run_seed(seed):
    row=cal[cal.seed==seed].iloc[0]
    theta=float(row.theta_ttd); tau_d=float(row.tau_delta); tau_h=float(row.tau_h)
    X=np.load(R/f"data/X_te_seed{seed}.npy").astype(np.float32)
    yfr=np.load(R/f"data/y_fr_te_seed{seed}.npy").astype(np.float32)
    yep=np.load(R/f"data/y_ep_te_seed{seed}.npy").astype(int)
    out=[]
    for role,cfgname in [("baseline","baseline_hybrid"),("student","aftkd_hybrid")]:
        m=E.Model(E.load_pt(R/f"models/{role}_seed{seed}.pt"))
        for cond,kz in [("A_normal",False),("B_kin_gate_zero",True)]:
            P,S,gl=run_hybrid(m,X,tau_d,tau_h,kin_zero_gate=kz)
            met,rows=eval_cond(P,S,yfr,yep,theta)
            out.append(dict(seed=seed,config=cfgname,cond=cond,tau_delta=tau_d,tau_h=tau_h,**met))
            pd.DataFrame(rows).to_csv(HERE/f"logs_ep_seed{seed}_{cfgname}_{cond}.csv",index=False)
            pd.DataFrame(gl,columns=["ep","t","dk_gate","H_gate","activated"]).to_csv(
                HERE/f"logs_gate_seed{seed}_{cfgname}_{cond}.csv.gz",index=False,compression="gzip")
            print(seed,cfgname,cond,met,flush=True)
    f=HERE/"s2_windowed_results.csv"
    df=pd.DataFrame(out)
    if f.exists():
        old=pd.read_csv(f); old=old[old.seed!=seed]; df=pd.concat([old,df])
    df.to_csv(f,index=False)

def done(seed):
    f=HERE/"s2_windowed_results.csv"
    return f.exists() and (pd.read_csv(f).seed==seed).any()

if __name__=="__main__":
    for s in [int(a) for a in sys.argv[1:]]:
        if done(s): print(s,"skip"); continue
        run_seed(s)
