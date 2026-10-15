import zipfile, pickle, io
import numpy as np
DTYPES={"FloatStorage":np.float32,"DoubleStorage":np.float64,"LongStorage":np.int64,"IntStorage":np.int32,"HalfStorage":np.float16,"BoolStorage":np.bool_}
def load_pt(path):
    zf=zipfile.ZipFile(path)
    prefix=zf.namelist()[0].split("/")[0]
    storages={}
    class Stor:
        def __init__(self,key,dtype): self.key=key; self.dtype=dtype
    class Unp(pickle.Unpickler):
        def find_class(self,mod,name):
            if name=="_rebuild_tensor_v2":
                def rt(storage,offset,size,stride,requires_grad,hooks,*a):
                    data=storages[storage.key]
                    n=int(np.prod(size)) if size else 1
                    arr=data[offset:offset+n] if size else data[offset:offset+1]
                    if size:
                        return np.lib.stride_tricks.as_strided(arr, shape=size, strides=[s*arr.itemsize for s in stride]).copy()
                    return arr.copy()
                return rt
            if name in DTYPES: return DTYPES[name]
            if name=="OrderedDict":
                from collections import OrderedDict; return OrderedDict
            return lambda *a,**k: None
        def persistent_load(self,pid):
            dtype_cls,key=pid[1],pid[2]
            dt=dtype_cls if isinstance(dtype_cls,type) else np.float32
            if key not in storages:
                raw=zf.read(f"{prefix}/data/{key}")
                storages[key]=np.frombuffer(raw,dtype=dt)
            return Stor(key,dt)
    sd=Unp(io.BytesIO(zf.read(f"{prefix}/data.pkl"))).load()
    return {k:np.asarray(v,dtype=np.float32) for k,v in sd.items()}
def sigmoid(x): return 1/(1+np.exp(-x))
class NpLSTM2:
    def __init__(self, sd, H=64):
        self.H=H
        self.l=[(sd[f"lstm.weight_ih_l{li}"],sd[f"lstm.weight_hh_l{li}"],sd[f"lstm.bias_ih_l{li}"],sd[f"lstm.bias_hh_l{li}"]) for li in (0,1)]
        self.wf=sd["fc_fr.weight"].reshape(-1); self.bf=float(sd["fc_fr.bias"].reshape(-1)[0])
    def last_logit(self, X):
        H=self.H
        h=[np.zeros(H,dtype=np.float32),np.zeros(H,dtype=np.float32)]
        c=[np.zeros(H,dtype=np.float32),np.zeros(H,dtype=np.float32)]
        for t in range(X.shape[0]):
            inp=X[t]
            for li,(Wih,Whh,bih,bhh) in enumerate(self.l):
                g=Wih@inp+bih+Whh@h[li]+bhh
                i,f,gg,o=g[:H],g[H:2*H],g[2*H:3*H],g[3*H:]
                c[li]=sigmoid(f)*c[li]+sigmoid(i)*np.tanh(gg)
                h[li]=sigmoid(o)*np.tanh(c[li])
                inp=h[li]
        return float(self.wf@h[1]+self.bf)
def prefix(xs,t,w=96):
    sl=xs[:t+1]
    if sl.shape[0]<w:
        pad=np.repeat(xs[0:1],w-sl.shape[0],axis=0); sl=np.concatenate([pad,sl],0)
    else: sl=sl[-w:]
    return sl
