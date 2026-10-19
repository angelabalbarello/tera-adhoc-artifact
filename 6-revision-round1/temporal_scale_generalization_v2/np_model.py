#!/usr/bin/env python3
"""Carregador manual de checkpoints .pt (zip do PyTorch) + forward LSTM em numpy.
Reproduz MultiTaskLSTM(D=31,H=64,bi=False,num_layers=2) em modo eval (dropout off).
"""
import zipfile, pickle, io, hashlib
import numpy as np

_DTYPES = {'FloatStorage': np.float32, 'DoubleStorage': np.float64,
           'LongStorage': np.int64, 'IntStorage': np.int32, 'HalfStorage': np.float16}

def load_state_dict(pt_path):
    zf = zipfile.ZipFile(pt_path)
    names = zf.namelist()
    pkl_name = [n for n in names if n.endswith('data.pkl')][0]
    prefix = pkl_name[:-len('data.pkl')]
    storages = {}
    class Stub:
        def __init__(self, *a, **k): pass
    class TensorStub:
        def __init__(self, storage_key, dtype): self.key, self.dtype = storage_key, dtype
    def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, *extra):
        key, dtype = storage
        raw = zf.read(prefix + 'data/' + key)
        arr = np.frombuffer(raw, dtype=dtype).copy()
        if storage_offset: arr = arr[storage_offset:]
        n = int(np.prod(size)) if size else 1
        # stride-aware reconstruction (checkpoints salvos são contiguous)
        return np.lib.stride_tricks.as_strided(arr, shape=tuple(size),
                strides=tuple(s*arr.itemsize for s in stride)).copy() if size else arr[:1].copy()
    class Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == '_rebuild_tensor_v2': return rebuild_tensor_v2
            if name == 'OrderedDict':
                import collections; return collections.OrderedDict
            if name.endswith('Storage'):
                cls = type(name, (), {'dtype': _DTYPES.get(name, np.float32)})
                return cls
            return Stub
        def persistent_load(self, pid):
            # ('storage', StorageClass, key, device, numel)
            _, storage_cls, key, device, numel = pid
            return (key, getattr(storage_cls, 'dtype', np.float32))
    sd = Unpickler(io.BytesIO(zf.read(pkl_name))).load()
    return {k: np.asarray(v) for k, v in sd.items()}

def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()

def _sigmoid(x): return 1.0/(1.0+np.exp(-x))

class NpLSTM:
    """LSTM 2 camadas causal + head de frame, batch vetorizado."""
    def __init__(self, sd):
        self.p = {}
        for l in (0,1):
            for nm in ('weight_ih','weight_hh','bias_ih','bias_hh'):
                self.p[f'{nm}_l{l}'] = sd[f'lstm.{nm}_l{l}'].astype(np.float32)
        self.w_fr = sd['fc_fr.weight'].astype(np.float32)   # (1,64)
        self.b_fr = sd['fc_fr.bias'].astype(np.float32)
    def forward_last_frame_logit(self, Xw):
        """Xw: (B, W, D) janela causal; retorna logit do frame final (B,)."""
        B, W, D = Xw.shape
        h = [np.zeros((B,64),np.float32), np.zeros((B,64),np.float32)]
        c = [np.zeros((B,64),np.float32), np.zeros((B,64),np.float32)]
        inp = Xw
        for l in (0,1):
            Wi, Wh = self.p[f'weight_ih_l{l}'], self.p[f'weight_hh_l{l}']
            bi, bh = self.p[f'bias_ih_l{l}'], self.p[f'bias_hh_l{l}']
            outs = np.empty((B,W,64), np.float32)
            ht, ct = h[l], c[l]
            xW = inp @ Wi.T + bi  # (B,W,256) pre-computa parte de entrada
            for t in range(W):
                g = xW[:,t,:] + ht @ Wh.T + bh
                i_g = _sigmoid(g[:, 0:64]); f_g = _sigmoid(g[:, 64:128])
                g_g = np.tanh(g[:,128:192]); o_g = _sigmoid(g[:,192:256])
                ct = f_g*ct + i_g*g_g
                ht = o_g*np.tanh(ct)
                outs[:,t,:] = ht
            inp = outs
        return (inp[:,-1,:] @ self.w_fr.T + self.b_fr)[:,0]

def make_prefix_window(x_seq, t, window=96):
    x_slice = x_seq[:t+1, :]
    if x_slice.shape[0] < window:
        pad = np.repeat(x_slice[0:1,:], window - x_slice.shape[0], axis=0)
        x_slice = np.concatenate([pad, x_slice], axis=0)
    else:
        x_slice = x_slice[-window:, :]
    return x_slice.astype(np.float32, copy=False)

def fixed_policy_probs(model, X, window=96):
    """P[i,t] = prob do modelo com janela-prefixo em t (igual infer_stream_fixed).
    Vetorizado no eixo dos episodios."""
    n, T, D = X.shape
    P = np.zeros((n,T), np.float32)
    for t in range(T):
        Xw = np.stack([make_prefix_window(X[i], t, window) for i in range(n)])
        P[:, t] = _sigmoid(model.forward_last_frame_logit(Xw))
    return P
