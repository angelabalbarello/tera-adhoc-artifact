#!/usr/bin/env python3
"""Verify deployed-model complexity directly from a saved checkpoint.

Reads the state_dict of the causal student (2-layer LSTM) without needing a
working torch/CUDA install (parses the torch zip/pickle) and prints parameters,
MACs/timestep and FLOPs/timestep under an explicit convention.

Convention: 1 MAC (multiply-accumulate) = 2 FLOPs. Biases counted as 1 add each.
"""
import zipfile, pickle, io, math, sys, argparse

def load_shapes(path):
    zf = zipfile.ZipFile(path)
    pk = [n for n in zf.namelist() if n.endswith('data.pkl')][0]
    def rebuild(storage, off, size, stride, *a, **k):
        return ('T', tuple(size))
    class Stub:  # noqa
        def __init__(self, *a, **k): pass
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == '_rebuild_tensor_v2':
                return rebuild
            if mod.startswith('torch'):
                return Stub
            return super().find_class(mod, name)
        def persistent_load(self, pid):
            return None
    obj = U(io.BytesIO(zf.read(pk))).load()
    d = obj.get('state_dict', obj) if isinstance(obj, dict) else obj
    return {k: v[1] for k, v in d.items() if isinstance(v, tuple)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', nargs='?', default='../../../../FGCS/Inferencia/modelos_salvos/abl_A/student_seed42.pt')
    args = ap.parse_args()
    shapes = load_shapes(args.checkpoint)
    total = 0
    macs = 0
    adds_bias = 0
    layers = {}
    for k, shp in shapes.items():
        n = math.prod(shp) if shp else 1
        total += n
        if 'weight' in k:
            macs += n          # each weight = 1 MAC per timestep (seq nets reuse weights every step)
        else:
            adds_bias += n
        layers[k] = (shp, n)
    D = shapes['lstm.weight_ih_l0'][1]
    H = shapes['lstm.weight_hh_l0'][1]
    nlayers = 1 + max(int(k.split('_l')[-1]) for k in shapes if k.startswith('lstm.weight_ih'))
    print(f'checkpoint: {args.checkpoint}')
    for k, (shp, n) in layers.items():
        print(f'  {k:42s} {shp}  {n}')
    print(f'D (input dim)      : {D}')
    print(f'H (hidden dim)     : {H}')
    print(f'LSTM layers        : {nlayers}')
    print(f'TOTAL parameters   : {total}')
    print(f'MACs per timestep  : {macs}  (= sum of weight-matrix elements)')
    print(f'bias adds/timestep : {adds_bias}')
    print(f'FLOPs per timestep : {2*macs + adds_bias}  (1 MAC = 2 FLOPs) '
          f'~= {(2*macs+adds_bias)/1000:.1f} KFLOPs')
    print(f'ops per timestep   : {macs + adds_bias}  (1 MAC = 1 op) '
          f'~= {(macs+adds_bias)/1000:.1f} K ops')
    print('Per-layer closed form: layer1 = 4(DH + H^2 + 2H); layer_k>1 = 4(2H^2 + 2H); heads = 2(H+1)')
    l1 = 4*(D*H + H*H + 2*H); l2 = 4*(2*H*H + 2*H); heads = 2*(H+1)
    print(f'  closed form params: {l1} + {l2} + {heads} = {l1+l2+heads}')

if __name__ == '__main__':
    main()
