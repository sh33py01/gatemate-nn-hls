#!/usr/bin/env python3
"""
export_tosa.py -- export a model with given weights to TOSA MLIR (01_tosa.mlir), batch 1,
then prove every parameter appears bit-for-bit among the exported constants.

Usage: python3 export_tosa.py weights.pth out/01_tosa.mlir [--model mynet|jet]
  mynet: MyNet (train_mynet.py), input 1x1x32x32
  jet  : hls4ml jet tagger without softmax (hls4ml_jet.py), input 1x16
Needs torch + torch_mlir, and train_mynet.py / hls4ml_jet.py next to this file.
"""
import os, re, struct, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

args = [a for a in sys.argv[1:] if not a.startswith('--')]
model_name = sys.argv[sys.argv.index('--model') + 1] if '--model' in sys.argv else 'mynet'
if model_name in args: args.remove(model_name)
pth, out = args

if model_name == 'mynet':
    from train_mynet import MyNet
    model, x = MyNet(), torch.ones(1, 1, 32, 32)
elif model_name == 'jet':
    from hls4ml_jet import JetTagger
    model, x = JetTagger(softmax=False), torch.ones(1, 16)
else:
    raise SystemExit(f"unknown model {model_name}")
model.load_state_dict(torch.load(pth, map_location='cpu'))
model.eval()

try:
    from torch_mlir import torchscript
    module = torchscript.compile(model, x, output_type="tosa")
except ImportError:
    import torch_mlir
    module = torch_mlir.compile(model, x, output_type="tosa")
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
open(out, 'w').write(str(module))
print(f"wrote {out}  (model {model_name})")

# ---- round trip: every parameter must equal (bit for bit) some f32 constant, in any layout ----
text = open(out).read()
consts = []
for body, shape_s in re.findall(r'"tosa\.const"\(\)\s*<\{value\s*=\s*dense<(.*?)>\s*:\s*tensor<([0-9x]+)xf32>\}>', text):
    n = int(np.prod([int(v) for v in shape_s.split('x')]))
    if body.startswith('"0x'):
        v = np.frombuffer(bytes.fromhex(body[3:-1]), '<f4')
    else:
        v = np.array([float(t) for t in re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', body)], np.float32)
        if v.size == 1 and n > 1: v = np.full(n, v[0], np.float32)
    consts.append(np.sort(v.astype(np.float32)))
bad = []
for k, p in model.state_dict().items():
    pv = np.sort(p.detach().numpy().astype(np.float32).ravel())
    if not any(c.shape == pv.shape and np.array_equal(c, pv) for c in consts):
        bad.append(k)
print(f"parameters found bit-exact in the MLIR: {len(model.state_dict()) - len(bad)}/{len(model.state_dict())}")
assert not bad, f"not found in the exported constants: {bad}"
with torch.no_grad():
    print(f"PyTorch output for all-ones input (expected RTL_OUT): {model(x).numpy()[0].tolist()}")
