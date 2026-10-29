#!/usr/bin/env python3
"""
extract_tosa_weights.py
Pull the exact weights baked into the hardware out of 01_tosa.mlir,
load them into MyNet, cross-check against an independent numpy
implementation of the TOSA graph, and save a .pth for gatemate_inference.py.

Usage:
    python3 extract_tosa_weights.py output/01_tosa.mlir -o output/tosa_weights.pth
"""
import argparse
import re
import struct

import numpy as np
import torch
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 5, stride=2, padding=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 16, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(self.fc3(x))


CONST_RE = re.compile(
    r'(%\d+)\s*=\s*"tosa\.const"\(\)\s*<\{value\s*=\s*dense<(.*?)>\s*:\s*tensor<([0-9x]+)xf32>\}>'
)
NUM_RE = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def parse_consts(path):
    text = open(path).read()
    consts = {}
    for name, body, shape_s in CONST_RE.findall(text):
        shape = tuple(int(s) for s in shape_s.split('x'))
        n = int(np.prod(shape))
        body = body.strip()
        if body.startswith('"0x'):                       # hex blob: raw little-endian f32
            raw = bytes.fromhex(body[3:-1])
            assert len(raw) == 4 * n, f"{name}: {len(raw)} bytes, expected {4*n}"
            vals = np.array(struct.unpack(f'<{n}f', raw), dtype=np.float32)
        else:
            vals = np.array([float(v) for v in NUM_RE.findall(body)], dtype=np.float32)
            if vals.size == 1 and n > 1:                 # splat constant
                vals = np.full(n, vals[0], dtype=np.float32)
            assert vals.size == n, f"{name}: parsed {vals.size} values, expected {n}"
        assert shape not in consts, f"duplicate shape {shape}: shape-based mapping unsafe"
        consts[shape] = (name, vals.reshape(shape))
    return consts


def build_state_dict(c):
    def get(shape):
        assert shape in c, f"no f32 constant with shape {shape} in the MLIR"
        return c[shape][1]
    return {
        # TOSA conv weight is OHWI -> PyTorch OIHW
        'conv.weight': torch.from_numpy(get((1, 5, 5, 1)).transpose(0, 3, 1, 2).copy()),
        'conv.bias':   torch.from_numpy(get((1,)).copy()),
        # Linear weights are stored in PyTorch [out, in] layout (TOSA transposes them later)
        'fc1.weight':  torch.from_numpy(get((8, 256)).copy()),
        'fc1.bias':    torch.from_numpy(get((1, 8)).reshape(8).copy()),
        'fc2.weight':  torch.from_numpy(get((4, 8)).copy()),
        'fc2.bias':    torch.from_numpy(get((1, 4)).reshape(4).copy()),
        'fc3.weight':  torch.from_numpy(get((2, 4)).copy()),
        'fc3.bias':    torch.from_numpy(get((1, 2)).reshape(2).copy()),
    }


def tosa_graph_numpy(x, c):
    """Independent re-implementation of the TOSA ops in 01_tosa.mlir (NHWC conv)."""
    w = c[(1, 5, 5, 1)][1]
    b = c[(1,)][1]
    img = np.pad(x.reshape(32, 32), 2)                    # pad 2,2,2,2
    conv = np.empty((16, 16), dtype=np.float64)
    for i in range(16):
        for j in range(16):
            conv[i, j] = np.sum(img[2*i:2*i+5, 2*j:2*j+5] * w[0, :, :, 0]) + b[0]
    h = np.maximum(conv.reshape(1, 256), 0)
    h = np.maximum(h @ c[(8, 256)][1].T + c[(1, 8)][1], 0)
    h = np.maximum(h @ c[(4, 8)][1].T + c[(1, 4)][1], 0)
    z = h @ c[(2, 4)][1].T + c[(1, 2)][1]
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mlir')
    ap.add_argument('-o', '--out', default='tosa_weights.pth')
    args = ap.parse_args()

    consts = parse_consts(args.mlir)
    print("Constants found (shape <- SSA name):")
    for shape, (name, _) in sorted(consts.items(), key=lambda kv: kv[1][0]):
        print(f"  {name:>4}  {shape}")

    model = MyNet()
    model.load_state_dict(build_state_dict(consts))      # strict: fails on any missing/extra key
    model.eval()

    # Cross-check PyTorch model vs independent numpy TOSA graph on several inputs
    rng = np.random.default_rng(0)
    tests = {'zeros': np.zeros((1, 1, 32, 32), np.float32),
             'ones':  np.ones((1, 1, 32, 32), np.float32)}
    for k in range(3):
        tests[f'randn{k}'] = rng.standard_normal((1, 1, 32, 32)).astype(np.float32)

    worst = 0.0
    for label, x in tests.items():
        with torch.no_grad():
            pt = model(torch.from_numpy(x)).numpy()
        ref = tosa_graph_numpy(x.astype(np.float64), consts)
        err = float(np.abs(pt - ref).max())
        worst = max(worst, err)
        print(f"  {label:7s} torch={pt[0]}  numpy-tosa={ref[0]}  max|diff|={err:.2e}")
    assert worst < 1e-5, f"PyTorch model disagrees with TOSA graph (max diff {worst}) - mapping is wrong"
    print("PASS: PyTorch model reproduces the TOSA graph")

    torch.save(model.state_dict(), args.out)
    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()
