#!/usr/bin/env python3
"""
jetf_inference.py -- jet tagger from the SODA float32 flow on the GateMate.
Sends 16 float32 (64 bytes), receives AA 55 | 5 float32 + 4 pad | cycles | flags, and compares
with the PyTorch model (no softmax, logits) on the same standardized test jets.

Usage: python3 hw/jetf_inference.py --port /dev/ttyUSBx --weights output/jet/jet_model.pth \
          --data output/jet/test_set.npz --n 1000 --csv output/jetf/board.csv
"""
import argparse, os, struct, sys, time
import numpy as np, serial, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hls4ml_jet import JetTagger

ap = argparse.ArgumentParser()
ap.add_argument('--port', default='/dev/ttyUSB0'); ap.add_argument('--baud', type=int, default=57600)
ap.add_argument('--weights', required=True); ap.add_argument('--data', required=True)
ap.add_argument('--n', type=int, default=1000); ap.add_argument('--csv')
args = ap.parse_args()

m = JetTagger(softmax=False); m.load_state_dict(torch.load(args.weights, map_location='cpu')); m.eval()
d = np.load(args.data, allow_pickle=True)
x, y = d['x'][:args.n].astype(np.float32), d['y'][:args.n]
with torch.no_grad():
    ref = m(torch.from_numpy(x)).numpy()

def infer(ser, xi):
    ser.reset_input_buffer(); ser.write(xi.astype('<f4').tobytes()); ser.flush()
    buf, t0 = b'', time.time()
    while time.time() - t0 < 3:
        buf += ser.read(ser.in_waiting or 1)
        i = buf.find(b'\xAA\x55')
        if i >= 0 and len(buf) >= i + 2 + 24 + 5:
            out = np.array(struct.unpack('<5f', buf[i + 2:i + 22]), np.float32)
            cyc, flags = struct.unpack('<IB', buf[i + 26:i + 31])
            return out, cyc, flags
    raise TimeoutError(f"no reply ({len(buf)} bytes)")

ok = agree = 0; worst = 0.0; cycs = set(); rows = []; t0 = time.time()
with serial.Serial(args.port, args.baud, timeout=0.05) as ser:
    time.sleep(0.2)
    for i in range(len(x)):
        out, cyc, flags = infer(ser, x[i]); cycs.add(cyc)
        err = float(np.abs(out - ref[i]).max()); worst = max(worst, err)
        ok += int(np.argmax(out) == y[i]); agree += int(np.argmax(out) == np.argmax(ref[i]))
        rows.append([i, int(y[i]), int(np.argmax(out)), int(np.argmax(ref[i])), err, cyc, flags])
        if (i + 1) % 100 == 0 or i + 1 == len(x):
            print(f"{i+1:5d}/{len(x)}  board acc {ok/(i+1):.4f}  agree {agree}/{i+1}  max abs err {worst:.3g}  ({time.time()-t0:.0f} s)")
n = len(x)
print(f"\nBoard accuracy: {ok/n:.4f}   PyTorch float32 on same jets: {float((ref.argmax(1) == y).mean()):.4f}")
print(f"Prediction agreement board vs PyTorch: {agree}/{n};  max abs error on logits {worst:.3g}")
print(f"Cycles per inference: {sorted(cycs)} -> {min(cycs)/10e6*1e6:.1f} us @ 10 MHz")
if args.csv:
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, 'w') as f:
        f.write('idx,label,board_pred,pytorch_pred,max_abs_err,cycles,flags\n')
        for r in rows: f.write(','.join(map(str, r)) + '\n')
