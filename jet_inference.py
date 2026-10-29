#!/usr/bin/env python3
"""
jet_inference.py -- run the hls4ml jet tagger on the GateMate over UART.
Sends 16 raw ap_fixed<16,6> inputs (32 bytes), receives AA 55 | 5 x int64 raw ap_fixed<38,18> |
cycles | flags. Checks every output against hls4ml's bit-accurate emulation and reports accuracy.

Usage: python3 hw/jet_inference.py --port /dev/ttyUSB0 \
          --emu output/jet/hls4ml_prj_emulation.npz --n 1000 --csv output/jet/board_M4.csv
"""
import argparse, struct, time
import numpy as np
import serial

ap = argparse.ArgumentParser()
ap.add_argument('--port', default='/dev/ttyUSB0')
ap.add_argument('--baud', type=int, default=57600)
ap.add_argument('--emu', required=True, help='*_emulation.npz from hls4ml_jet.py')
ap.add_argument('--n', type=int, default=1000)
ap.add_argument('--in-w', type=int, default=16); ap.add_argument('--in-frac', type=int, default=10)
ap.add_argument('--out-frac', type=int, default=20)
ap.add_argument('--csv')
args = ap.parse_args()

d = np.load(args.emu)
x, y, emu = d['x'][:args.n], d['y'][:args.n], d['out'][:args.n]
W = args.in_w
raw_in = np.floor(x.astype(np.float64) * 2 ** args.in_frac).astype(np.int64)
raw_in = ((raw_in + 2 ** (W - 1)) % 2 ** W) - 2 ** (W - 1)            # ap_fixed: truncate + wrap
raw_emu = np.round(emu.astype(np.float64) * 2 ** args.out_frac).astype(np.int64)

def infer(ser, xi):
    ser.reset_input_buffer()
    ser.write(struct.pack('<16h', *xi)); ser.flush()
    buf, t0 = b'', time.time()
    while time.time() - t0 < 3:
        buf += ser.read(ser.in_waiting or 1)
        i = buf.find(b'\xAA\x55')
        if i >= 0 and len(buf) >= i + 2 + 40 + 5:
            out = struct.unpack('<5q', buf[i + 2:i + 42])
            cyc, flags = struct.unpack('<IB', buf[i + 42:i + 47])
            return np.array(out), cyc, flags
    raise TimeoutError(f"no reply ({len(buf)} bytes: {buf[:48].hex()})")

exact = hw_ok = emu_ok = 0; cycs = set(); rows = []
t0 = time.time()
with serial.Serial(args.port, args.baud, timeout=0.05) as ser:
    time.sleep(0.2)
    for i in range(len(x)):
        out, cyc, flags = infer(ser, raw_in[i])
        cycs.add(cyc)
        same = bool((out == raw_emu[i]).all()) and flags == 0
        exact += same
        hw_ok += int(np.argmax(out) == y[i]); emu_ok += int(np.argmax(raw_emu[i]) == y[i])
        rows.append([i, int(y[i]), int(np.argmax(out)), int(np.argmax(raw_emu[i])), int(same), cyc, flags] + list(out))
        if (i + 1) % 100 == 0 or i + 1 == len(x):
            print(f"{i+1:5d}/{len(x)}  bit-exact {exact}/{i+1}  board acc {hw_ok/(i+1):.4f}  "
                  f"emulation acc {emu_ok/(i+1):.4f}  ({time.time()-t0:.0f} s)")
n = len(x)
print(f"\nBit-exact vs hls4ml emulation: {exact}/{n}")
print(f"Board accuracy     : {hw_ok/n:.4f}   (hls4ml emulation on same jets: {emu_ok/n:.4f})")
print(f"Cycles per inference: {sorted(cycs)}  -> {min(cycs)/10e6*1e6:.1f} us @ 10 MHz")
if args.csv:
    with open(args.csv, 'w') as f:
        f.write('idx,label,board_pred,emu_pred,bit_exact,cycles,flags,o0,o1,o2,o3,o4\n')
        for r in rows: f.write(','.join(map(str, r)) + '\n')
    print(f"per-jet results -> {args.csv}")
