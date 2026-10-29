#!/usr/bin/env python3
"""
gatemate_inference.py -- run MyNet on the GateMate board over UART and compare
with the PyTorch model loaded with the exact weights baked into the hardware.

Board protocol (top_uart.v): send 4096 bytes (1x1x32x32 float32 LE);
board replies AA 55 | out0 f32 | out1 f32 | cycles u32 | flags u8  (15 bytes).

Usage:
  python3 gatemate_inference.py --weights output/tosa_weights.pth
  python3 gatemate_inference.py --weights output/tosa_weights.pth --random 20
  python3 gatemate_inference.py --weights output/trained/mynet_trained.pth \
      --dataset output/trained/test_set.npz --n 500 --csv output/trained/board_results.csv
"""
import argparse, struct, time
import numpy as np
import serial
import torch
import torch.nn as nn

F_CLK = 10e6


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 5, stride=2, padding=2)
        self.fc1, self.fc2, self.fc3 = nn.Linear(256, 8), nn.Linear(8, 4), nn.Linear(4, 2)

    def forward(self, x):
        x = torch.relu(self.conv(x)).flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)


def ulps(a, b):
    ia = np.frombuffer(np.float32(a).tobytes(), np.int32)[0]
    ib = np.frombuffer(np.float32(b).tobytes(), np.int32)[0]
    return abs(int(ia) - int(ib))


def board_infer(ser, x):
    ser.reset_input_buffer()
    ser.write(x.astype('<f4').tobytes())
    ser.flush()
    buf = b''
    deadline = time.time() + 5.0
    while time.time() < deadline:
        buf += ser.read(ser.in_waiting or 1)
        i = buf.find(b'\xAA\x55')
        if i >= 0 and len(buf) >= i + 15:
            o0, o1, cyc, flags = struct.unpack('<ffIB', buf[i + 2:i + 15])
            return np.array([o0, o1], np.float32), cyc, flags
    raise TimeoutError(f"no reply (got {len(buf)} bytes: {buf[:32].hex()})")


def evaluate_dataset(args, model):
    d = np.load(args.dataset)
    x, y = d['x'][:args.n], d['y'][:args.n]
    rows, hw_ok, ref_ok, agree, worst, cyc_set = [], 0, 0, 0, 0, set()
    t0 = time.time()
    with serial.Serial(args.port, args.baud, timeout=0.1) as ser:
        time.sleep(0.2)
        for i in range(len(x)):
            xi = x[i:i+1]
            hw, cyc, flags = board_infer(ser, xi)
            with torch.no_grad():
                ref = model(torch.from_numpy(xi)).numpy()[0]
            err = max(ulps(hw[0], ref[0]), ulps(hw[1], ref[1]))
            worst = max(worst, err); cyc_set.add(cyc)
            hp, rp = int(np.argmax(hw)), int(np.argmax(ref))
            hw_ok += hp == y[i]; ref_ok += rp == y[i]; agree += hp == rp
            rows.append((i, int(y[i]), hp, rp, float(hw[0]), float(hw[1]), float(ref[0]), float(ref[1]), err, cyc, flags))
            if (i + 1) % 25 == 0 or i + 1 == len(x):
                print(f"{i+1:4d}/{len(x)}  board acc {hw_ok/(i+1):.4f}  pytorch acc {ref_ok/(i+1):.4f}  "
                      f"agree {agree}/{i+1}  max err {worst} ulp  ({time.time()-t0:.0f} s)")
    n = len(x)
    print(f"\nBoard accuracy   : {hw_ok}/{n} = {hw_ok/n:.4f}")
    print(f"PyTorch accuracy : {ref_ok}/{n} = {ref_ok/n:.4f}  (same {n} images)")
    print(f"Prediction agreement board vs PyTorch: {agree}/{n}")
    print(f"Max output error : {worst} ulp;  cycle counts: {sorted(cyc_set)} "
          f"({min(cyc_set)/F_CLK*1e3:.3f} ms @10MHz)")
    if args.csv:
        with open(args.csv, 'w') as f:
            f.write('idx,label,board_pred,pytorch_pred,hw0,hw1,ref0,ref1,err_ulp,cycles,flags\n')
            for r in rows:
                f.write(','.join(str(v) for v in r) + '\n')
        print(f"per-image results -> {args.csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='/dev/ttyUSB0')      # PL2303: check `dmesg | grep pl2303`; FTDI ports reset the board
    ap.add_argument('--baud', type=int, default=57600)
    ap.add_argument('--weights', required=True, help='tosa_weights.pth from extract_tosa_weights.py')
    ap.add_argument('--random', type=int, default=3, help='number of random inputs besides ones/zeros')
    ap.add_argument('--tol-ulps', type=int, default=16)
    ap.add_argument('--dataset', help='test_set.npz from train_mynet.py: run board accuracy evaluation')
    ap.add_argument('--n', type=int, default=200, help='number of test images for --dataset')
    ap.add_argument('--csv', help='optional CSV file for per-image results')
    args = ap.parse_args()

    model = MyNet()
    model.load_state_dict(torch.load(args.weights))   # strict: keys must match
    model.eval()

    if args.dataset:
        return evaluate_dataset(args, model)

    rng = np.random.default_rng(42)
    tests = [('ones', np.ones((1, 1, 32, 32), np.float32)),
             ('zeros', np.zeros((1, 1, 32, 32), np.float32))]
    tests += [(f'randn{i}', rng.standard_normal((1, 1, 32, 32)).astype(np.float32))
              for i in range(args.random)]

    fails = 0
    cycles_seen = set()
    with serial.Serial(args.port, args.baud, timeout=0.1) as ser:
        time.sleep(0.2)
        for name, x in tests:
            hw, cyc, flags = board_infer(ser, x)
            with torch.no_grad():
                ref = model(torch.from_numpy(x)).numpy()[0]
            err = max(ulps(hw[0], ref[0]), ulps(hw[1], ref[1]))
            ok = err <= args.tol_ulps and flags == 0
            fails += not ok
            cycles_seen.add(cyc)
            print(f"{name:8s} hw={hw} ref={ref} max_err={err} ulp "
                  f"abs={np.abs(hw - ref).max():.2e} cycles={cyc} "
                  f"({cyc / F_CLK * 1e3:.3f} ms @10MHz) flags={flags} {'OK' if ok else 'MISMATCH'}")
    print(f"\n{len(tests) - fails}/{len(tests)} match within {args.tol_ulps} ulp; "
          f"cycle counts observed: {sorted(cycles_seen)}")


if __name__ == '__main__':
    main()
