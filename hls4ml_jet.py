#!/usr/bin/env python3
"""
hls4ml_jet.py -- the hls4ml tutorial jet tagger (Duarte et al. 2018): 16 -> 64 -> 32 -> 32 -> 5,
ReLU, softmax. Trains it in PyTorch on the hls4ml LHC jet dataset (OpenML), converts it
with hls4ml (Vivado backend, io_parallel, fixed point) and measures the fixed-point
accuracy with hls4ml's bit-accurate C++ emulation.

Outputs in --out (default output/jet):
  jet_model.pth, test_set.npz (x float32 scaled, y labels), train_report.txt
  hls4ml_prj/   hls4ml project (firmware/myproject.cpp = input for Bambu)
  summary.txt   float32 vs fixed-point accuracy

Usage:
  python3 hls4ml_jet.py                          # train + convert, ap_fixed<16,6>, reuse 1
  python3 hls4ml_jet.py --skip-train --precision 'ap_fixed<8,4>' --reuse 4 --prj hls4ml_prj_8b_r4
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn


class JetTagger(nn.Module):
    def __init__(self, softmax=True):
        super().__init__()
        self.fc1, self.relu1 = nn.Linear(16, 64), nn.ReLU()
        self.fc2, self.relu2 = nn.Linear(64, 32), nn.ReLU()
        self.fc3, self.relu3 = nn.Linear(32, 32), nn.ReLU()
        self.fc4 = nn.Linear(32, 5)
        if softmax:                       # hls4ml has no converter for nn.Identity, so omit the module
            self.softmax = nn.Softmax(dim=1)

    def logits(self, x):
        x = self.relu1(self.fc1(x)); x = self.relu2(self.fc2(x)); x = self.relu3(self.fc3(x))
        return self.fc4(x)

    def forward(self, x):
        z = self.logits(x)
        return self.softmax(z) if hasattr(self, 'softmax') else z


def load_data(cache):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d['X'], d['y'], list(d['classes'])
    from sklearn.datasets import fetch_openml
    print("downloading hls4ml_lhc_jets_hlf from OpenML (~830k jets)...")
    data = fetch_openml('hls4ml_lhc_jets_hlf', as_frame=False, parser='auto')
    X = data['data'].astype(np.float32)
    classes = sorted(set(data['target']))
    y = np.array([classes.index(t) for t in data['target']], dtype=np.int64)
    np.savez_compressed(cache, X=X, y=y, classes=np.array(classes))
    return X, y, classes


def accuracy(pred, y):
    return float((np.argmax(pred, 1) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='output/jet')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--precision', default='ap_fixed<16,6>')
    ap.add_argument('--reuse', type=int, default=1)
    ap.add_argument('--prj', default='hls4ml_prj')
    ap.add_argument('--no-softmax', action='store_true', help='leave softmax out of the hardware model')
    ap.add_argument('--n-eval', type=int, default=20000, help='test jets for the fixed-point emulation')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)

    # ---- data: same split/scaling as the hls4ml tutorial (80/20, StandardScaler) ----
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y, classes = load_data(os.path.join(args.out, 'jets_raw.npz'))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)

    model = JetTagger(softmax=not args.no_softmax)
    pth = os.path.join(args.out, 'jet_model.pth')
    if args.skip_train:
        model.load_state_dict(torch.load(pth, map_location='cpu'))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss(); log = []
        for ep in range(args.epochs):
            model.train(); perm = np.random.permutation(len(Xtr))
            for i in range(0, len(perm), 1024):
                idx = perm[i:i+1024]
                loss = lossf(model.logits(torch.from_numpy(Xtr[idx])), torch.from_numpy(ytr[idx]))
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                acc = accuracy(model(torch.from_numpy(Xte)).numpy(), yte)
            line = f"epoch {ep+1:2d}: loss {loss.item():.4f}  test acc {acc:.4f}"; print(line); log.append(line)
        torch.save(model.state_dict(), pth)
        open(os.path.join(args.out, 'train_report.txt'), 'w').write('\n'.join(log) + '\n')
    model.eval()
    np.savez_compressed(os.path.join(args.out, 'test_set.npz'), x=Xte, y=yte, classes=np.array(classes),
                        scaler_mean=sc.mean_, scaler_scale=sc.scale_)

    # ---- hls4ml conversion ----
    import hls4ml
    config = hls4ml.utils.config_from_pytorch_model(
        model, input_shape=(16,), granularity='name', backend='Vivado',
        default_precision=args.precision, default_reuse_factor=args.reuse)
    prj = os.path.join(args.out, args.prj)
    ne = min(args.n_eval, len(Xte))
    xtb, ytb = Xte[:ne], yte[:ne]
    with torch.no_grad():
        ref = model(torch.from_numpy(xtb)).numpy()
    np.savetxt(os.path.join(args.out, 'tb_input.dat'), xtb[:100], fmt='%.8f')
    np.savetxt(os.path.join(args.out, 'tb_output.dat'), ref[:100], fmt='%.8f')
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model, output_dir=prj, project_name='myproject', backend='Vivado', hls_config=config,
        io_type='io_parallel', part='xcu250-figd2104-2L-e',
        input_data_tb=os.path.join(args.out, 'tb_input.dat'),
        output_data_tb=os.path.join(args.out, 'tb_output.dat'))
    hls_model.write()
    print(f"hls4ml project written to {prj} (firmware/myproject.cpp)")

    # ---- bit-accurate fixed-point emulation (needs only g++) ----
    hls_model.compile()
    hw = hls_model.predict(np.ascontiguousarray(xtb)).reshape(ne, -1)
    acc_f, acc_q = accuracy(ref, ytb), accuracy(hw, ytb)
    agree = float((np.argmax(hw, 1) == np.argmax(ref, 1)).mean())
    s = (f"model: 16-64-32-32-5{' (no softmax)' if args.no_softmax else ''}, precision {args.precision}, reuse {args.reuse}\n"
         f"classes: {classes}\n"
         f"float32 accuracy  ({ne} test jets): {acc_f:.4f}\n"
         f"hls4ml {args.precision} accuracy: {acc_q:.4f}\n"
         f"prediction agreement float vs fixed: {agree:.4f}\n")
    print(s); open(os.path.join(args.out, f'summary_{args.prj}.txt'), 'w').write(s)
    # inputs + bit-accurate hls4ml outputs, the reference for the Bambu/GateMate version
    np.savez_compressed(os.path.join(args.out, f'{args.prj}_emulation.npz'), x=xtb, y=ytb, out=hw, ref_float=ref)


if __name__ == '__main__':
    main()
