#!/usr/bin/env python3
"""
train_mynet.py -- train the exact MyNet that runs on the GateMate, on MNIST as a
2-class task, and export everything the hardware evaluation needs.

Task (default): digit 0-4 -> class 0, digit 5-9 -> class 1.   (--task even_odd for parity)
Input: 28x28 MNIST zero-padded to 32x32 (LeNet convention), normalised.

Outputs (in --out, default output/trained):
  mynet_trained.pth        state_dict for the hardware network
  test_set.npz             x (N,1,32,32) float32 preprocessed, y (N,) labels, digit (N,)
  train_report.txt         accuracy per epoch, final float32 test accuracy

Usage:
  python3 train_mynet.py                 # downloads MNIST to data/mnist on first run
  python3 train_mynet.py --epochs 15 --task even_odd
"""
import argparse, gzip, os, struct, urllib.request
import numpy as np
import torch
import torch.nn as nn

MIRRORS = ["https://ossci-datasets.s3.amazonaws.com/mnist/",
           "https://storage.googleapis.com/cvdf-datasets/mnist/"]
FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
         "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz"}
MEAN, STD = 0.1307, 0.3081


class MyNet(nn.Module):
    """Identical to the network in the SODA flow. forward() ends in softmax (exported);
    logits() is used for training with cross-entropy."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 5, stride=2, padding=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1, self.fc2, self.fc3 = nn.Linear(256, 8), nn.Linear(8, 4), nn.Linear(4, 2)
        self.softmax = nn.Softmax(dim=1)

    def logits(self, x):
        x = self.flatten(self.relu(self.conv(x)))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    def forward(self, x):
        return self.softmax(self.logits(x))


def fetch(datadir):
    os.makedirs(datadir, exist_ok=True)
    out = {}
    for key, name in FILES.items():
        path = os.path.join(datadir, name)
        if not os.path.exists(path):
            for m in MIRRORS:
                try:
                    print(f"downloading {m}{name}")
                    urllib.request.urlretrieve(m + name, path)
                    break
                except Exception as e:
                    print(f"  failed: {e}")
            else:
                raise SystemExit(f"could not download {name}; put it in {datadir} manually")
        with gzip.open(path, "rb") as f:
            data = f.read()
        if key.endswith("_x"):
            n, r, c = struct.unpack(">III", data[4:16])
            out[key] = np.frombuffer(data, np.uint8, offset=16).reshape(n, r, c)
        else:
            out[key] = np.frombuffer(data, np.uint8, offset=8)
    return out


def preprocess(img_u8):
    x = img_u8.astype(np.float32) / 255.0
    x = np.pad(x, ((0, 0), (2, 2), (2, 2)))                 # 28x28 -> 32x32
    x = (x - MEAN) / STD
    return x[:, None, :, :].astype(np.float32)             # (N,1,32,32)


def relabel(digits, task):
    return (digits % 2).astype(np.int64) if task == "even_odd" else (digits >= 5).astype(np.int64)


def accuracy(model, x, y, bs=2000):
    model.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            correct += (model(torch.from_numpy(x[i:i+bs])).argmax(1).numpy() == y[i:i+bs]).sum()
    return correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mnist")
    ap.add_argument("--out", default="output/trained")
    ap.add_argument("--task", choices=["low_high", "even_odd"], default="low_high")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true", help="random data, only to test the script")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.synthetic:
        rng = np.random.default_rng(0)
        d = {"train_x": rng.integers(0, 256, (2000, 28, 28), dtype=np.uint8),
             "train_y": rng.integers(0, 10, 2000).astype(np.uint8),
             "test_x": rng.integers(0, 256, (500, 28, 28), dtype=np.uint8),
             "test_y": rng.integers(0, 10, 500).astype(np.uint8)}
    else:
        d = fetch(args.data)
    xtr, xte = preprocess(d["train_x"]), preprocess(d["test_x"])
    ytr, yte = relabel(d["train_y"], args.task), relabel(d["test_y"], args.task)

    model = MyNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.CrossEntropyLoss()
    log = [f"task={args.task} epochs={args.epochs} lr={args.lr} seed={args.seed}"]
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(xtr))
        for i in range(0, len(perm), 128):
            idx = perm[i:i+128]
            loss = lossf(model.logits(torch.from_numpy(xtr[idx])), torch.from_numpy(ytr[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        line = f"epoch {ep+1:2d}: loss {loss.item():.4f}  train acc {accuracy(model, xtr, ytr):.4f}  test acc {accuracy(model, xte, yte):.4f}"
        print(line); log.append(line)

    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out, "mynet_trained.pth"))
    np.savez_compressed(os.path.join(args.out, "test_set.npz"), x=xte, y=yte, digit=d["test_y"])
    final = f"FINAL float32 test accuracy: {accuracy(model, xte, yte):.4f} on {len(xte)} images"
    print(final); log.append(final)
    open(os.path.join(args.out, "train_report.txt"), "w").write("\n".join(log) + "\n")
    print(f"saved {args.out}/mynet_trained.pth, test_set.npz, train_report.txt")


if __name__ == "__main__":
    main()
