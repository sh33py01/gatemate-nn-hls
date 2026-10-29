#!/usr/bin/env python3
"""jet_tb_data.py SRC OUTDIR -- turn SRC/jet_tb.c inputs + SRC/expected.txt (hls4ml emulation, raw)
into $readmemh files for tb_jet.v: jet_in.hex (4 x 64-bit words per jet), jet_out.hex (5 words)."""
import re, sys, os
src, out = sys.argv[1], sys.argv[2]
tb = open(os.path.join(src, 'jet_tb.c')).read()
rows = re.findall(r'\{([-0-9,\s]+)\}', tb.split('= {', 1)[1])
X = [[int(v) for v in r.split(',')] for r in rows]
E = [[int(v) for v in l.split()[2:]] for l in open(os.path.join(src, 'expected.txt'))]
assert len(X) == len(E) and all(len(x) == 16 for x in X)
with open(os.path.join(out, 'jet_in.hex'), 'w') as f:
    for x in X:
        for w in range(4):
            f.write('%016x\n' % sum((x[4 * w + j] & 0xFFFF) << (16 * j) for j in range(4)))
with open(os.path.join(out, 'jet_out.hex'), 'w') as f:
    for e in E:
        for v in e:
            f.write('%016x\n' % (v & (2 ** 64 - 1)))
print(f"{len(X)} jets -> {out}/jet_in.hex, jet_out.hex")
