#!/usr/bin/env python3
"""
prepare_soda.py -- automate the two manual edits that turn 02_linalg.mlir into
03_searched-edited.mlir (thesis notes §3.2 and §3.3):

 1. every strided memref.copy %src, %dst : T1 to T2  ->  identity linalg.generic copy
    (memref.copy lowers to a runtime call with llvm.stacksave, which Bambu rejects)
 2. wrap the body of @forward in  soda.launch { ... soda.terminator }
    (gives soda-opt a single kernel to outline)

Usage: python3 prepare_soda.py 02_linalg.mlir 03_searched-edited.mlir [--func forward]
Check against a hand-edited file:  diff <(python3 prepare_soda.py in.mlir /dev/stdout) old.mlir
"""
import re
import sys

args = [a for a in sys.argv[1:] if not a.startswith('--')]
func = 'forward'
if '--func' in sys.argv:
    func = sys.argv[sys.argv.index('--func') + 1]
    args.remove(func)
src, dst = args
lines = open(src).read().split('\n')

# ---- 1. memref.copy -> linalg.generic ----
copy_re = re.compile(r'^(\s*)memref\.copy (%[\w.]+), (%[\w.]+) : (memref<([0-9x]+)x(f32|f16|i32|i8)>) to (memref<.*>)\s*$')
need_maps, out = set(), []
for ln in lines:
    m = copy_re.match(ln)
    if not m or 'strided<' not in (m.group(4) + m.group(7)):
        out.append(ln); continue      # contiguous copies lower to memcpy, which Bambu accepts
    ind, s, d, ts, dims, ety, td = m.groups()
    rank = len(dims.split('x'))
    need_maps.add(rank)
    it = ', '.join(['"parallel"'] * rank)
    out += [f'{ind}linalg.generic {{indexing_maps = [#prep_id{rank}, #prep_id{rank}], iterator_types = [{it}]}} '
            f'ins({s} : {ts}) outs({d} : {td}) {{',
            f'{ind}^bb0(%prep_in: {ety}, %prep_out: {ety}):',
            f'{ind}  linalg.yield %prep_in : {ety}',
            f'{ind}}}']
    print(f"memref.copy {s} -> {d} replaced by linalg.generic (rank {rank})", file=sys.stderr)
lines = out

# ---- 2. soda.launch around the function body ----
start = next(i for i, l in enumerate(lines) if re.match(r'\s*func\.func @' + re.escape(func) + r'\(', l))
assert lines[start].rstrip().endswith('{'), "func.func signature spans several lines; adjust script"
ret = next(i for i in range(start + 1, len(lines)) if re.match(r'^\s*return\b', lines[i]))
assert re.match(r'^\s*return\s*$', lines[ret]), \
    f"@{func} returns a value ({lines[ret].strip()}); soda.launch needs a buffer-semantics function"
assert 'soda.launch' not in '\n'.join(lines), "input already contains soda.launch"
ind = re.match(r'^(\s*)', lines[ret]).group(1)
lines = lines[:ret] + [f'{ind}soda.terminator', f'{ind}}}'] + lines[ret:]
lines = lines[:start + 1] + [f'{ind}soda.launch {{'] + lines[start + 1:]

# ---- affine map aliases for the copies ----
if need_maps:
    pos = next(i for i, l in enumerate(lines) if l.startswith('module'))
    for r in sorted(need_maps):
        d = ', '.join(f'd{k}' for k in range(r))
        lines.insert(pos, f'#prep_id{r} = affine_map<({d}) -> ({d})>')

open(dst, 'w').write('\n'.join(lines))
print(f"soda.launch inserted after line {start + 1}, terminator before return -> {dst}", file=sys.stderr)
