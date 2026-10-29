#!/usr/bin/env python3
"""
gen_int_c.py -- translate an hls4ml io_parallel MLP (dense + relu, Vivado backend) into plain
integer C for Bambu, bit-exact with hls4ml's ap_fixed arithmetic, with a configurable
number of parallel multipliers M (Bambu's equivalent of hls4ml's reuse factor).

Exact semantics reproduced (ap_fixed defaults AP_TRN / AP_WRAP, as used by hls4ml here):
  product  = data * weight              (full precision, frac = fd + fw)
  accum    = bias + sum(products)       (each term converted to accum_t: floor + wrap)
  dense out= (result_t) accum           (floor + wrap)
  relu     = x > 0 ? (out_t) x : 0      (floor + wrap)
Integer addition is exact in 64 bits, and wrapping is modular, so the summation order does
not matter: any M gives identical results.

Parallelism: M multiplications per inner-loop iteration. The M weights and M activations
used together are packed into one wide memory word (like hls4ml's ARRAY_RESHAPE), so each
layer needs one memory (two or more lanes of 64 bits for M >= 8) instead of M small ones;
dense + relu are fused so only 16-bit activations are stored. constraints.xml limits Bambu to
M multipliers. hls4ml reuse factor equivalent per layer: RF = n_in*n_out / M.

Usage: python3 gen_int_c.py output/jet/hls4ml_prj output/jet/int_M4 --mult 4 \
           --tb-from output/jet/bambu_src
"""
import argparse, os, re, shutil

ap = argparse.ArgumentParser()
ap.add_argument('prj'); ap.add_argument('out')
ap.add_argument('--mult', type=int, default=1, help='parallel multipliers M')
ap.add_argument('--tb-from', help='dir with jet_tb.c + expected.txt (from prep_hls4ml_for_bambu.py)')
args = ap.parse_args()
M = args.mult
fw = os.path.join(args.prj, 'firmware')
defs = open(os.path.join(fw, 'defines.h')).read()
params = open(os.path.join(fw, 'parameters.h')).read()
top = open(os.path.join(fw, 'myproject.cpp')).read()

def fx(t):
    m = re.search(r'typedef ap_fixed<(\d+),\s*(-?\d+)\s*(,[^>]*)?> ' + re.escape(t) + r';', defs)
    assert m, f"type {t} not found / not ap_fixed"
    assert not m.group(3) or m.group(3).replace(' ', '') in (',AP_TRN', ',AP_TRN,AP_WRAP', ',AP_TRN,AP_WRAP,0'), \
        f"{t}: only default AP_TRN/AP_WRAP supported, got {m.group(0)}"
    w, i = int(m.group(1)), int(m.group(2))
    return w, w - i          # width, fractional bits

def cfg(name, key):
    m = re.search(r'struct ' + name + r' : .*?\{(.*?)\};', params, re.S)
    v = re.search(r'typedef (\w+) ' + key + ';', m.group(1)) if key.endswith('_t') else \
        re.search(r'static const unsigned ' + key + r' = (\d+);', m.group(1))
    return v.group(1) if key.endswith('_t') else int(v.group(1))

def values(wname):
    s = open(os.path.join(fw, 'weights', wname + '.h')).read()
    body = re.search(r'#else\s*\n[^{]*\{(.*?)\};', s, re.S) or re.search(r'=\s*\{(.*?)\};', s, re.S)
    return [float(v) for v in body.group(1).replace('\n', ' ').split(',') if v.strip()]

def quant(v, w, f):                       # double -> ap_fixed raw: floor, then wrap
    import math
    r = math.floor(v * 2.0 ** f)
    r = ((r + 2 ** (w - 1)) % 2 ** w) - 2 ** (w - 1)
    return r

layers = []
for m in re.finditer(r'nnet::(dense|relu)<(\w+), (\w+), (\w+)>\((\w+), (\w+)(?:, (\w+), (\w+))?\);', top):
    kind, ti, to, c, a, b, wn, bn = m.groups()
    layers.append(dict(kind=kind, tin=ti, tout=to, cfg=c, src=a, dst=b, w=wn, b=bn))
assert layers and all(l['kind'] in ('dense', 'relu') for l in layers), "only dense/relu supported"
sig = re.search(r'void myproject\(\s*input_t (\w+)\[(\d+)\],\s*result_t (\w+)\[(\d+)\]', top)
in_name, n_in0, out_name, n_out_last = sig.group(1), int(sig.group(2)), sig.group(3), int(sig.group(4))

def ctype(w):
    return 'short' if w <= 16 else ('int' if w <= 32 else 'long long')

def conv(expr, f_from, w_to, f_to):          # C expression converting raw between fixed formats
    s = f_from - f_to
    e = f'(({expr}) >> {s})' if s > 0 else (f'(({expr}) << {-s})' if s < 0 else f'({expr})')
    return f'wrap({e}, {w_to})'

code, decls = [], []
L = min(M, 4)                     # 16-bit values per memory word (64-bit max: i386 clang has no __int128)
G = M // L                        # memory words read in parallel (lanes)
assert M in (1, 2, 4, 8, 16), "M must be a power of two <= 16"
WT = {1: 'unsigned short', 2: 'unsigned int', 4: 'unsigned long long'}[L]
lane = lambda word, j: f'(short)({word} >> {16 * j})' if j else f'(short)({word})'

# fuse dense + following relu (hls4ml: relu(cast(acc)) -> layer type)
steps, i = [], 0
while i < len(layers):
    L0 = layers[i]
    if L0['kind'] == 'dense' and i + 1 < len(layers) and layers[i + 1]['kind'] == 'relu' \
            and layers[i + 1]['src'] == L0['dst']:
        steps.append((L0, layers[i + 1])); i += 2
    else:
        assert L0['kind'] == 'dense', "relu not preceded by dense is not supported"
        steps.append((L0, None)); i += 1

act = []
def act_decl(name, n):
    assert n % M == 0, f"{name}: {n} not divisible by M={M}"
    for g in range(G):
        act.append(f'  {WT} {name}_p{g}[{n // M}];')

n_in0 = n_in0
act_decl(in_name, n_in0)
copy_in = (f'  for (int c = 0; c < {n_in0 // M}; c++) {{\n' +
           '\n'.join(f'    {in_name}_p{g}[c] = ' + ' | '.join(
               f'(({WT})(unsigned short)in_raw[c * {M} + {g * L + j}] << {16 * j})' for j in range(L)) + ';'
               for g in range(G)) + '\n  }')
src_name = in_name
for D, R in steps:
    n_in, n_out = cfg(D['cfg'], 'n_in'), cfg(D['cfg'], 'n_out')
    assert D['src'] == src_name and n_in % M == 0
    wd, fd = fx(D['tin']); ww, fwt = fx(cfg(D['cfg'], 'weight_t'))
    wb, fb = fx(cfg(D['cfg'], 'bias_t')); wa, fa = fx(cfg(D['cfg'], 'accum_t'))
    wr, fr = fx(D['tout'])
    Wv, Bv = values(D['w']), values(D['b'])
    assert len(Wv) == n_in * n_out and len(Bv) == n_out
    q = lambda ii, jj: quant(Wv[ii * n_out + jj], ww, fwt) & 0xFFFF        # hls4ml index ii*n_out+jj
    for g in range(G):                    # word [o][c], lane j  <->  input ii = c*M + g*L + j
        rows = []
        for jj in range(n_out):
            rows.append(','.join(str(sum(q(c * M + g * L + j, jj) << (16 * j) for j in range(L)))
                                 + ('ULL' if L == 4 else 'U') for c in range(n_in // M)))
        decls.append(f'static const {WT} {D["w"]}_p{g}[{n_out}][{n_in // M}] = {{\n  {{' +
                     '},\n  {'.join(rows) + '}};')
    decls.append(f'static const long long {D["b"]}_acc[{n_out}] = {{' +
                 ','.join(str(quant(v, wb, fb) * 2 ** (fa - fb) if fa >= fb else quant(v, wb, fb) >> (fb - fa))
                          for v in Bv) + '};')
    pf = fd + fwt
    prods = []
    for g in range(G):
        for j in range(L):
            p = f'(long long)(int){lane("xw" + str(g), j)} * (int){lane("ww" + str(g), j)}'
            prods.append(conv(p, pf, wa, fa) if pf != fa else p)
    NOUNROLL = '#pragma clang loop unroll(disable)'
    def mac(o_expr, ind):
        loads = ' '.join(f'{WT} xw{g} = {src_name}_p{g}[c], ww{g} = {D["w"]}_p{g}[{o_expr}][c];' for g in range(G))
        return (f'{ind}long long acc = {D["b"]}_acc[{o_expr}];\n'
                f'{ind}{NOUNROLL}\n'
                f'{ind}for (int c = 0; c < {n_in // M}; c++) {{\n'
                f'{ind}  {loads}\n'
                f'{ind}  acc += {" + ".join(prods)};\n'
                f'{ind}}}\n'
                f'{ind}acc = wrap(acc, {wa});\n')
    code.append(f'  /* {D["cfg"]}: dense {n_in}->{n_out}{" + relu" if R else ""}, {M} multiplication(s)/cycle '
                f'(hls4ml-equivalent reuse factor {n_in * n_out // M}) */')
    if R:
        wo, fo = fx(R['tout'])
        act_decl(R['dst'], n_out)
        lanes = '\n'.join(
            f'      {"if" if g == 0 else "else if"} (k / {L} == {g}) t{g} |= ({WT})(unsigned short)a << (16 * (k % {L}));'
            for g in range(G)) if G > 1 else f'      t0 |= ({WT})(unsigned short)a << (16 * k);'
        code.append(f'  for (int ob = 0; ob < {n_out // M}; ob++) {{\n' +
                    '\n'.join(f'    {WT} t{g} = 0;' for g in range(G)) + '\n' +
                    f'    {NOUNROLL}\n'
                    f'    for (int k = 0; k < {M}; k++) {{\n' + mac(f'ob * {M} + k', '      ') +
                    f'      long long v = {conv("acc", fa, wr, fr)};\n'
                    f'      long long a = v > 0 ? {conv("v", fr, wo, fo)} : 0;\n' + lanes + '\n    }\n' +
                    '\n'.join(f'    {R["dst"]}_p{g}[ob] = t{g};' for g in range(G)) + '\n  }')
        src_name = R['dst']
    else:
        assert D['dst'] == out_name, "a dense layer without relu must be the output layer"
        code.append(f'  {NOUNROLL}\n  for (int o = 0; o < {n_out}; o++) {{\n' + mac('o', '    ') +
                    f'    out_raw[o] = {conv("acc", fa, wr, fr)};\n  }}')
copy_out = ''
src = f'''/* generated by gen_int_c.py from {os.path.abspath(args.prj)}  --  M = {M} parallel multipliers
   bit-exact integer version of the hls4ml model (ap_fixed raw values, AP_TRN/AP_WRAP) */
static inline long long wrap(long long v, int w) {{
  return (long long)((unsigned long long)v << (64 - w)) >> (64 - w);
}}
{chr(10).join(decls)}

void jet_top(const short *in_raw, long long *out_raw) {{
{chr(10).join(act)}
{copy_in}
{chr(10).join(code)}
{copy_out}
}}
'''
os.makedirs(args.out, exist_ok=True)
open(os.path.join(args.out, 'jet_top.c'), 'w').write(src)
open(os.path.join(args.out, 'constraints.xml'), 'w').write(
    f'<?xml version="1.0"?>\n<constraints>\n  <HLS_constraints>\n'
    f'    <tech_constraints fu_name="mult_expr_FU" fu_library="STD_FU" n="{M}"/>\n'
    f'  </HLS_constraints>\n</constraints>\n')
if args.tb_from:
    for f in ('jet_tb.c', 'expected.txt'):
        shutil.copy(os.path.join(args.tb_from, f), args.out)
print(f"jet_top.c ({len(layers)} layers, M={M}), constraints.xml -> {args.out}")
