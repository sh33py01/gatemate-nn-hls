#!/usr/bin/env python3
"""
fold_weight_transpose.py
Remove the runtime weight transposes that torch-mlir emits for nn.Linear.

Pattern (memref level, one per Linear layer):
    %T = memref.alloc() ... : memref<KxNxf32>
    linalg.transpose ins(%W : memref<NxKxf32>) outs(%T : memref<KxNxf32>) permutation = [1, 0]
    %E = memref.expand_shape %T [[0, 1], [2]] output_shape [1, K, N] : ... into memref<1xKxNxf32>
    linalg.batch_matmul ins(%A, %E : memref<1x1xKxf32>, memref<1xKxNxf32>) outs(%O : memref<1x1xNxf32>)

Rewritten to read the original weight %W directly, with no copy buffer:
    linalg.generic {maps (b,m,n,k): A(b,m,k), W(n,k), O(b,m,n); k = reduction}
      body: O = O + A*W            (same mulf/addf order as batch_matmul -> bit-exact)

Usage: python3 fold_weight_transpose.py in.mlir out.mlir
"""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()

MAPS = ("#fwt_a = affine_map<(b, m, n, k) -> (b, m, k)>\n"
        "#fwt_w = affine_map<(b, m, n, k) -> (n, k)>\n"
        "#fwt_o = affine_map<(b, m, n, k) -> (b, m, n)>\n")

tr_re = re.compile(
    r'^(?P<ind>[ \t]*)linalg\.transpose ins\((?P<W>%[\w.]+) : memref<(?P<N>\d+)x(?P<K>\d+)xf32>\) '
    r'outs\((?P<T>%[\w.]+) : memref<(?P=K)x(?P=N)xf32>\) permutation = \[1, 0\][ \t]*\n', re.M)

count = 0
for m in list(tr_re.finditer(text)):
    W, T, N, K = m['W'], m['T'], m['N'], m['K']
    i = count
    # 1. alloc of the transpose buffer
    alloc_re = re.compile(r'^[ \t]*' + re.escape(T) + r' = memref\.alloc\(\)[^\n]*\n', re.M)
    # 2. expand_shape of it
    exp_re = re.compile(r'^[ \t]*(?P<E>%[\w.]+) = memref\.expand_shape ' + re.escape(T) +
                        r' [^\n]*\n', re.M)
    em = exp_re.search(text)
    assert alloc_re.search(text), f"alloc of {T} not found"
    assert em, f"expand_shape of {T} not found"
    E = em['E']
    # 3. the batch_matmul consuming it
    mm_re = re.compile(
        r'^(?P<ind>[ \t]*)linalg\.batch_matmul ins\((?P<A>%[\w.]+), ' + re.escape(E) +
        r' : (?P<At>memref<1x1x' + K + r'xf32>), memref<1x' + K + 'x' + N + r'xf32>\) '
        r'outs\((?P<O>%[\w.]+) : (?P<Ot>memref<1x1x' + N + r'xf32>)\)[ \t]*\n', re.M)
    mm = mm_re.search(text)
    assert mm, f"batch_matmul using {E} not found (K={K}, N={N})"
    # safety: T and E must have no other users
    assert len(re.findall(re.escape(T) + r'\b', text)) == 3, f"{T} has extra users"
    assert len(re.findall(re.escape(E) + r'\b', text)) == 2, f"{E} has extra users"

    ind = mm['ind']
    generic = (
        f"{ind}linalg.generic {{indexing_maps = [#fwt_a, #fwt_w, #fwt_o], "
        f"iterator_types = [\"parallel\", \"parallel\", \"parallel\", \"reduction\"]}} "
        f"ins({mm['A']}, {W} : {mm['At']}, memref<{N}x{K}xf32>) outs({mm['O']} : {mm['Ot']}) {{\n"
        f"{ind}^bb0(%fwt{i}_a: f32, %fwt{i}_w: f32, %fwt{i}_acc: f32):\n"
        f"{ind}  %fwt{i}_p = arith.mulf %fwt{i}_a, %fwt{i}_w : f32\n"
        f"{ind}  %fwt{i}_s = arith.addf %fwt{i}_acc, %fwt{i}_p : f32\n"
        f"{ind}  linalg.yield %fwt{i}_s : f32\n"
        f"{ind}}}\n")
    text = mm_re.sub(lambda _: generic, text, count=1)
    text = exp_re.sub('', text, count=1)
    text = tr_re.sub('', text, count=1)
    text = alloc_re.sub('', text, count=1)
    print(f"folded transpose of {W} ({N}x{K}): removed {T}, {E}; batch_matmul -> generic")
    count += 1

assert count > 0, "no weight transpose found"
assert 'linalg.transpose' not in text, "a linalg.transpose is left over"
# affine aliases must precede their use: put them before the first 'module'
pos = text.find('module')
text = text[:pos] + MAPS + text[pos:]
open(dst, 'w').write(text)
print(f"{count} transposes folded -> {dst}")
