#!/bin/bash
# [MODEL=jet NOUT=5] flow_container.sh NAME [weights.pth]
#   MODEL: mynet (default) | jet ; NOUT: outputs printed by the simulation (default 2)
# Run in the DEVCONTAINER from tutorials/pytorch. Full SODA/Bambu flow for MyNet:
#   export (if output/NAME/01_tosa.mlir is missing) -> linalg -> soda.launch/copy edits ->
#   soda-opt -> transpose fold -> mlir-opt -> LLVM IR -> Bambu (shared units, ALL_BRAM) + simulation
# Regression test of this script on the known-good model:
#   mkdir -p output/regress && cp output/01_tosa.mlir output/regress/ && bash hw/flow_container.sh regress
set -euo pipefail
NAME=$1; PTH=${2:-}; MODEL=${MODEL:-mynet}; NOUT=${NOUT:-2}
D=output/$NAME; H=hw; S=../../scripts
mkdir -p $D
if [ ! -f $D/01_tosa.mlir ]; then
  [ -n "$PTH" ] || { echo "no $D/01_tosa.mlir and no weights given"; exit 1; }
  python3 $H/export_tosa.py "$PTH" $D/01_tosa.mlir --model $MODEL | tee $D/export.log
fi
echo "== tosa -> linalg"
$S/tosa_to_linalg.sh $D/01_tosa.mlir $D/02_linalg.mlir
[ -s $D/02_linalg.mlir ] || { echo "tosa_to_linalg.sh did not write $D/02_linalg.mlir - show me: head -40 $S/tosa_to_linalg.sh"; exit 1; }
echo "== soda.launch + copy rewrite"
python3 $H/prepare_soda.py $D/02_linalg.mlir $D/03_searched-edited.mlir
echo "== soda-opt"
soda-opt -soda-outline-bambu-code -soda-extract-arguments-to-c-testbench=using-bare-ptr \
  -soda-generate-bambu-accelcode=no-aa $D/03_searched-edited.mlir -o $D/04_pre_expand.mlir
# print the kernel outputs in the (regenerated) testbench
python3 - "$NOUT" <<'PY'
import sys
n = int(sys.argv[1]); p = 'forward_kernel_testbench.c'
s = ''.join(l for l in open(p) if 'RTL_OUT' not in l)
call = 'forward_kernel((void*) P0, (void*) P1);'
assert call in s, "kernel call not found in forward_kernel_testbench.c"
fmt = ' '.join(['%.9g'] * n); args = ', '.join(f'((float*)P1)[{i}]' for i in range(n))
s = s.replace(call, call + '\n      printf("RTL_OUT ' + fmt + '\\n", ' + args + ');', 1)
open(p, 'w').write(s)
print("testbench prints", n, "outputs")
PY
echo "== fold weight transposes"
python3 $H/fold_weight_transpose.py $D/04_pre_expand.mlir $D/04_nt.mlir
echo "== lower to LLVM"
mlir-opt \
  --promote-buffers-to-stack='max-alloc-size-in-bytes=65536 max-rank-of-allocated-memref=4' \
  -expand-strided-metadata -memref-expand -convert-linalg-to-loops -lower-affine -arith-expand \
  -convert-math-to-llvm -convert-scf-to-cf -finalize-memref-to-llvm \
  --convert-func-to-llvm='use-bare-ptr-memref-call-conv=1' --convert-arith-to-llvm \
  --convert-cf-to-llvm --reconcile-unrealized-casts $D/04_nt.mlir -o $D/04_nt_llvm.mlir
mlir-translate --mlir-to-llvmir $D/04_nt_llvm.mlir -o $D/05_nt.ll
grep -q "define void @forward_kernel(ptr %0, ptr %1)" $D/05_nt.ll || { echo "unexpected kernel signature"; grep "define.*forward_kernel" $D/05_nt.ll; exit 1; }
[ "$(grep -c 'malloc\|stacksave' $D/05_nt.ll || true)" = "0" ] || { echo "malloc/stacksave left in $D/05_nt.ll"; exit 1; }
echo "== Bambu + simulation"
BAMBU_DEVICE=nangate45 BAMBU_MEMPOLICY=ALL_BRAM BAMBU_RUN_SIMULATION=true \
  $S/ll_to_verilog_share.sh $D/05_nt.ll $D/sim/06_verilog.v > $D/bambu_run.log 2>&1 \
  || { tail -30 $D/bambu_run.log; exit 1; }
echo "== result"
grep -E "RTL_OUT|Average execution" $D/sim/bambu-log | sort -u
[ -f $D/export.log ] && grep "expected RTL_OUT" $D/export.log || true
echo "Next (WSL): bash hw/flow_board.sh $NAME ${PTH:-output/tosa_weights.pth}"
