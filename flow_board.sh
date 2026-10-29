#!/bin/bash
# flow_board.sh NAME weights.pth [noflash]
# Run in WSL (OSS CAD Suite) from tutorials/pytorch after flow_container.sh NAME.
# Wrapper simulation with expected outputs computed from the weights, then build + flash.
set -euo pipefail
NAME=$1; PTH=$2; FLASH=${3:-}
D=output/$NAME
PY=${PY:-$HOME/thesis-venv/bin/python3}
EXP=$($PY - "$PTH" <<'PYEOF'
import sys, struct, torch
sys.path.insert(0, 'hw')
from train_mynet import MyNet
m = MyNet(); m.load_state_dict(torch.load(sys.argv[1], map_location='cpu')); m.eval()
h = lambda v: '%08x' % struct.unpack('<I', struct.pack('<f', float(v)))[0]
with torch.no_grad():
    o = m(torch.ones(1, 1, 32, 32))[0]; z = m(torch.zeros(1, 1, 32, 32))[0]
print(f"+e_ones0={h(o[0])} +e_ones1={h(o[1])} +e_zeros0={h(z[0])} +e_zeros1={h(z[1])}")
PYEOF
)
echo "expected: $EXP"
cd $D/sim && rm -rf obj_dir
verilator --binary --timing -Wno-fatal -Wno-lint -Wno-style --top-module tb \
  ../../../hw/tb_nn_core.v ../../../hw/nn_core.v 06_verilog.v -o tbsim > verilator.log 2>&1 \
  || { tail -20 verilator.log; exit 1; }
./obj_dir/tbsim $EXP | tee tb.log | grep -E "^\[|OVERALL"
grep -q "OVERALL: PASS" tb.log || { echo "wrapper simulation FAILED - not building"; exit 1; }
cd - > /dev/null
KDIR=$D/sim TAG=$NAME bash hw/build_board.sh $FLASH
