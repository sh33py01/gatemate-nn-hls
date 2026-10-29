#!/bin/bash
# check_env.sh -- verify the WSL setup. Prints OK / MISSING for each component.
#   bash ~/soda-benchmarks/tutorials/pytorch/hw/setup/check_env.sh
OSS=$HOME/oss-cad-suite/bin
VENV=$HOME/thesis-venv/bin/python
SODA=$HOME/soda-benchmarks
fail=0
ok()  { printf '  \033[32mOK\033[0m       %s\n' "$*"; }
bad() { printf '  \033[31mMISSING\033[0m  %s\n' "$*"; fail=1; }
chk() { local name=$1; shift; if out=$("$@" 2>&1); then ok "$name: $(echo "$out" | head -1)"; else bad "$name"; fi; }

echo "FPGA tools (OSS CAD Suite)"
chk yosys            "$OSS/yosys" -V
chk nextpnr-himbaechel "$OSS/nextpnr-himbaechel" --version
chk gmpack           test -x "$OSS/gmpack"
chk openFPGALoader   "$OSS/openFPGALoader" --Version
chk verilator        "$OSS/verilator" --version

echo "Python venv"
chk python           "$VENV" --version
chk "torch/numpy/serial/sklearn/hls4ml" "$VENV" -c \
  "import torch, numpy, serial, sklearn, hls4ml; print('torch', torch.__version__, '| hls4ml', hls4ml.__version__)"

echo "SODA side"
chk soda-benchmarks  test -d "$SODA/tutorials/pytorch/hw"
chk ll_to_verilog_share.sh test -x "$SODA/scripts/ll_to_verilog_share.sh"
chk docker           docker version --format '{{.Server.Version}}'
chk "SODA image"     docker image inspect agostini01/soda-bookworm:v19.02 --format '{{.Id}}'

echo "Board (needs attach_board.ps1 on Windows first)"
if lsusb 2>/dev/null | grep -q '0403:6010'; then ok "GateMate board (FTDI 0403:6010)"; else bad "GateMate board not visible in WSL"; fi
if p=$(bash "$(dirname "$0")/find_uart.sh" 2>/dev/null); then ok "UART adapter: $p"; else bad "PL2303 UART adapter not visible"; fi

echo
[ $fail = 0 ] && echo "All checks passed." || echo "Some checks failed (board checks fail if the board is not attached - that is fine for now)."
