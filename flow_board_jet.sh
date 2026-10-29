#!/bin/bash
# [PREFIX=pk] flow_board_jet.sh M [noflash]   (WSL, OSS CAD Suite, from tutorials/pytorch)
# uses output/jet/${PREFIX}_M$M (default PREFIX=int)
# Jet tagger with M multipliers: wrapper simulation (real Bambu kernel inside nn_core, checked
# against hls4ml's bit-accurate emulation), then GateMate build (+ flash unless noflash).
set -euo pipefail
M=$1; FLASH=${2:-}
P=${PREFIX:-int}; SRC=output/jet/${P}_M$M; D=$SRC/bambu; R=../../../..
[ -f $D/06_verilog.v ] || { echo "$D/06_verilog.v missing: run run_bambu_int.sh $SRC in the container first"; exit 1; }
cd $D
python3 $R/hw/jet_tb_data.py .. .
NJ=$(wc -l < ../expected.txt)
rm -rf obj_dir
verilator --binary --timing -Wno-fatal -Wno-lint -Wno-style --top-module tb -GNJ=$NJ \
  $R/hw/tb_jet.v $R/hw/nn_core.v $R/hw/jet_shim.v 06_verilog.v -o tbsim > verilator.log 2>&1 || { tail -20 verilator.log; exit 1; }
./obj_dir/tbsim | tee tb.log | tail -3
grep -q "OVERALL: PASS" tb.log || { echo "wrapper simulation FAILED - not building"; exit 1; }
cd - > /dev/null
KDIR=$D TOP=top_jet EXTRA_V=$PWD/hw/jet_shim.v TAG=jet_${P}_M$M bash hw/build_board.sh $FLASH
