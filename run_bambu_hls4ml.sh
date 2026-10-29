#!/bin/bash
# run_bambu_hls4ml.sh SRC OUT   (devcontainer, from tutorials/pytorch)
# Bambu on hls4ml firmware prepared by prep_hls4ml_for_bambu.py, with the same Bambu settings as
# the working MyNet build (nangate45, 5 ns, ALL_BRAM, 2 channels, shared units), plus Verilator
# co-simulation of jet_tb.c, checked against hls4ml's bit-accurate emulation.
set -euo pipefail
SRC=$(realpath $1); OUT=$2
mkdir -p $OUT && cd $OUT
bambu -v3 -O2 --std=c++14 --compiler=I386_CLANG16 \
  --device=nangate45 --clock-period=5 \
  --experimental-setup=BAMBU-BALANCED-MP --channels-number=2 \
  --memory-allocation-policy=ALL_BRAM --enable-function-proxy \
  -I$SRC -I$SRC/firmware -I$SRC/firmware/ap_types \
  --top-fname=jet_top --generate-tb=$SRC/jet_tb.c --simulate --simulator=VERILATOR \
  $SRC/jet_top.cpp $SRC/firmware/myproject.cpp > bambu-log 2>&1 \
  || { echo "Bambu failed:"; grep -iE "error|unsupported|undefined" bambu-log | head -20; tail -15 bambu-log; exit 1; }
cp jet_top.v 06_verilog.v 2>/dev/null || true
grep -E "^JET_OUT" bambu-log | sort -u > got.txt || true
echo "== Bambu RTL vs hls4ml emulation"
if diff got.txt $SRC/expected.txt > /dev/null; then echo "BIT-EXACT: $(wc -l < got.txt) jets"; else echo "MISMATCH:"; diff got.txt $SRC/expected.txt | head; fi
grep -E "Average execution|Total number of flip-flops in function jet_top" bambu-log | sort -u
