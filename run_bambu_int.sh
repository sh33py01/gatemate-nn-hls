#!/bin/bash
# run_bambu_int.sh SRC   (devcontainer, from tutorials/pytorch)
# Bambu on the integer-C jet tagger made by gen_int_c.py, multiplier count limited by
# SRC/constraints.xml, same settings as the MyNet build; Verilator co-simulation checked
# against hls4ml's bit-accurate emulation. Results go to SRC/bambu/.
set -euo pipefail
SRC=$(realpath $1); OUT=$SRC/bambu
mkdir -p $OUT && cd $OUT
bambu -v3 --compiler=I386_CLANG16 \
  --device=nangate45 --clock-period=5 \
  --experimental-setup=BAMBU-BALANCED-MP --channels-number=2 \
  --memory-allocation-policy=ALL_BRAM --enable-function-proxy \
  --top-fname=jet_top --generate-tb=$SRC/jet_tb.c --simulate --simulator=VERILATOR \
  $SRC/jet_top.c $SRC/constraints.xml > bambu-log 2>&1 \
  || { echo "Bambu failed:"; grep -iE "error" bambu-log | head -20; tail -15 bambu-log; exit 1; }
cp jet_top.v 06_verilog.v
grep -E "^JET_OUT" bambu-log | sort -u -k2,2n > got.txt || true
N=$(wc -l < $SRC/expected.txt)
if diff got.txt $SRC/expected.txt > /dev/null; then echo "RTL vs hls4ml emulation: BIT-EXACT ($N jets)";
else echo "RTL vs hls4ml emulation: MISMATCH"; diff got.txt $SRC/expected.txt | head; fi
echo "multipliers instantiated: $(grep -oE '\bmult_expr_FU[A-Za-z0-9_]*_i[0-9]+' 06_verilog.v | sort -u | wc -l) (limit $(grep -oE 'n="[0-9]+"' $SRC/constraints.xml))"
grep -E "Average execution|Total number of flip-flops in function jet_top" bambu-log | sort -u
