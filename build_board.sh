#!/bin/bash
# Build + flash: Bambu kernel + nn_core + UART top.
# Run from tutorials/pytorch with OSS CAD Suite sourced.
#   bash hw/build_board.sh [noflash]
# Optional environment variables:
#   KDIR=output/sim_nt                       kernel directory (06_verilog.v + .mem files)
#   SYNTH_OPTS="-luttree -nomx8"             extra synth_gatemate options
#   FORCE_BRAM=1                             force Bambu's STD_DP_BRAM memories into block RAM
#   PATCH_RW=0                               do not convert Bambu no_rw_check comments to attributes
#   TOP=top_jet                              top module (default top_uart = MyNet)
#   KMOD=jet_top                             Bambu top name, renamed to forward_kernel for nn_core
#   EXTRA_V=/abs/path/shim.v                 extra Verilog file (e.g. hw/jet_shim.v)
#   TAG=name                                 output goes to output/board_<TAG>
set -euo pipefail
ROOT=$(pwd)
K=$ROOT/${KDIR:-output/sim_nt}
H=$ROOT/hw
O=$ROOT/output/board${TAG:+_$TAG}
SYNTH_OPTS=${SYNTH_OPTS:--luttree -nomx8}
TOP=${TOP:-top_uart}
PRE=""
if [ "${FORCE_BRAM:-0}" = "1" ]; then
  PRE="hierarchy -top $TOP; setattr -set ram_style \"block\" *STD_DP_BRAM*/m:*;"
fi
mkdir -p $O
{ yosys -V; echo "KDIR=$K"; echo "SYNTH_OPTS=$SYNTH_OPTS"; echo "FORCE_BRAM=${FORCE_BRAM:-0}"; echo "PATCH_RW=${PATCH_RW:-1}"; echo "TOP=$TOP KMOD=${KMOD:-}"; } | tee $O/versions.txt
cd $K                               # $readmem paths in the kernel are relative
# Bambu marks its RAMs with  /* synthesis syn_ramstyle = "no_rw_check" */  -- a comment, which Yosys
# ignores. Rewrite it as a real attribute so Yosys may use undefined-collision (TDP) block RAM.
if [ "${PATCH_RW:-1}" = "1" ]; then
  sed -E 's#(reg [^;]* memory \[[^]]*\])[[:space:]]*/\* synthesis syn_ramstyle = "no_rw_check" \*/#(* no_rw_check *) \1#' \
    06_verilog.v > 06_gm.v
  echo "no_rw_check attributes added: $(grep -c '(\* no_rw_check \*)' 06_gm.v)"
  VSRC=06_gm.v
else
  cp 06_verilog.v 06_gm.v; VSRC=06_gm.v
fi
if [ -n "${KMOD:-}" ]; then
  sed -i "s/^module ${KMOD}(/module forward_kernel(/" $VSRC
  grep -q "^module forward_kernel(" $VSRC || { echo "module ${KMOD} not found in 06_verilog.v"; exit 1; }
fi
yosys -l $O/yosys.log -p "read_verilog -defer $VSRC $H/nn_core.v $H/top_uart.v $H/top_jet.v ${EXTRA_V:-}; \
  $PRE synth_gatemate -top $TOP $SYNTH_OPTS; write_json $O/top.json" > /dev/null \
  || { grep -iE "error" $O/yosys.log | tail -5; exit 1; }
cd $ROOT
if grep -iqE "can not open|could not open|cannot open|failed to open" $O/yosys.log; then
  echo "ERROR: a \$readmem file was not found -> weights would be zero"; exit 1; fi
echo "--- memories still in flip-flops:"
grep "using FF mapping" $O/yosys.log | sed 's/.*Datapath_i\.//' || true
nextpnr-himbaechel --device CCGM1A1 --json $O/top.json --top $TOP \
  -o ccf=$H/top_uart.ccf -o out=$O/top.cfg > $O/nextpnr.log 2>&1 \
  || { grep -A20 "Device utilisation" $O/nextpnr.log; tail -3 $O/nextpnr.log; exit 1; }
gmpack $O/top.cfg $O/top.bit
grep -A20 "Device utilisation" $O/nextpnr.log | tee $O/utilisation.txt
grep -iE "Max frequency" $O/nextpnr.log | tail -3 | tee -a $O/utilisation.txt
[ "${1:-}" = "noflash" ] || openFPGALoader -b gatemate_evb_jtag $O/top.bit
echo "BUILD OK -> $O"
