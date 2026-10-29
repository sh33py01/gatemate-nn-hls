# Neural-network inference on the GateMate CCGM1A1 with open-source HLS

Scripts and RTL for running small neural networks on the **Cologne Chip GateMate CCGM1A1**
FPGA (evaluation board V3.2A) using only open-source tools:

```
PyTorch -> torch-mlir (TOSA) -> SODA-OPT -> MLIR/LLVM IR -> Bambu HLS -> Verilog
        -> Yosys (synth_gatemate) -> nextpnr-himbaechel -> gmpack -> openFPGALoader
```

and for comparing it with the **hls4ml** approach (quantised fixed point, reuse factor) on the
same board. Every result below was measured on the board over UART and checked against
PyTorch or against hls4ml's bit-accurate emulation.

Part of the Master's thesis of Erta Tershana, Computer Engineering, Università degli Studi
di Padova (2026).

## Results (GateMate CCGM1A1, 10 MHz board clock)

| Design | Arithmetic | Cycles | Latency | LUT | FF | BRAM (of 64 halves) | Board check |
|---|---|---|---|---|---|---|---|
| MyNet CNN, MNIST 0–4 vs 5–9 | float32 (SODA) | 62,976 | 6.30 ms | 64% | 24% | 20 | 95.0% on 500 images = PyTorch, 500/500 same class |
| Jet tagger | float32 (SODA) | 38,493 | 3.85 ms | 39% | 13% | 25 | 77.7% on 1,000 jets = PyTorch, 1000/1000 same class |
| Jet tagger, 1 multiplier | `ap_fixed<16,6>` | 13,045 | 1.30 ms | 11% | 3% | 12 | 78.0%, 1000/1000 bit-exact vs hls4ml |
| Jet tagger, 16 multipliers | `ap_fixed<16,6>` | 1,676 | 168 µs | 50% | 16% | 12 | 78.0%, 1000/1000 bit-exact vs hls4ml |

Reuse-factor sweep (jet tagger, `ap_fixed<16,6>`; hls4ml-equivalent reuse factor of fc1 =
1024/M):

| M multipliers | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Cycles | 13,045 | 6,789 | 4,597 | 2,949 | 1,676 |
| LUT | 11% | 13% | 17% | 27% | 50% |
| FF | 3% | 4% | 4% | 8% | 16% |
| BRAM halves | 12 | 12 | 16 | 20 | 12 |
| f_max (nextpnr) | 31.8 MHz | 30.4 MHz | 23.8 MHz | 16.5 MHz | 15.9 MHz |

Accuracy of the jet tagger over 20,000 test jets (emulation): float32 77.01%,
`ap_fixed<16,6>` 77.04%.

## What had to be changed to make the SODA/Bambu flow work on GateMate

These are the main findings. Each one is handled by the scripts in this repository.

1. **Bare-pointer calling convention.** MLIR's default lowering gives the kernel 18
   parameters (memref descriptors), while soda-opt's testbench passes 2 pointers; the
   simulation crashes. Fix: `--convert-func-to-llvm='use-bare-ptr-memref-call-conv=1'`.
2. **No heap, no external weights.** With `NO_BRAM`, Bambu expects weights, `expf` tables
   and a 384 KB `malloc` heap in external memory. Fix: `--promote-buffers-to-stack` and
   `--memory-allocation-policy=ALL_BRAM`.
3. **Shared floating-point units.** With `--disable-function-proxy` every soft-float call site
   becomes its own unit (about 18 multipliers, 10 adders); removing the flag gives one per type.
4. **Runtime weight transposes.** torch-mlir emits a transpose for every `nn.Linear` weight,
   which becomes a runtime copy (8 KB for MyNet's fc1, implemented in flip-flops).
   `fold_weight_transpose.py` reads the original weights with swapped indices instead.
5. **Block-RAM inference.** Bambu marks its RAMs with
   `/* synthesis syn_ramstyle = "no_rw_check" */`, a comment Yosys ignores, so some
   dual-port RAMs end up in flip-flops. `build_board.sh` rewrites it as `(* no_rw_check *)`.
   For MyNet this took the design from 109% to 64% LUT.
6. **Yosys/nextpnr settings.** `synth_gatemate -luttree -nomx8` (nextpnr-himbaechel has no
   `CC_LUT3/4` or `CC_MX8`), `read_verilog -defer` (Bambu's memory templates have a default
   `array.mem`).
7. **Reuse factor in Bambu.** hls4ml's C++ cannot be compiled by Bambu (its clang crashes
   on the Xilinx `ap_fixed` headers). `gen_int_c.py` translates the hls4ml model into
   bit-exact integer C with M multiplications per loop iteration, and a Bambu
   `tech_constraints` file limits the design to M multipliers.

## Repository contents

All files are meant to live in `soda-benchmarks/tutorials/pytorch/hw/`; the scripts use that
layout.

| File | Purpose |
|---|---|
| `nn_core.v` | Wrapper: Bambu kernel memory bus (2 channels), input BRAM, output registers, cycle counter, illegal-access flag |
| `top_uart.v`, `top_jet.v`, `top_uart.ccf` | UART top (57600 8N1) for MyNet / jet tagger, pin constraints |
| `jet_shim.v` | Port adapter for kernels Bambu compiles from C (`in_raw`/`out_raw` → `P0`/`P1`) |
| `tb_nn_core.v`, `tb_jet.v`, `jet_tb_data.py` | Verilator testbenches: wrapper + real Bambu kernel |
| `build_board.sh` | Yosys → nextpnr → gmpack → openFPGALoader |
| `train_mynet.py` | Train MyNet on MNIST (2 classes) |
| `export_tosa.py`, `extract_tosa_weights.py` | Export to TOSA with a bit-exact weight check; extract weights from an existing TOSA file |
| `prepare_soda.py`, `fold_weight_transpose.py` | Automatic `soda.launch` insertion, strided-copy rewrite, transpose fold |
| `flow_container.sh`, `flow_board.sh` | Full SODA/Bambu flow (devcontainer) and board build (WSL) |
| `gatemate_inference.py` | MyNet board test (random inputs or MNIST accuracy) |
| `hls4ml_jet.py` | Train the jet tagger, convert with hls4ml, emulate fixed point |
| `prep_hls4ml_for_bambu.py`, `gen_int_c.py`, `run_bambu_int.sh`, `flow_board_jet.sh` | hls4ml model → integer C → Bambu (M multipliers) → board |
| `jet_inference.py`, `jetf_inference.py` | Jet tagger board tests (fixed point / float32) |
| `run_bambu_hls4ml.sh` | Direct Bambu run on hls4ml's C++ (documents the front-end crash) |
| `setup/` | New-PC setup: `windows_setup.ps1`, `wsl_setup.sh`, `attach_board.ps1`, `check_env.sh`, `find_uart.sh` |
| `requirements.txt` | Python packages for the WSL venv |
| `results/` | Board result CSVs and utilisation reports |

## Setup on a new PC (Windows 11 + WSL2)

The setup is scripted. Two scripts install everything the flow needs; a third passes the
board through to WSL.

**1. Windows part** (PowerShell **as Administrator**; download `setup/windows_setup.ps1`
and `setup/attach_board.ps1` from this repository first):

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\windows_setup.ps1        # WSL2, Ubuntu 24.04, usbipd-win, VS Code + WSL/Dev Containers
```

On a PC without WSL the script stops after enabling it: reboot and run it again. Then open
*Ubuntu 24.04* from the Start menu once to create the Linux user.

**2. Linux part** (Ubuntu terminal):

```bash
git clone https://github.com/pnnl/soda-benchmarks.git ~/soda-benchmarks
git clone https://github.com/sh33py01/gatemate-nn-hls.git ~/soda-benchmarks/tutorials/pytorch/hw
bash ~/soda-benchmarks/tutorials/pytorch/hw/setup/wsl_setup.sh
bash ~/soda-benchmarks/tutorials/pytorch/hw/setup/check_env.sh
```

`wsl_setup.sh` installs the apt packages, Docker, Python 3.11, the OSS CAD Suite
(`~/oss-cad-suite`), the Python venv (`~/thesis-venv`, from `requirements.txt`), udev rules
for the board and the UART adapter, the Bambu wrapper `ll_to_verilog_share.sh` (without
`--disable-function-proxy`), the SODA Docker image, and a shell function `gmenv` that
activates OSS CAD Suite and the venv in the right order. It can be run again safely.

The OSS CAD Suite is installed at its latest release. The results below were obtained with
Yosys 0.68+206; to install a specific release use `OSS_CAD_TAG=<release tag> bash setup/wsl_setup.sh`.

**3. Board.** Windows does not give USB devices to WSL automatically. After every reboot or
re-plug (PowerShell as Administrator):

```powershell
.\attach_board.ps1          # or  .\attach_board.ps1 -Auto  to re-attach on re-plug
```

This attaches the board's FTDI chip (JTAG, `0403:6010`) and the PL2303 UART adapter
(`067b:2303`/`067b:23a3`). No Windows drivers need to be installed: both devices are used
from Linux, which has the drivers built in.

**4. SODA/Bambu side.** `code ~/soda-benchmarks`, then *Dev Containers: Reopen in Container*.

### Requirements (installed by the scripts)

* Windows + WSL2 (Ubuntu 24.04) or native Linux (run only `setup/wsl_setup.sh`).
* [pnnl/soda-benchmarks](https://github.com/pnnl/soda-benchmarks) devcontainer
  (image `agostini01/soda-bookworm:v19.02`: soda-opt, mlir-opt, Bambu, torch-mlir).
  **Bambu and the TOSA export must run inside the devcontainer.**
* [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build): Yosys, nextpnr-himbaechel,
  gmpack, openFPGALoader, Verilator.
* Python 3.11 venv: `torch` (CPU), `numpy`, `pyserial`, `hls4ml==1.3.0`, `scikit-learn`, `pandas`.
* GateMate evaluation board V3.2A and a 3.3 V USB-UART adapter on Pmod J17A:
  pin 2 = `IO_NB_A1` (FPGA TX → adapter RX), pin 3 = `IO_NB_A2` (FPGA RX ← adapter TX), pin 5 = GND.

## Usage

Commands marked *container* run in the devcontainer terminal
(`/workspaces/soda-benchmarks/tutorials/pytorch`), the others in WSL with OSS CAD Suite
sourced (`~/soda-benchmarks/tutorials/pytorch`).

**Find the UART port first.** The board's FTDI chip also creates serial ports; opening one of
them resets the FPGA and erases the design. Use the PL2303 port:
`PORT=$(bash hw/setup/find_uart.sh)`, then pass `--port $PORT` to the board scripts below.
In WSL, start each session with `gmenv`.

### MyNet (float32, SODA flow)

```bash
python3 hw/train_mynet.py --epochs 10                                   # -> output/trained/
bash hw/flow_container.sh trained output/trained/mynet_trained.pth      # container
bash hw/flow_board.sh trained output/trained/mynet_trained.pth          # WSL: sim + build + flash
python3 hw/gatemate_inference.py --port /dev/ttyUSBx --weights output/trained/mynet_trained.pth \
    --dataset output/trained/test_set.npz --n 500 --csv output/trained/board_results.csv
```

### Jet tagger, fixed point with M multipliers (hls4ml model through Bambu)

```bash
python3 hw/hls4ml_jet.py --no-softmax                                   # train + hls4ml + emulation
python3 hw/prep_hls4ml_for_bambu.py output/jet/hls4ml_prj output/jet/bambu_src \
    --emu output/jet/hls4ml_prj_emulation.npz --n-tb 50
for M in 1 2 4 8 16; do python3 hw/gen_int_c.py output/jet/hls4ml_prj output/jet/rf_M$M \
    --mult $M --tb-from output/jet/bambu_src; done
for M in 1 2 4 8 16; do bash hw/run_bambu_int.sh output/jet/rf_M$M; done           # container
for M in 1 2 4 8 16; do PREFIX=rf bash hw/flow_board_jet.sh $M noflash; done       # WSL
openFPGALoader -b gatemate_evb_jtag output/board_jet_rf_M16/top.bit
python3 hw/jet_inference.py --port /dev/ttyUSBx --emu output/jet/hls4ml_prj_emulation.npz --n 1000
```

### Jet tagger, float32 (SODA flow)

```bash
MODEL=jet NOUT=5 bash hw/flow_container.sh jetf output/jet/jet_model.pth           # container
KDIR=output/jetf/sim TOP=top_jetf TAG=jetf bash hw/build_board.sh                  # WSL
python3 hw/jetf_inference.py --port /dev/ttyUSBx --weights output/jet/jet_model.pth \
    --data output/jet/test_set.npz --n 1000
```

## UART protocol

Host sends the input bytes (MyNet: 4096 = 1×1×32×32 float32; jet tagger: 32 = 16 × int16
raw `ap_fixed<16,6>`, or 64 = 16 × float32). The board runs the kernel and replies
`AA 55 | result words (little endian) | cycles (u32) | flags (u8)`. Bit 0 of `flags` is set if
the kernel accessed memory outside the input/output buffers. LED D1 toggles after each
inference.

## Notes and limitations

* Bambu schedules for the nangate45 ASIC library at 5 ns; it has no GateMate target. The
  GateMate timing comes from nextpnr; all designs meet the 10 MHz board clock.
* Latencies are kernel cycles counted in hardware; UART transfer time is not included.
* The fixed-point jet tagger is an integer-C translation of hls4ml's quantised model
  (verified bit-exact), not hls4ml's own C++. The softmax is not implemented in hardware.
* Generated files are not part of this repository: Bambu's Verilog contains PandA IP library
  code (LGPLv3) and hls4ml projects include Xilinx `ap_types` headers. Everything is
  regenerated by the scripts.

## License

See `LICENSE`. This repository contains only original scripts and RTL; the tools it uses
(SODA-OPT, Bambu/PandA, hls4ml, Yosys, nextpnr, torch-mlir) keep their own licenses.

## Citation

If you use this work, please cite the thesis:
E. Tershana, *Neural network inference on the GateMate FPGA with open-source HLS*,
Master's thesis, Università degli Studi di Padova, 2026. (Final title to be updated.)
