#!/bin/bash
# find_uart.sh -- print the serial port of the USB-UART adapter (PL2303).
# Do NOT use the FTDI ports of the board: opening them resets the FPGA.
#   PORT=$(bash hw/setup/find_uart.sh) && python3 hw/jet_inference.py --port $PORT ...
shopt -s nullglob
for link in /dev/serial/by-id/*; do
  case "$link" in
    *Prolific*|*PL2303*|*pl2303*) readlink -f "$link"; exit 0 ;;
  esac
done
# fallback: kernel log
dev=$(dmesg 2>/dev/null | grep -o 'pl2303 converter now attached to ttyUSB[0-9]*' | tail -1 | grep -o 'ttyUSB[0-9]*')
if [ -n "$dev" ]; then echo "/dev/$dev"; exit 0; fi
echo "PL2303 adapter not found. Attach it with setup/attach_board.ps1 (Windows) and check lsusb." >&2
ls /dev/ttyUSB* 2>/dev/null >&2
exit 1
