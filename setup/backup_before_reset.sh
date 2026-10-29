#!/bin/bash
# backup_before_reset.sh -- archive everything from the thesis work that is NOT in git
# (trained models, datasets, results, bitstreams, hls4ml projects, notes) into one .tgz on the
# Windows Desktop. COPY THAT FILE TO A USB DRIVE OR CLOUD STORAGE: a factory reset wipes C:.
#
#   bash ~/soda-benchmarks/tutorials/pytorch/hw/setup/backup_before_reset.sh
set -euo pipefail
STAMP=$(date +%Y%m%d_%H%M)
WINUSER=$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r' || true)
DEST=${DEST:-/mnt/c/Users/$WINUSER/Desktop}
[ -d "$DEST" ] || DEST=$HOME
OUT=$DEST/gatemate_backup_$STAMP.tgz

cd "$HOME"
ITEMS=()
for p in soda-benchmarks/tutorials/pytorch/output \
         soda-benchmarks/tutorials/pytorch/hw \
         soda-benchmarks/tutorials/pytorch/*.mlir \
         soda-benchmarks/tutorials/pytorch/*.c \
         soda-benchmarks/scripts/ll_to_verilog_share.sh \
         thesis thesis-notes Documents .bashrc; do
  for q in $p; do [ -e "$q" ] && ITEMS+=("$q"); done
done
[ ${#ITEMS[@]} -gt 0 ] || { echo "nothing found to back up"; exit 1; }

echo "Archiving:"; printf '  %s\n' "${ITEMS[@]}"
echo "Size before compression: $(du -shc "${ITEMS[@]}" 2>/dev/null | tail -1 | cut -f1)"
tar --exclude='obj_dir' --exclude='*.vcd' --exclude='*.fst' --exclude='__pycache__' \
    -czf "$OUT" "${ITEMS[@]}"
echo
echo "Backup written: $OUT ($(du -h "$OUT" | cut -f1))"
echo "Check it:       tar -tzf \"$OUT\" | less"
echo "NOW COPY IT TO A USB DRIVE OR CLOUD STORAGE before resetting the PC."
