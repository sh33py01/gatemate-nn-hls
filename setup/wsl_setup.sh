#!/bin/bash
# wsl_setup.sh -- Linux (WSL Ubuntu 24.04) part of the setup. Run once on a fresh machine,
# after setup/windows_setup.ps1. Safe to run again: finished steps are skipped.
#
#   bash ~/soda-benchmarks/tutorials/pytorch/hw/setup/wsl_setup.sh
#
# Installs / prepares:
#   * apt packages, Docker engine (for the SODA devcontainer), Python 3.11
#   * OSS CAD Suite in ~/oss-cad-suite (Yosys, nextpnr-himbaechel, gmpack, openFPGALoader, Verilator)
#   * Python venv ~/thesis-venv (torch CPU, numpy, pyserial, hls4ml 1.3.0, scikit-learn)
#   * USB permissions for the board (FTDI) and the UART adapter (PL2303)
#   * soda-benchmarks layout with this repository in tutorials/pytorch/hw
#   * Bambu wrapper without --disable-function-proxy, SODA Docker image
#   * shell function `gmenv` that activates OSS CAD Suite + venv in the right order
#
# Options (environment variables):
#   OSS_CAD_TAG=2026-10-15   install this OSS CAD Suite release instead of the latest
#                            (results in the thesis: Yosys 0.68+206, see README)
#   SKIP_DOCKER_PULL=1       do not download the SODA image now (~several GB)
set -euo pipefail

SODA_IMAGE=agostini01/soda-bookworm:v19.02
SODA_DIR=$HOME/soda-benchmarks
HW_DIR=$SODA_DIR/tutorials/pytorch/hw
OSS_DIR=$HOME/oss-cad-suite
VENV=$HOME/thesis-venv
SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)     # repository root

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$*"; }

[ "$(id -u)" -ne 0 ] || { echo "Run as your normal user (the script uses sudo where needed)."; exit 1; }
grep -qi microsoft /proc/version || warn "Not running in WSL - continuing anyway (native Linux works too)."

# ---------------------------------------------------------------------------
say "apt packages"
sudo apt-get update
sudo apt-get install -y git curl wget ca-certificates build-essential usbutils \
  software-properties-common zip unzip jq docker.io
if ! command -v python3.11 >/dev/null; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
fi
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

# ---------------------------------------------------------------------------
say "groups (docker, dialout, plugdev)"
for g in docker dialout plugdev; do
  getent group $g >/dev/null || sudo groupadd $g
  id -nG "$USER" | grep -qw $g || { sudo usermod -aG $g "$USER"; NEED_RELOGIN=1; }
done

say "systemd / docker service"
if ! grep -qs '^systemd=true' /etc/wsl.conf && grep -qi microsoft /proc/version; then
  printf '[boot]\nsystemd=true\n' | sudo tee -a /etc/wsl.conf >/dev/null
  NEED_WSL_RESTART=1
fi
if [ -d /run/systemd/system ]; then
  sudo systemctl enable --now docker
else
  sudo service docker start || true
fi

# ---------------------------------------------------------------------------
say "USB permissions (udev rules)"
sudo tee /etc/udev/rules.d/99-gatemate.rules >/dev/null <<'EOF'
# GateMate evaluation board, FTDI FT2232H (JTAG via openFPGALoader)
SUBSYSTEM=="usb", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6010", MODE="0666", GROUP="plugdev"
# Prolific PL2303 USB-UART adapter
SUBSYSTEM=="tty", ATTRS{idVendor}=="067b", MODE="0666", GROUP="dialout"
EOF
sudo udevadm control --reload-rules 2>/dev/null || true
sudo udevadm trigger 2>/dev/null || true

# ---------------------------------------------------------------------------
say "OSS CAD Suite -> $OSS_DIR"
if [ -x "$OSS_DIR/bin/yosys" ] && [ -z "${OSS_CAD_TAG:-}" ]; then
  echo "already installed: $("$OSS_DIR/bin/yosys" -V)"
else
  API=https://api.github.com/repos/YosysHQ/oss-cad-suite-build/releases
  if [ -n "${OSS_CAD_TAG:-}" ]; then REL=$(curl -fsSL "$API/tags/$OSS_CAD_TAG")
  else                              REL=$(curl -fsSL "$API/latest"); fi
  URL=$(echo "$REL" | jq -r '.assets[].browser_download_url | select(test("linux-x64-.*\\.tgz$"))' | head -1)
  [ -n "$URL" ] || { echo "Could not find the OSS CAD Suite download (tag ${OSS_CAD_TAG:-latest})."; exit 1; }
  echo "downloading $URL"
  TMP=$(mktemp -d)
  curl -fL --progress-bar "$URL" -o "$TMP/oss.tgz"
  rm -rf "$OSS_DIR"
  tar -xzf "$TMP/oss.tgz" -C "$HOME"          # unpacks to ~/oss-cad-suite
  rm -rf "$TMP"
  "$OSS_DIR/bin/yosys" -V
fi

# ---------------------------------------------------------------------------
say "Python venv -> $VENV"
[ -x "$VENV/bin/python" ] || python3.11 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/python" -m pip install -r "$SELF_DIR/requirements.txt"

# ---------------------------------------------------------------------------
say "soda-benchmarks layout"
[ -d "$SODA_DIR/.git" ] || git clone https://github.com/pnnl/soda-benchmarks.git "$SODA_DIR"
if [ "$SELF_DIR" != "$HW_DIR" ]; then
  if [ -e "$HW_DIR" ]; then
    warn "$HW_DIR already exists and is not this checkout ($SELF_DIR); leaving it alone."
  else
    ORIGIN=$(git -C "$SELF_DIR" remote get-url origin 2>/dev/null || true)
    if [ -n "$ORIGIN" ]; then git clone "$ORIGIN" "$HW_DIR"; else cp -r "$SELF_DIR" "$HW_DIR"; fi
    echo "repository placed in $HW_DIR (the scripts expect this location)"
  fi
fi
chmod +x "$HW_DIR"/*.sh "$HW_DIR"/setup/*.sh 2>/dev/null || true

S=$SODA_DIR/scripts
if [ -f "$S/ll_to_verilog.sh" ] && [ ! -f "$S/ll_to_verilog_share.sh" ]; then
  sed 's/--disable-function-proxy//' "$S/ll_to_verilog.sh" > "$S/ll_to_verilog_share.sh"
  chmod +x "$S/ll_to_verilog_share.sh"
  echo "created scripts/ll_to_verilog_share.sh (shared floating-point units)"
fi

# ---------------------------------------------------------------------------
say "SODA Docker image ($SODA_IMAGE)"
if [ "${SKIP_DOCKER_PULL:-0}" = "1" ]; then
  echo "skipped (SKIP_DOCKER_PULL=1); VS Code pulls it when the devcontainer is opened"
elif sudo docker image inspect "$SODA_IMAGE" >/dev/null 2>&1; then
  echo "already present"
else
  sudo docker pull "$SODA_IMAGE"
fi

# ---------------------------------------------------------------------------
say "shell helper 'gmenv'"
if ! grep -q 'gmenv()' "$HOME/.bashrc"; then
  cat >> "$HOME/.bashrc" <<'EOF'

# GateMate thesis: OSS CAD Suite first, then the venv, so that python3 is the venv's
gmenv() {
  source "$HOME/oss-cad-suite/environment"
  source "$HOME/thesis-venv/bin/activate"
  cd "$HOME/soda-benchmarks/tutorials/pytorch"
}
EOF
  echo "added gmenv() to ~/.bashrc"
fi

# ---------------------------------------------------------------------------
say "Done"
[ -n "${NEED_WSL_RESTART:-}" ] && warn "systemd was enabled: in PowerShell run  wsl --shutdown  and reopen Ubuntu."
[ -n "${NEED_RELOGIN:-}" ]     && warn "New groups were added: close and reopen the Ubuntu terminal (or wsl --shutdown)."
cat <<EOF

Check everything with:   bash $HW_DIR/setup/check_env.sh

Daily use:
  WSL side:        gmenv            (OSS CAD Suite + venv, cd to tutorials/pytorch)
  SODA/Bambu side: code $SODA_DIR   -> "Dev Containers: Reopen in Container"
  Board:           PowerShell (Admin): setup\\attach_board.ps1 ; then  bash hw/setup/find_uart.sh
EOF
