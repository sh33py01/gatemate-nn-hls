# windows_setup.ps1 -- Windows part of the setup (run once on a fresh PC).
#
# Run in PowerShell AS ADMINISTRATOR:
#   Set-ExecutionPolicy -Scope Process Bypass -Force
#   .\windows_setup.ps1
#
# Installs: WSL2 + Ubuntu 24.04, usbipd-win (USB passthrough of the board and the
# UART adapter into WSL), VS Code with the WSL and Dev Containers extensions.
# It is safe to run again; finished steps are skipped.
# On a PC without WSL it stops after enabling WSL and asks for a reboot: reboot,
# then run it again.

$ErrorActionPreference = "Stop"
$Distro = "Ubuntu-24.04"

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }

# --- administrator check ---------------------------------------------------
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Run this script in PowerShell as Administrator." -ForegroundColor Red
    exit 1
}

# --- winget ----------------------------------------------------------------
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Host "winget not found. Install 'App Installer' from the Microsoft Store, then rerun." -ForegroundColor Red
    exit 1
}

function WingetInstall($pkgId) {
    $installed = winget list --id $pkgId --exact --accept-source-agreements 2>$null | Out-String
    if ($installed -match [regex]::Escape($pkgId)) {
        Write-Host "$pkgId already installed"
    } else {
        winget install --id $pkgId --exact --silent --accept-package-agreements --accept-source-agreements
    }
}

# --- WSL -------------------------------------------------------------------
Step "WSL2"
wsl --status *> $null
if ($LASTEXITCODE -ne 0) {
    wsl --install --no-distribution
    Write-Host "`nWSL was enabled. REBOOT the PC, then run this script again." -ForegroundColor Yellow
    exit 0
}
wsl --set-default-version 2 | Out-Null
wsl --update

Step "Ubuntu ($Distro)"
# wsl -l -q prints UTF-16; strip the NUL characters before matching
$distros = (wsl -l -q | Out-String) -replace "`0", ""
if ($distros -notmatch [regex]::Escape($Distro)) {
    wsl --install -d $Distro --no-launch
    Write-Host "Ubuntu installed. It will ask for a Linux user name and password on first start."
} else {
    Write-Host "$Distro already installed"
}

# --- USB passthrough -------------------------------------------------------
Step "usbipd-win (USB devices into WSL)"
WingetInstall "dorssel.usbipd-win"

# --- VS Code ---------------------------------------------------------------
Step "VS Code + extensions"
WingetInstall "Microsoft.VisualStudioCode"
$codeCandidates = @(
    "$env:LOCALAPPDATA\Programs\Microsoft VS Code\bin\code.cmd",
    "$env:ProgramFiles\Microsoft VS Code\bin\code.cmd"
)
$code = $codeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($code) {
    & $code --install-extension ms-vscode-remote.remote-wsl --force
    & $code --install-extension ms-vscode-remote.remote-containers --force
} else {
    Write-Host "VS Code installed but code.cmd not found yet; install the 'WSL' and 'Dev Containers' extensions from VS Code." -ForegroundColor Yellow
}

# --- next steps --------------------------------------------------------------
Step "Done with the Windows part"
Write-Host @"
Next:
  1. Open 'Ubuntu 24.04' from the Start menu once and create your Linux user.
  2. In Ubuntu run (replace <you> with your GitHub user):
       git clone https://github.com/pnnl/soda-benchmarks.git ~/soda-benchmarks
       git clone https://github.com/<you>/gatemate-nn-hls.git ~/soda-benchmarks/tutorials/pytorch/hw
       bash ~/soda-benchmarks/tutorials/pytorch/hw/setup/wsl_setup.sh
  3. Plug in the board and the UART adapter, then (PowerShell as Administrator):
       .\attach_board.ps1
"@
