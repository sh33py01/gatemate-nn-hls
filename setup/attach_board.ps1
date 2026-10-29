# attach_board.ps1 -- pass the GateMate board (FTDI, JTAG) and the USB-UART adapter
# (Prolific PL2303) through to WSL. Run in PowerShell as Administrator after every
# reboot or re-plug (Windows does not keep the attachment).
#
#   .\attach_board.ps1           attach once
#   .\attach_board.ps1 -Auto     keep re-attaching when the devices are re-plugged
#                                (leave the window open)
param([switch]$Auto)

$Devices = @(
    @{ Name = "GateMate board (FTDI FT2232H, JTAG)"; Id = "0403:6010" },
    @{ Name = "USB-UART adapter (PL2303)";            Id = "067b:2303" },
    @{ Name = "USB-UART adapter (PL2303, newer)";     Id = "067b:23a3" }
)

if (-not (Get-Command usbipd -ErrorAction SilentlyContinue)) {
    Write-Host "usbipd not found: run windows_setup.ps1 first." -ForegroundColor Red; exit 1
}
# WSL must be running for 'attach --wsl'
wsl -d Ubuntu-24.04 -e true

$list = usbipd list | Out-String
foreach ($d in $Devices) {
    if ($list -notmatch [regex]::Escape($d.Id)) { continue }
    Write-Host "== $($d.Name) [$($d.Id)]"
    usbipd bind --hardware-id $d.Id 2>$null          # needs Administrator, once per device
    if ($Auto) {
        Start-Process powershell -ArgumentList "-NoExit", "-Command",
            "usbipd attach --wsl --auto-attach --hardware-id $($d.Id)"
    } else {
        usbipd attach --wsl --hardware-id $d.Id
    }
}
Write-Host "`nIn Ubuntu check with:  lsusb ; bash ~/soda-benchmarks/tutorials/pytorch/hw/setup/find_uart.sh"
