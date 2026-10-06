# Installs Laya for Sarathi's capability (LoRA) routing.
#
#   .\scripts\laya-setup.ps1                 # English checkpoint (~0.85 GB)
#   .\scripts\laya-setup.ps1 -Multilingual   # plus 100+ languages (~0.68 GB more)
#   .\scripts\laya-setup.ps1 -Check          # verify only, download nothing
#   .\scripts\laya-setup.ps1 -NoXet          # plain HTTP, if the download stalls
#
# All the work is in laya_setup.py, which runs on the system interpreter,
# installs only laya==0.3.20 (no dependency changes), and puts the weights in
# %APPDATA%\com.sarathi.app\laya. Restart Sarathi afterwards.
param(
    [switch]$Multilingual,
    [switch]$Check,
    [switch]$ForcePackage,
    # Plain HTTP instead of HuggingFace's Xet client, if the download stalls.
    [switch]$NoXet
)

$ErrorActionPreference = 'Stop'

$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) {
    throw 'python was not found on PATH. Sarathi uses the system interpreter; install Python 3.10+ first.'
}

$arguments = @((Join-Path $PSScriptRoot 'laya_setup.py'))
if ($Multilingual) { $arguments += '--multilingual' }
if ($Check) { $arguments += '--check' }
if ($ForcePackage) { $arguments += '--force-package' }
if ($NoXet) { $arguments += '--no-xet' }

& $python @arguments
exit $LASTEXITCODE
