# Launcher: loads .env (LICHESS_BOT_TOKEN), prepends LibTorch / TensorRT /
# CUDA bin directories to PATH so chess_engine.exe can resolve native DLLs
# when invoked through lichess-bot's subprocess chain, then runs the bot.
#
# Usage:  .\run_bot.ps1

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot

# --- Load .env ---
$envFile = Join-Path $repoRoot '.env'
if (-not (Test-Path $envFile)) {
    Write-Error ".env not found at $envFile"
}
Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith('#') -and $line -match '^([^=]+)=(.*)$') {
        $k = $matches[1].Trim()
        $v = $matches[2].Trim().Trim('"').Trim("'")
        [Environment]::SetEnvironmentVariable($k, $v, 'Process')
    }
}

if (-not $env:LICHESS_BOT_TOKEN) {
    Write-Error "LICHESS_BOT_TOKEN not set after loading .env"
}

# --- Prepend DLL directories to PATH (LibTorch + TensorRT) ---
# The engine.exe lives next to its own DLLs in the scikit-build wheel dir,
# so Windows' exe-dir-first search usually resolves everything. But if
# TENSORRT_PATH is set, add its bin to be safe, matching what
# visualization/server.py does for its engine subprocess.
$extras = @()
if ($env:TENSORRT_PATH) {
    $trtBin = Join-Path $env:TENSORRT_PATH 'bin'
    if (Test-Path $trtBin) { $extras += $trtBin }
}
if ($extras.Count -gt 0) {
    $env:PATH = ($extras -join ';') + ';' + $env:PATH
}

# --- Run the bot ---
Push-Location (Join-Path $repoRoot 'lichess-bot')
try {
    python lichess-bot.py --config config.yml
} finally {
    Pop-Location
}
