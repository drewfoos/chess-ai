# Build script for chess-ai on Windows (PowerShell).
#
# Configures and builds the C++ engine, optional neural evaluator, and the
# chess_mcts Python bindings (via pybind11) in one shot.
#
# Usage:
#   .\build.ps1                      # Full build (Release, neural + python bindings)
#   .\build.ps1 -Clean               # Remove build/ and reconfigure from scratch
#   .\build.ps1 -NoNeural            # Core engine only (no LibTorch, no chess_mcts)
#   .\build.ps1 -Config Debug        # Debug build
#   .\build.ps1 -Jobs 8              # Parallel build jobs (default: all cores)

[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$NoNeural,
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo')]
    [string]$Config = 'Release',
    [int]$Jobs = 0,
    [string]$BuildDir = 'build',
    [string]$LibTorchDir = ''
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Section($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Info($msg)    { Write-Host "  $msg" -ForegroundColor DarkGray }
function Warn($msg)    { Write-Host "  $msg" -ForegroundColor Yellow }

Section "chess-ai build"
Info "Build dir:  $BuildDir"
Info "Config:     $Config"
Info "Neural:     $(if ($NoNeural) { 'disabled' } else { 'enabled (LibTorch + pybind11)' })"

# Clean
if ($Clean -and (Test-Path $BuildDir)) {
    Section "Cleaning $BuildDir"
    Remove-Item -Recurse -Force $BuildDir
}

# Locate LibTorch
$enableNeural = -not $NoNeural
$torchPrefix = ''
if ($enableNeural) {
    if ($LibTorchDir -ne '') {
        $torchPrefix = (Resolve-Path $LibTorchDir).Path
    } elseif (Test-Path "$PSScriptRoot\libtorch") {
        $torchPrefix = (Resolve-Path "$PSScriptRoot\libtorch").Path
    } elseif ($env:LIBTORCH_DIR -and (Test-Path $env:LIBTORCH_DIR)) {
        $torchPrefix = (Resolve-Path $env:LIBTORCH_DIR).Path
    } else {
        Warn "LibTorch not found at .\libtorch or `$env:LIBTORCH_DIR — configuring without neural support."
        Warn "Pass -LibTorchDir <path> or set `$env:LIBTORCH_DIR to build neural + python bindings."
        $enableNeural = $false
    }
    if ($enableNeural) { Info "LibTorch:   $torchPrefix" }
}

# Locate pybind11 (Python package install)
$pybind11Dir = ''
if ($enableNeural) {
    try {
        $pybind11Dir = (& python -c "import pybind11, pathlib; print(pathlib.Path(pybind11.get_cmake_dir()).as_posix())" 2>$null).Trim()
    } catch {}
    if (-not $pybind11Dir) {
        Warn "pybind11 not found via Python (pip install pybind11) — Python bindings will be skipped."
    } else {
        Info "pybind11:   $pybind11Dir"
    }
}

# Configure
Section "Configure (cmake)"
$cmakeArgs = @('-B', $BuildDir, '-G', 'Visual Studio 17 2022', '-A', 'x64')
if ($enableNeural) {
    $cmakeArgs += @(
        "-DCMAKE_PREFIX_PATH=$torchPrefix",
        "-DTorch_DIR=$torchPrefix/share/cmake/Torch",
        '-DENABLE_NEURAL=ON'
    )
    if ($pybind11Dir) {
        $cmakeArgs += @('-DBUILD_PYTHON=ON', "-Dpybind11_DIR=$pybind11Dir")
    }
}
Info ("cmake " + ($cmakeArgs -join ' '))
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed (exit $LASTEXITCODE)" }

# Build
Section "Build ($Config)"
$buildArgs = @('--build', $BuildDir, '--config', $Config)
if ($Jobs -gt 0) {
    $buildArgs += @('--parallel', "$Jobs")
} else {
    $buildArgs += @('--parallel')
}
& cmake @buildArgs
if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE)" }

# Summary
Section "Done"
$engineExe = Join-Path $BuildDir "$Config\chess_engine.exe"
if (Test-Path $engineExe) { Info "Engine:     $engineExe" }
$pyd = Get-ChildItem "$BuildDir\$Config\chess_mcts*.pyd" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($pyd) { Info "Python mod: $($pyd.FullName)" }
Info "Run tests:  ctest --test-dir $BuildDir --build-config $Config --output-on-failure"
