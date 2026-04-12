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
#   .\build.ps1 -Ninja               # Force Ninja generator (auto when ninja is on PATH)
#   .\build.ps1 -NoNinja             # Force MSBuild (Visual Studio) generator
#
# Speed tips:
#   - Ninja: `winget install Ninja-build.Ninja`  (auto-detected, ~30% faster)
#   - sccache: `winget install Mozilla.sccache` then `$env:CMAKE_C_COMPILER_LAUNCHER='sccache'; $env:CMAKE_CXX_COMPILER_LAUNCHER='sccache'`

[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$NoNeural,
    [switch]$TensorRT,
    [switch]$NoTensorRT,
    [switch]$Ninja,
    [switch]$NoNinja,
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo')]
    [string]$Config = 'Release',
    [int]$Jobs = 0,
    [string]$BuildDir = 'build',
    [string]$LibTorchDir = '',
    [string]$TensorRTDir = ''
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Section($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Info($msg)    { Write-Host "  $msg" -ForegroundColor DarkGray }
function Warn($msg)    { Write-Host "  $msg" -ForegroundColor Yellow }

Section "chess-ai build"

# TensorRT: auto-enable when $env:TENSORRT_PATH is set AND -NoTensorRT not passed.
# -TensorRT forces it on (errors if SDK missing). -NoTensorRT forces it off.
$trtAuto = (-not $TensorRT) -and (-not $NoTensorRT) -and $env:TENSORRT_PATH -and (Test-Path $env:TENSORRT_PATH)
if ($trtAuto) { $TensorRT = $true }
if ($NoTensorRT) { $TensorRT = $false }

# Ninja: default to on when `ninja` is on PATH (~30% faster than MSBuild on this project).
# -NoNinja forces MSBuild. -Ninja forces ninja (errors if missing).
$ninjaCmd = Get-Command ninja -ErrorAction SilentlyContinue
$ninjaAuto = (-not $Ninja) -and (-not $NoNinja) -and $ninjaCmd
if ($ninjaAuto) { $Ninja = $true }
if ($NoNinja) { $Ninja = $false }
if ($Ninja -and -not $ninjaCmd) {
    throw "ninja not found on PATH. Install with 'winget install Ninja-build.Ninja' or pass -NoNinja."
}

# Ninja on Windows needs the MSVC toolchain env (cl.exe, libs, includes) sourced.
# We locate vcvars64.bat via vswhere, capture its env, and import into this session.
function Import-VCEnvironment {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) {
        throw "vswhere not found at $vswhere. Install Visual Studio 2022 with the C++ workload."
    }
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) { throw "No VS 2022 C++ toolchain installation detected by vswhere." }
    $vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars." }
    # Run vcvars then dump env; parse and apply to current PowerShell session.
    & cmd /c "`"$vcvars`" && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }
    Info "MSVC env imported from: $vsPath"
}

Info "Build dir:  $BuildDir"
Info "Config:     $Config"
Info "Generator:  $(if ($Ninja) { 'Ninja Multi-Config' + $(if ($ninjaAuto) { ' (auto-detected)' } else { '' }) } else { 'Visual Studio 17 2022' })"
Info "Neural:     $(if ($NoNeural) { 'disabled' } else { 'enabled (LibTorch + pybind11)' })"
Info "TensorRT:   $(if ($TensorRT) { 'enabled' + $(if ($trtAuto) { ' (auto-detected)' } else { '' }) } else { 'disabled' })"

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

# Locate TensorRT (only if requested)
$trtRoot = ''
if ($TensorRT) {
    if (-not $enableNeural) {
        throw "-TensorRT requires neural support (don't combine with -NoNeural)"
    }
    if ($TensorRTDir -ne '') {
        $trtRoot = (Resolve-Path $TensorRTDir).Path
    } elseif ($env:TENSORRT_PATH -and (Test-Path $env:TENSORRT_PATH)) {
        $trtRoot = (Resolve-Path $env:TENSORRT_PATH).Path
    } else {
        throw "TensorRT not found. Set `$env:TENSORRT_PATH to the SDK root (e.g. C:\TensorRT-10.16.1.11) or pass -TensorRTDir <path>."
    }
    Info "TensorRT:   $trtRoot"
}

# Locate pybind11 (Python package install) and pin CMake to the same interpreter
$pybind11Dir = ''
$pythonExe = ''
if ($enableNeural) {
    try {
        $pythonExe = (& python -c "import sys; print(sys.executable)" 2>$null).Trim()
        $pybind11Dir = (& python -c "import pybind11, pathlib; print(pathlib.Path(pybind11.get_cmake_dir()).as_posix())" 2>$null).Trim()
    } catch {}
    if (-not $pybind11Dir) {
        Warn "pybind11 not found via Python (pip install pybind11) — Python bindings will be skipped."
    } else {
        Info "pybind11:   $pybind11Dir"
        Info "Python exe: $pythonExe"
    }
}

# Import MSVC env for Ninja (cl.exe etc.)
if ($Ninja) {
    Section "Import MSVC environment"
    Import-VCEnvironment
}

# Configure
Section "Configure (cmake)"
if ($Ninja) {
    # Ninja Multi-Config keeps the --config Release pattern working.
    $cmakeArgs = @('-B', $BuildDir, '-G', 'Ninja Multi-Config')
} else {
    $cmakeArgs = @('-B', $BuildDir, '-G', 'Visual Studio 17 2022', '-A', 'x64')
}
if ($enableNeural) {
    $cmakeArgs += @(
        "-DCMAKE_PREFIX_PATH=$torchPrefix",
        "-DTorch_DIR=$torchPrefix/share/cmake/Torch",
        '-DENABLE_NEURAL=ON'
    )
    if ($pybind11Dir) {
        $cmakeArgs += @('-DBUILD_PYTHON=ON', "-Dpybind11_DIR=$pybind11Dir")
        if ($pythonExe) {
            $pythonExeCMake = $pythonExe -replace '\\', '/'
            $cmakeArgs += @("-DPython_EXECUTABLE=$pythonExeCMake", "-DPYTHON_EXECUTABLE=$pythonExeCMake")
        }
    }
    if ($trtRoot) {
        $trtRootCMake = $trtRoot -replace '\\', '/'
        $cmakeArgs += @('-DENABLE_TENSORRT=ON', "-DTENSORRT_ROOT=$trtRootCMake")
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
