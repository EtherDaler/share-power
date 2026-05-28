#Requires -Version 5.1
<#
.SYNOPSIS
  Сборка нативного воркера distgpu-worker.exe (Windows).
  Нужны: CMake, MSVC или Build Tools, Git (для FetchContent), Python в PATH.
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Require-Cmd($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Write-Error "[install_native] Не найдено в PATH: $name"
    }
}

Require-Cmd cmake
if (-not (Get-Command python -ErrorAction SilentlyContinue) -and -not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Error "[install_native] Нужен Python в PATH (для subprocess обучения)"
}

$BuildDir = Join-Path $Root "build"
Write-Host "[install_native] CMake → $BuildDir"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

cmake -S $Root -B $BuildDir -DCMAKE_BUILD_TYPE=Release `
  -DIXWEBSOCKET_USE_TLS=OFF -DIXWEBSOCKET_USE_ZLIB=OFF
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[install_native] Сборка…"
cmake --build $BuildDir --config Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$exe = Join-Path $BuildDir "Release\distgpu-worker.exe"
if (-not (Test-Path $exe)) {
    $exe = Join-Path $BuildDir "distgpu-worker.exe"
}
if (-not (Test-Path $exe)) {
    Write-Error "[install_native] Не найден distgpu-worker.exe после сборки"
}

$out = Join-Path $Root "distgpu-worker.exe"
Copy-Item -Force $exe $out
Write-Host "[install_native] Готово: $out"
Write-Host "  `$env:SERVER_URL = 'ws://127.0.0.1:8765/worker'"
Write-Host "  `$env:WORKER_TOKEN = '...'"
Write-Host "  & '$out'"
