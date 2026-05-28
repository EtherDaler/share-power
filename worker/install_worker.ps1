#Requires -Version 5.1
<#
.SYNOPSIS
  Установка окружения DistGPU worker (Windows): проверка Python, venv, websockets, torch.

.DESCRIPTION
  Переменные окружения:
    PYTHON                        — полный путь к python.exe
    DISTGPU_TORCH                 — пресет: cpu | cu118 | cu121 | cu124 | cu126 | cu128 | nightly-cu128
                                    По умолчанию: cu124
    DISTGPU_PYTORCH_INDEX_URL     — свой индекс PyTorch (перекрывает пресет)
    DISTGPU_PYTORCH_EXTRA         — доп. аргументы pip для torch, напр. "--pre" (через пробел)

  Примеры:
    .\install_worker.ps1
    $env:DISTGPU_TORCH='nightly-cu128'; .\install_worker.ps1
#>

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Find-PythonExe {
    if ($env:PYTHON) {
        if (-not (Test-Path -LiteralPath $env:PYTHON)) {
            throw "[install] PYTHON указывает на несуществующий файл: $($env:PYTHON)"
        }
        return (Resolve-Path -LiteralPath $env:PYTHON).Path
    }
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        foreach ($ver in @("3.12", "3.11", "3.10")) {
            $out = & py "-$ver" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                $e = $out.Trim()
                if (Test-Path -LiteralPath $e) { return $e }
            }
        }
    }
    foreach ($name in @("python", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        $out = & $cmd.Source -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) {
            $e = $out.Trim()
            if (Test-Path -LiteralPath $e) { return $e }
        }
    }
    return $null
}

$pyExe = Find-PythonExe
if (-not $pyExe) {
    Write-Error "[install] Не найден Python 3.10+. Установите Python или задайте `$env:PYTHON='C:\Path\python.exe'"
}

& $pyExe -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "[install] Нужен Python 3.10+. Текущий: $(& $pyExe -V)"
}

Write-Host "[install] Используется: $(& $pyExe -V) ($pyExe)"

$venv = Join-Path $ScriptDir ".venv"
if (-not (Test-Path $venv)) {
    Write-Host "[install] Создаю venv: $venv"
    & $pyExe -m venv $venv
} else {
    Write-Host "[install] venv уже существует: $venv"
}

$pip = Join-Path $venv "Scripts\pip.exe"
$pyV = Join-Path $venv "Scripts\python.exe"

& $pyV -m pip install -U pip wheel setuptools
& $pip install -r (Join-Path $ScriptDir "requirements-base.txt")

function Install-Torch {
    param([string]$Preset)

    if ($env:DISTGPU_PYTORCH_INDEX_URL) {
        Write-Host "[install] torch: пользовательский index-url"
        $pipArgs = @("install")
        if ($env:DISTGPU_PYTORCH_EXTRA) {
            $pipArgs += $env:DISTGPU_PYTORCH_EXTRA.Split(" ", [StringSplitOptions]::RemoveEmptyEntries)
        }
        $pipArgs += @("torch", "--index-url", $env:DISTGPU_PYTORCH_INDEX_URL)
        & $pip @pipArgs
        return
    }

    switch ($Preset) {
        "cpu" {
            Write-Host "[install] torch: CPU (PyPI)"
            & $pip install torch
        }
        "cu118" {
            Write-Host "[install] torch: cu118"
            & $pip install torch --index-url "https://download.pytorch.org/whl/cu118"
        }
        "cu121" {
            Write-Host "[install] torch: cu121"
            & $pip install torch --index-url "https://download.pytorch.org/whl/cu121"
        }
        "cu124" {
            Write-Host "[install] torch: cu124"
            & $pip install torch --index-url "https://download.pytorch.org/whl/cu124"
        }
        "cu126" {
            Write-Host "[install] torch: cu126"
            & $pip install torch --index-url "https://download.pytorch.org/whl/cu126"
        }
        "cu128" {
            Write-Host "[install] torch: cu128 (при ошибке используйте nightly-cu128)"
            & $pip install torch --index-url "https://download.pytorch.org/whl/cu128"
        }
        "nightly-cu128" {
            Write-Host "[install] torch: nightly cu128 (RTX 50xx / Blackwell)"
            & $pip install --pre torch --index-url "https://download.pytorch.org/whl/nightly/cu128"
        }
        default {
            Write-Error "[install] Неизвестный DISTGPU_TORCH=$Preset. Допустимо: cpu cu118 cu121 cu124 cu126 cu128 nightly-cu128"
        }
    }
}

$preset = if ($env:DISTGPU_TORCH) { $env:DISTGPU_TORCH.Trim() } else { "cu124" }
Write-Host "[install] Пресет PyTorch: $preset"
Install-Torch -Preset $preset

Write-Host "[install] Проверка импортов..."
& $pyV (Join-Path $ScriptDir "verify_env.py")

Write-Host ""
Write-Host "[install] Готово. Запуск: .\run_worker.ps1"
Write-Host "[install] Задайте SERVER_URL и WORKER_TOKEN при необходимости."
