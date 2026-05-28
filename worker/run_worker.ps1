#Requires -Version 5.1
<#
.SYNOPSIS
  Запуск worker/agent.py из виртуального окружения .venv (Windows).

.DESCRIPTION
  Пример:
    $env:SERVER_URL = "ws://192.168.1.10:8765/worker"
    $env:WORKER_TOKEN = "secret-token-123"
    .\run_worker.ps1
#>

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$worker = Join-Path $ScriptDir "worker.py"

if (-not (Test-Path -LiteralPath $worker)) {
    Write-Error "[run] Не найден $worker"
}

# worker.py сам создаст .venv при первом запуске; если venv есть — быстрее стартует
if (Test-Path -LiteralPath $py) {
    & $py $worker @args
} else {
    $base = if ($env:PYTHON) { $env:PYTHON } else { "python" }
    & $base $worker @args
}
