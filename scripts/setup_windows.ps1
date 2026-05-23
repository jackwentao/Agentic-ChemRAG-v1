$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:HF_HOME = Join-Path $ProjectRoot "storage\models"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $ProjectRoot "storage\models\hub"
$env:SENTENCE_TRANSFORMERS_HOME = Join-Path $ProjectRoot "storage\models\sentence-transformers"

New-Item -ItemType Directory -Force -Path "storage\models" | Out-Null
New-Item -ItemType Directory -Force -Path "storage\postgres" | Out-Null
New-Item -ItemType Directory -Force -Path "storage\uploads" | Out-Null
New-Item -ItemType Directory -Force -Path "storage\images" | Out-Null

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

docker compose up -d postgres redis

.\.venv\Scripts\python.exe scripts\check_env.py
pytest

Write-Host ""
Write-Host "Setup complete. Runtime data is under:" -ForegroundColor Green
Write-Host "  $ProjectRoot\storage"
