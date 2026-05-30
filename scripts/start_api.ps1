$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:HF_HOME = Join-Path $ProjectRoot "storage\models"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $ProjectRoot "storage\models"
Remove-Item Env:SENTENCE_TRANSFORMERS_HOME -ErrorAction SilentlyContinue

.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload '--reload-exclude=.venv/*' '--reload-exclude=storage/*' '--reload-exclude=*.log'
