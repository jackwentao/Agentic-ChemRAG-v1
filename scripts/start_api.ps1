$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:HF_HOME = Join-Path $ProjectRoot "storage\models"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $ProjectRoot "storage\models\hub"
$env:SENTENCE_TRANSFORMERS_HOME = Join-Path $ProjectRoot "storage\models\sentence-transformers"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload '--reload-exclude=.venv/*' '--reload-exclude=storage/*' '--reload-exclude=*.log'
