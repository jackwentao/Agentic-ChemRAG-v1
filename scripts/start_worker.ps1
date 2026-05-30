$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:HF_HOME = Join-Path $ProjectRoot "storage\models"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $ProjectRoot "storage\models"
Remove-Item Env:SENTENCE_TRANSFORMERS_HOME -ErrorAction SilentlyContinue

.\.venv\Scripts\celery.exe -A backend.app.tasks.celery_app worker --loglevel=INFO --pool=solo
