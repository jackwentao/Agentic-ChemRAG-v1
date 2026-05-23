$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:HF_HOME = Join-Path $ProjectRoot "storage\models"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $ProjectRoot "storage\models\hub"
$env:SENTENCE_TRANSFORMERS_HOME = Join-Path $ProjectRoot "storage\models\sentence-transformers"

.\.venv\Scripts\celery.exe -A backend.app.tasks.celery_app worker --loglevel=INFO --pool=solo
