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
New-Item -ItemType Directory -Force -Path "storage\minio" | Out-Null
New-Item -ItemType Directory -Force -Path "storage\router_cache" | Out-Null

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "已创建 .env，请在启动 API 前填写 DEEPSEEK_API_KEY。" -ForegroundColor Yellow
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

$NpmCandidates = @(
    "D:\2java\nodejs\npm.cmd",
    "C:\Program Files\nodejs\npm.cmd"
)
$Npm = $NpmCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Npm) {
    $NpmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if ($NpmCommand) {
        $Npm = $NpmCommand.Source
    }
}
if (-not $Npm) {
    throw "未找到 npm.cmd，请安装 Node.js 或将 npm.cmd 添加到 PATH。"
}
Push-Location "frontend"
try {
    & $Npm install
} finally {
    Pop-Location
}

docker compose up -d postgres redis minio

.\.venv\Scripts\python.exe scripts\check_env.py
.\.venv\Scripts\python.exe -m pytest

Write-Host ""
Write-Host "安装与基础服务启动完成。运行数据目录：" -ForegroundColor Green
Write-Host "  $ProjectRoot\storage"
