$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $ProjectRoot "frontend")

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
    throw "npm.cmd was not found. Please install Node.js or add npm.cmd to PATH."
}

& $Npm run dev -- --host 127.0.0.1 --port 5173
