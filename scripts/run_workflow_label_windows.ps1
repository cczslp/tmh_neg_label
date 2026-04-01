$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "python"
$LogDir = Join-Path $RepoRoot "logs"
$OutputPath = Join-Path $RepoRoot "output\ks_ds_test_workflow.xlsx"
$StdoutLog = Join-Path $LogDir "ks_ds_test_workflow.stdout.log"
$StderrLog = Join-Path $LogDir "ks_ds_test_workflow.stderr.log"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$Arguments = @(
    "main.py",
    "--method", "workflow",
    "--input", "output/ks_ds_test.xlsx",
    "--column", "title",
    "--output", $OutputPath,
    "--base-url", "https://api.siliconflow.cn/v1",
    "--model", "deepseek-ai/DeepSeek-V3.2"
)

$Process = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList $Arguments `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -WindowStyle Hidden `
    -PassThru

Write-Output "Started workflow labeling in background."
Write-Output "PID: $($Process.Id)"
Write-Output "Stdout: $StdoutLog"
Write-Output "Stderr: $StderrLog"
