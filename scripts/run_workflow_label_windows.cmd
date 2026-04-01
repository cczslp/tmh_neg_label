@echo off
setlocal

set "REPO_ROOT=%~dp0.."
set "LOG_DIR=%REPO_ROOT%\logs"
set "STDOUT_LOG=%LOG_DIR%\ks_ds_test_workflow.stdout.log"
set "STDERR_LOG=%LOG_DIR%\ks_ds_test_workflow.stderr.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

start "" /min cmd /c "" ^
pushd "%REPO_ROOT%" ^&^& ^
python main.py --method workflow --input output/ks_ds_test.xlsx --column title --output output/ks_ds_test_workflow.xlsx --base-url https://api.siliconflow.cn/v1 --model deepseek-ai/DeepSeek-V3.2 1>>"%STDOUT_LOG%" 2>>"%STDERR_LOG%" ^&^& ^
popd ^
""

echo Started workflow labeling in background.
echo Stdout: %STDOUT_LOG%
echo Stderr: %STDERR_LOG%
