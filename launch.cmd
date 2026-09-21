@echo off
cd /d "%~dp0"
chcp 65001 >nul
if not exist ".venv\Scripts\python.exe" (
  echo 请先右键运行 setup.ps1，或执行 powershell -ExecutionPolicy Bypass -File setup.ps1
  pause
  exit /b 1
)
echo 1. 校准区域与箭头
echo 2. 试跑 10 行并保存 PNG
echo 3. 连续录屏
choice /c 123 /n /m "请选择 1 / 2 / 3："
if errorlevel 3 goto full
if errorlevel 2 goto trial
".venv\Scripts\python.exe" app.py calibrate
goto done
:trial
".venv\Scripts\python.exe" app.py run --limit 10 --snapshots
goto done
:full
".venv\Scripts\python.exe" app.py run
:done
pause
