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
echo 4. 自动打开企业微信并进入通讯录（不录屏）
echo 5. 首次设置自动打开的导航样本
choice /c 12345 /n /m "请选择 1 / 2 / 3 / 4 / 5："
if errorlevel 5 goto startup_setup
if errorlevel 4 goto prepare
if errorlevel 3 goto full
if errorlevel 2 goto trial
".venv\Scripts\python.exe" app.py calibrate
goto done
:trial
".venv\Scripts\python.exe" app.py run --limit 10 --snapshots
goto done
:full
".venv\Scripts\python.exe" app.py run
goto done
:prepare
".venv\Scripts\python.exe" -X utf8 startup.py prepare
goto done
:startup_setup
".venv\Scripts\python.exe" -X utf8 startup.py setup
:done
pause
