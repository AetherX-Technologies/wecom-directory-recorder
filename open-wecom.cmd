@echo off
cd /d "%~dp0"
chcp 65001 >nul
if not exist ".venv\Scripts\python.exe" (
  echo 请先运行 setup.ps1 安装依赖。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 startup.py prepare
pause
