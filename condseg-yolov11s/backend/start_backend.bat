@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"
set "PYTHONPATH=%CD%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "Write-Host (-join ([char[]](70,97,115,116,65,80,73,32,21518,31471,26381,21153,21551,21160,20013,46,46,46)))"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Write-Host (-join ([char[]](35775,38382,22320,22336,58,32,104,116,116,112,58,47,47,49,50,55,46,48,46,48,46,49,58,56,48,48,48)))"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Write-Host (-join ([char[]](25353,32,67,116,114,108,43,67,32,21487,20572,27490,26381,21153)))"
echo.

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

endlocal
pause
