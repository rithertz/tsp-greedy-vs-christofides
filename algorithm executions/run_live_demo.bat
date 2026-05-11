@echo off
setlocal
cd /d "%~dp0\.."

where python >nul 2>nul
if %errorlevel%==0 (
    python "algorithm executions\live_tsp_demo.py"
    goto :eof
)

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 "algorithm executions\live_tsp_demo.py"
    goto :eof
)

echo Python was not found on PATH.
echo Install Python 3 and run: python "algorithm executions\live_tsp_demo.py"
pause
