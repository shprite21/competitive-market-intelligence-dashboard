@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto python
py -3.12 -c "import sys" >nul 2>nul
if not errorlevel 1 (
    py -3.12 run.py %*
    goto done
)
py -3.11 -c "import sys" >nul 2>nul
if not errorlevel 1 (
    py -3.11 run.py %*
    goto done
)
:python
python run.py %*
:done
if errorlevel 1 pause
endlocal
