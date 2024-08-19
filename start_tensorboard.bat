@echo off
setlocal

:: Check if the log directory argument is provided
if "%~1"=="" (
    echo Error: Please provide the log directory as an argument.
    echo Usage: %~nx0 <logdir>
    endlocal
    exit /b 1
)

:: Set the log directory from the first argument
set LOGDIR=%~1

:: Set the host to bind TensorBoard to
set HOST=0.0.0.0

:: Set the port to bind TensorBoard to
set PORT=6006

:: Get the local IPv4 address
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /r /c:"IPv4 Address.*192\.168\." /c:"IPv4 Address.*10\."') do for /f "tokens=1 delims= " %%b in ("%%a") do set LOCAL_IP=%%b

:: Start TensorBoard
echo Starting TensorBoard with logdir %LOGDIR% bound to %HOST% on port %PORT%
echo TensorBoard will be started at http://127.0.0.1:%PORT%/ or http://%LOCAL_IP%:%PORT%/
tensorboard --logdir=%LOGDIR% --host=%HOST% --port=%PORT%

endlocal
pause
