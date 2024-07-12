@echo off
setlocal

:: Set the log directory
set LOGDIR=runs

:: Set the host to bind TensorBoard to
set HOST=0.0.0.0

:: Set the port to bind TensorBoard to
set PORT=6006

:: Get the local IPv4 address
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /r /c:"IPv4 Address.*192\.168\." /c:"IPv4 Address.*10\."') do for /f "tokens=1 delims= " %%b in ("%%a") do set LOCAL_IP=%%b

:: Start TensorBoard
echo Starting TensorBoard with logdir %LOGDIR% bound to %HOST% on port %PORT%
echo TensorBoard will be started at http://%HOST%:%PORT%/ or http://%LOCAL_IP%:%PORT%/
tensorboard --logdir=%LOGDIR% --host=%HOST% --port=%PORT%

endlocal
pause
