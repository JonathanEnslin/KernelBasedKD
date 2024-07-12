@echo off
setlocal

:: Define the parameters for each run
set PARAMS1=--params ./params.json --param_set params5 --model_name resnet20
set PARAMS2=--params ./params.json --param_set params5 --model_name resnet56
set PARAMS3=--params ./params.json --param_set params5 --model_name resnet110

:: Run the training script with the first set of parameters
@REM echo Running training run 1
@REM python train.py %PARAMS1%
@REM if %ERRORLEVEL% neq 0 (
@REM     echo Training run 1 failed.
@REM     exit /b %ERRORLEVEL%
@REM )

:: Run the training script with the second set of parameters
echo Running training run 2
python train.py %PARAMS2%
if %ERRORLEVEL% neq 0 (
    echo Training run 2 failed.
    exit /b %ERRORLEVEL%
)

:: Run the training script with the third set of parameters
echo Running training run 3
python train.py %PARAMS3%
if %ERRORLEVEL% neq 0 (
    echo Training run 3 failed.
    exit /b %ERRORLEVEL%
)

echo All training runs completed successfully.
endlocal
pause
