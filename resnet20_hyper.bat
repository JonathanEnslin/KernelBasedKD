@echo off
setlocal

:: Define the parameters for each run
set "PARAMS1=--params ./params.json --run_name testtest --param_set params3 --model_name resnet20 --use_val --val_size 0.1 --disable_test --dataset CIFAR100"
set "PARAMS2=--params ./params.json --param_set params6 --model_name resnet20 --use_val --val_size 0.1 --disable_test"
set "PARAMS3=--params ./params.json --param_set params7 --model_name resnet20 --use_val --val_size 0.1 --disable_test"
set "PARAMS4=--params ./params.json --param_set params8 --model_name resnet20 --use_val --val_size 0.1 --disable_test"

:: Number of times to run each configuration
set RUN_COUNT=3

:: Run the training script with each set of parameters
for /L %%j in (1,1,%RUN_COUNT%) do (
    echo Running training run 1, iteration %%j with parameters: %PARAMS1%
    python train.py %PARAMS1%
    if %ERRORLEVEL% neq 0 (
        echo Training run 1, iteration %%j failed.
        exit /b %ERRORLEVEL%
    )
)

@REM for /L %%j in (1,1,%RUN_COUNT% - 1) do (
@REM     echo Running training run 2, iteration %%j with parameters: %PARAMS2%
@REM     python train.py %PARAMS2%
@REM     if %ERRORLEVEL% neq 0 (
@REM         echo Training run 2, iteration %%j failed.
@REM         exit /b %ERRORLEVEL%
@REM     )
@REM )

for /L %%j in (1,1,%RUN_COUNT%) do (
    echo Running training run 3, iteration %%j with parameters: %PARAMS3%
    python train.py %PARAMS3%
    if %ERRORLEVEL% neq 0 (
        echo Training run 3, iteration %%j failed.
        exit /b %ERRORLEVEL%
    )
)

for /L %%j in (1,1,%RUN_COUNT%) do (
    echo Running training run 4, iteration %%j with parameters: %PARAMS4%
    python train.py %PARAMS4%
    if %ERRORLEVEL% neq 0 (
        echo Training run 4, iteration %%j failed.
        exit /b %ERRORLEVEL%
    )
)

echo All training runs completed successfully.
endlocal
pause
