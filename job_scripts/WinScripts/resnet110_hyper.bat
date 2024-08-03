@echo off
setlocal

:: Define the parameters for each run
set "PARAMS1=--params ./params.json --param_set params3 --model_name resnet110 --use_val --val_size 0.1 --disable_test --early_stopping_patience 15 --early_stopping_start_epoch 90" 
set "PARAMS2=--params ./params.json --param_set params4 --model_name resnet110 --use_val --val_size 0.1 --disable_test --early_stopping_patience 15 --early_stopping_start_epoch 210"
set "PARAMS4=--params ./params.json --param_set params5 --model_name resnet110 --use_val --val_size 0.1 --disable_test --early_stopping_patience 10 --early_stopping_start_epoch 200"

:: Number of times to run each configuration
set RUN_COUNT=3

:: Run the training script with each set of parameters
@REM for /L %%j in (1,1,%RUN_COUNT%) do (
@REM     echo Running training run 1, iteration %%j with parameters: %PARAMS1%
@REM     C:\Users\jonat\anaconda3\envs\COS700-ML-1\python.exe train.py %PARAMS1%
@REM     if %ERRORLEVEL% neq 0 (
@REM         echo Training run 1, iteration %%j failed.
@REM         exit /b %ERRORLEVEL%
@REM     )
@REM )

for /L %%j in (1,1,%RUN_COUNT%) do (
    echo Running training run 2, iteration %%j with parameters: %PARAMS2%
    C:\Users\jonat\anaconda3\envs\COS700-ML-1\python.exe train.py %PARAMS2%
    if %ERRORLEVEL% neq 0 (
        echo Training run 2, iteration %%j failed.
        exit /b %ERRORLEVEL%
    )
)

for /L %%j in (1,1,%RUN_COUNT%) do (
    echo Running training run 3, iteration %%j with parameters: %PARAMS4%
    C:\Users\jonat\anaconda3\envs\COS700-ML-1\python.exe train.py %PARAMS4%
    if %ERRORLEVEL% neq 0 (
        echo Training run 3, iteration %%j failed.
        exit /b %ERRORLEVEL%
    )
)

echo All training runs completed successfully.
endlocal
pause
