@echo off
setlocal

:: Define the parameters for each run
set "PARAMS1=--params ./params.json --param_set params3 --model_name resnet56 --use_val --val_size 0.1 --disable_test --early_stopping_patience 15 --early_stopping_start_epoch 90" 
set "PARAMS2=--params ./params.json --param_set params5 --model_name resnet56 --use_val --val_size 0.1 --disable_test --early_stopping_patience 15 --early_stopping_start_epoch 210"
set "PARAMS3=--params ./params.json --param_set params6 --model_name resnet56 --use_val --val_size 0.1 --disable_test --early_stopping_patience 15 --early_stopping_start_epoch 90"
set "PARAMS4=--params ./params.json --param_set params8 --model_name resnet56 --use_val --val_size 0.1 --disable_test --early_stopping_patience 15 --early_stopping_start_epoch 150"

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

for /L %%j in (1,1,%RUN_COUNT%) do (
    echo Running training run 2, iteration %%j with parameters: %PARAMS2%
    python train.py %PARAMS2%
    if %ERRORLEVEL% neq 0 (
        echo Training run 2, iteration %%j failed.
        exit /b %ERRORLEVEL%
    )
)

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
