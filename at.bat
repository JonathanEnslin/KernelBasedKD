@echo off
setlocal

:: Define the parameters for each run
set "PARAMS1=--device cuda --params params.json --param_set params3 --model_name resnet20 --run_name at_zoo_b0p1 --dataset CIFAR100 --at_mode zoo --at_beta 0.1"
set "PARAMS2=--device cuda --params params.json --param_set params3 --model_name resnet20 --run_name at_zoo_b0p5 --dataset CIFAR100 --at_mode zoo --at_beta 0.5"
set "PARAMS3=--device cuda --params params.json --param_set params3 --model_name resnet20 --run_name at_zoo_b0p9 --dataset CIFAR100 --at_mode zoo --at_beta 0.9"
set "PARAMS4=--device cuda --params params.json --param_set params3 --model_name resnet20 --run_name at_zoo_b1p1 --dataset CIFAR100 --at_mode zoo --at_beta 1.1"
set "PARAMS5=--device cuda --params params.json --param_set params3 --model_name resnet20 --run_name at_zoo_b10p0 --dataset CIFAR100 --at_mode zoo --at_beta 10.0"

:: Number of times to run each configuration
set RUN_COUNT=2

:: Run the training script with each set of parameters
for /L %%j in (1,1,%RUN_COUNT%) do (
    echo Running training run 1, iteration %%j with parameters: %PARAMS1%
    ~\anaconda3\envs\COS700-ML-1\python.exe ".\at_train copy.py" %PARAMS1%
    if %ERRORLEVEL% neq 0 (
        echo Training run 1, iteration %%j failed.
        exit /b %ERRORLEVEL%
    )
)

for /L %%j in (1,1,%RUN_COUNT%) do (
    echo Running training run 2, iteration %%j with parameters: %PARAMS2%
    ~\anaconda3\envs\COS700-ML-1\python.exe ".\at_train copy.py" %PARAMS2%
    if %ERRORLEVEL% neq 0 (
        echo Training run 2, iteration %%j failed.
        exit /b %ERRORLEVEL%
    )
)

for /L %%j in (1,1,%RUN_COUNT%) do (
    echo Running training run 3, iteration %%j with parameters: %PARAMS3%
    ~\anaconda3\envs\COS700-ML-1\python.exe ".\at_train copy.py" %PARAMS3%
    if %ERRORLEVEL% neq 0 (
        echo Training run 3, iteration %%j failed.
        exit /b %ERRORLEVEL%
    )
)

for /L %%j in (1,1,%RUN_COUNT%) do (
    echo Running training run 4, iteration %%j with parameters: %PARAMS4%
    ~\anaconda3\envs\COS700-ML-1\python.exe ".\at_train copy.py" %PARAMS4%
    if %ERRORLEVEL% neq 0 (
        echo Training run 4, iteration %%j failed.
        exit /b %ERRORLEVEL%
    )
)

for /L %%j in (1,1,%RUN_COUNT%) do (
    echo Running training run 5, iteration %%j with parameters: %PARAMS5%
    ~\anaconda3\envs\COS700-ML-1\python.exe ".\at_train copy.py" %PARAMS5%
    if %ERRORLEVEL% neq 0 (
        echo Training run 5, iteration %%j failed.
        exit /b %ERRORLEVEL%
    )
)

echo All training runs completed successfully.
endlocal
pause
