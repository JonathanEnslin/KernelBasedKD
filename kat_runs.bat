@echo off
setlocal enabledelayedexpansion

:: Set variables
set DATA_DIR=final_runs
set KD=kAT+Vanilla_Tuned
set ARGS=--num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir %DATA_DIR% --kd_params kd_params.json --kd_set %KD% --teacher_path resnet56_cifar100_73p18.pth --teacher_type resnet56

:: Run the Python command multiple times
for /l %%i in (1, 1, 15) do (
    echo Running loop %%i
    python kd_train.py %ARGS%
)


:: Set variables
set DATA_DIR=final_runs
set KD=kAT_Tuned
set ARGS=--num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir %DATA_DIR% --kd_params kd_params.json --kd_set %KD% --teacher_path resnet56_cifar100_73p18.pth --teacher_type resnet56

:: Run the Python command multiple times
for /l %%i in (1, 1, 15) do (
    echo Running loop %%i
    python kd_train.py %ARGS%
)


endlocal
