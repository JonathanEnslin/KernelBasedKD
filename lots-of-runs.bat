@echo off
setlocal
set startTime=%time%
set failedLoops=

echo Starting batch execution...

@REM for /L %%i in (1,1,3) do (
@REM     @echo Running set 1 - iteration %%i
@REM     C:/Users/jonat/anaconda3/envs/COS700-ML-1/python.exe kd_train.py --num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir final_runs --kd_params kd_params.json --kd_set kAT_Tuned_b_2000 --teacher_path resnet20_cifar100_71p25.pth --teacher_type resnet20
@REM     if errorlevel 1 (
@REM         set failedLoops=!failedLoops! Loop 1 - iteration %%i failed with kAT_Tuned_b_2000
@REM     )
@REM )

@REM for /L %%i in (1,1,3) do (
@REM     @echo Running set 2 - iteration %%i
@REM     C:/Users/jonat/anaconda3/envs/COS700-ML-1/python.exe kd_train.py --num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir final_runs --kd_params kd_params.json --kd_set kAT_Tuned_b_2000_all --teacher_path resnet20_cifar100_71p25.pth --teacher_type resnet20
@REM     if errorlevel 1 (
@REM         set failedLoops=!failedLoops! Loop 2 - iteration %%i failed with kAT_Tuned_b_2000_all
@REM     )
@REM )

@REM for /L %%i in (1,1,3) do (
@REM     @echo Running set 3 - iteration %%i
@REM     C:/Users/jonat/anaconda3/envs/COS700-ML-1/python.exe kd_train.py --num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir final_runs --kd_params kd_params.json --kd_set kAT_Tuned_b_2000_no_agg --teacher_path resnet20_cifar100_71p25.pth --teacher_type resnet20
@REM     if errorlevel 1 (
@REM         set failedLoops=!failedLoops! Loop 3 - iteration %%i failed with kAT_Tuned_b_2000_no_agg
@REM     )
@REM )

@REM for /L %%i in (1,1,3) do (
@REM     @echo Running set 4 - iteration %%i
@REM     C:/Users/jonat/anaconda3/envs/COS700-ML-1/python.exe kd_train.py --num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir final_runs --kd_params kd_params.json --kd_set kAT_Tuned_b_2000_no_agg_no_abs --teacher_path resnet20_cifar100_71p25.pth --teacher_type resnet20
@REM     if errorlevel 1 (
@REM         set failedLoops=!failedLoops! Loop 4 - iteration %%i failed with kAT_Tuned_b_2000_no_agg_no_abs
@REM     )
@REM )

@REM for /L %%i in (1,1,3) do (
@REM     @echo Running set 5 - iteration %%i
@REM     C:/Users/jonat/anaconda3/envs/COS700-ML-1/python.exe kd_train.py --num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir final_runs --kd_params kd_params.json --kd_set Vanilla_Tuned --teacher_path resnet20_cifar100_71p25.pth --teacher_type resnet20
@REM     if errorlevel 1 (
@REM         set failedLoops=!failedLoops! Loop 5 - iteration %%i failed with Vanilla_Tuned
@REM     )
@REM )

@REM for /L %%i in (1,1,3) do (
@REM     @echo Running set 6 - iteration %%i
@REM     C:/Users/jonat/anaconda3/envs/COS700-ML-1/python.exe kd_train.py --num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir final_runs --kd_params kd_params.json --kd_set Vanilla_Tuned --teacher_path resnet20_cifar100_71p25.pth --teacher_type resnet20
@REM     if errorlevel 1 (
@REM         set failedLoops=!failedLoops! Loop 6 - iteration %%i failed with Vanilla_Tuned
@REM     )
@REM )

@REM for /L %%i in (1,1,3) do (
@REM     @echo Running set 8 - iteration %%i
@REM     C:/Users/jonat/anaconda3/envs/COS700-ML-1/python.exe kd_train.py --num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir final_runs --kd_params kd_params.json --kd_set kAT+Vanilla_Tuned --teacher_path resnet20_cifar100_71p25.pth --teacher_type resnet20
@REM     if errorlevel 1 (
@REM         set failedLoops=!failedLoops! Loop 8 - iteration %%i failed with kAT+Vanilla_Tuned
@REM     )
@REM )

@REM for /L %%i in (1,1,3) do (
@REM     @echo Running set 9 - iteration %%i
@REM     C:/Users/jonat/anaconda3/envs/COS700-ML-1/python.exe kd_train.py --num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir final_runs --kd_params kd_params.json --kd_set AT+Vanilla --teacher_path resnet20x4_cifar100_75p02.pth --teacher_type resnet20x4
@REM     if errorlevel 1 (
@REM         set failedLoops=!failedLoops! Loop 9 - iteration %%i failed with AT+Vanilla
@REM     )
@REM )

@REM for /L %%i in (1,1,3) do (
@REM     @echo Running set 10 - iteration %%i
@REM     C:/Users/jonat/anaconda3/envs/COS700-ML-1/python.exe kd_train.py --num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir final_runs --kd_params kd_params.json --kd_set kAT_Tuned_b_2000_all --teacher_path resnet20x4_cifar100_75p02.pth --teacher_type resnet20x4
@REM     if errorlevel 1 (
@REM         set failedLoops=!failedLoops! Loop 10 - iteration %%i failed with kAT_Tuned_b_2000_all
@REM     )
@REM )


for /L %%i in (1,1,1) do (
    @echo Running set 4 - iteration %%i
    C:/Users/jonat/anaconda3/envs/COS700-ML-1/python.exe kd_train.py --num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir final_runs --kd_params kd_params.json --kd_set kAT+Vanilla_no_agg_no_abs --teacher_path resnet20_cifar100_71p25.pth --teacher_type resnet20
    if errorlevel 1 (
        set failedLoops=!failedLoops! Loop 4 - iteration %%i failed with kAT_Tuned_b_2000_no_agg_no_abs
    )
)


@REM Calculate and print the total time elapsed
set endTime=%time%
set /A elapsedSeconds=((%endTime:~0,2%*3600 + %endTime:~3,2%*60 + %endTime:~6,2%) - (%startTime:~0,2%*3600 + %startTime:~3,2%*60 + %startTime:~6,2%))
echo Batch execution completed in %elapsedSeconds% seconds.

@REM Print any failed loops at the end
if defined failedLoops (
    echo The following loops encountered errors:
    echo %failedLoops%
) else (
    echo All iterations completed successfully!
)

endlocal
