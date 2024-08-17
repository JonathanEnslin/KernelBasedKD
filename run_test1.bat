set "PYTHON_PATH=C:/Users/jonat/anaconda3/envs/COS700-ML-1/python.exe"

set "DATA_DIR=fat_initial_runs"

set "ARGS=--num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --run_tag fAT0p4 --output_data_dir %DATA_DIR% --teacher_type resnet56 --teacher_path resnet56_cifar100_73p18.pth --kd_params kd_params.json --kd_set FAT_0p4_0p4 --use_cached_logits"

"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%

@REM set "ARGS=--num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --run_tag Vanilla --output_data_dir %DATA_DIR% --teacher_type resnet56 --teacher_path resnet56_cifar100_73p18.pth --kd_params kd_params.json --kd_set Vanilla --use_cached_logits"

@REM "%PYTHON_PATH%" kd_train.py %ARGS%
@REM "%PYTHON_PATH%" kd_train.py %ARGS%
@REM "%PYTHON_PATH%" kd_train.py %ARGS%
@REM "%PYTHON_PATH%" kd_train.py %ARGS%

@REM set "ARGS=--num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --run_tag stepper1 --output_data_dir %DATA_DIR% --teacher_type resnet56 --teacher_path resnet56_cifar100_73p18.pth --kd_params kd_params.json --kd_set FAT_2_2 --use_cached_logits --batch_stepper sine_modulated_beta --batch_stepper_args period=8,amplitude=1.0,vertical_shift=0.0,through_relu=True"

@REM "%PYTHON_PATH%" kd_train.py %ARGS%
@REM "%PYTHON_PATH%" kd_train.py %ARGS%
@REM "%PYTHON_PATH%" kd_train.py %ARGS%
@REM "%PYTHON_PATH%" kd_train.py %ARGS%

@REM set "ARGS=--num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --run_tag stepper2 --output_data_dir %DATA_DIR% --teacher_type resnet56 --teacher_path resnet56_cifar100_73p18.pth --kd_params kd_params.json --kd_set FAT_2_2 --use_cached_logits --batch_stepper sine_modulated_beta --batch_stepper_args period=100,amplitude=1.0,vertical_shift=1.0,through_relu=False"

@REM "%PYTHON_PATH%" kd_train.py %ARGS%
@REM "%PYTHON_PATH%" kd_train.py %ARGS%
@REM "%PYTHON_PATH%" kd_train.py %ARGS%
@REM "%PYTHON_PATH%" kd_train.py %ARGS%

@REM set "ARGS=--num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --run_tag stepper3 --output_data_dir %DATA_DIR% --teacher_type resnet56 --teacher_path resnet56_cifar100_73p18.pth --kd_params kd_params.json --kd_set FAT_2_2 --use_cached_logits --batch_stepper sine_modulated_beta --batch_stepper_args period=6,amplitude=1.0,vertical_shift=0.5,through_relu=False"

@REM "%PYTHON_PATH%" kd_train.py %ARGS%
@REM "%PYTHON_PATH%" kd_train.py %ARGS%
@REM "%PYTHON_PATH%" kd_train.py %ARGS%
@REM "%PYTHON_PATH%" kd_train.py %ARGS%

@REM set "ARGS=--num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --run_tag stepper_neg --output_data_dir %DATA_DIR% --teacher_type resnet56 --teacher_path resnet56_cifar100_73p18.pth --kd_params kd_params.json --kd_set FAT_2_2 --use_cached_logits --batch_stepper sine_modulated_beta --batch_stepper_args period=6,amplitude=1.0,vertical_shift=-1.0,through_relu=False"

@REM "%PYTHON_PATH%" kd_train.py %ARGS%
@REM "%PYTHON_PATH%" kd_train.py %ARGS%
@REM "%PYTHON_PATH%" kd_train.py %ARGS%


@REM set "ARGS=--num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --run_tag wot --output_data_dir %DATA_DIR% --teacher_type resnet56 --teacher_path resnet56_cifar100_73p18.pth --kd_params kd_params.json --kd_set wot --use_cached_logits"

@REM "%PYTHON_PATH%" kd_train.py %ARGS%
@REM "%PYTHON_PATH%" kd_train.py %ARGS%


