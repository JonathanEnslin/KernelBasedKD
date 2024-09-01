set "PYTHON_PATH=C:/Users/jonat/anaconda3/envs/COS700-ML-1/python.exe"
set "DATA_DIR=kd_exp_runs"

set "ARGS=--num_workers 8 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir %DATA_DIR%  --use_val --val_size 0.2 --early_stopping_pat 9999 --early_stopping_start 9999 --disable_test"

"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%

set "KD=AT_b_1000"
set "ARGS=--num_workers 8 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir %DATA_DIR%  --use_val --val_size 0.2 --early_stopping_pat 9999 --early_stopping_start 9999 --disable_test --kd_params kd_params.json --kd_set %KD% --teacher_path resnet56_cifar100_73p18.pth --teacher_type resnet56"

"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%

set "KD=Vanilla"
set "ARGS=--num_workers 8 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir %DATA_DIR%  --use_val --val_size 0.2 --early_stopping_pat 9999 --early_stopping_start 9999 --disable_test --kd_params kd_params.json --kd_set %KD% --teacher_path resnet56_cifar100_73p18.pth --teacher_type resnet56"

"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%

set "KD=AT+Vanilla"
set "ARGS=--num_workers 8 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir %DATA_DIR%  --use_val --val_size 0.2 --early_stopping_pat 9999 --early_stopping_start 9999 --disable_test --kd_params kd_params.json --kd_set %KD% --teacher_path resnet56_cifar100_73p18.pth --teacher_type resnet56"

"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%

set "KD=kAT_2_2"
set "ARGS=--num_workers 8 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir %DATA_DIR%  --use_val --val_size 0.2 --early_stopping_pat 9999 --early_stopping_start 9999 --disable_test --kd_params kd_params.json --kd_set %KD% --teacher_path resnet56_cifar100_73p18.pth --teacher_type resnet56"

"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%

set "KD=kAT_0p4_2"
set "ARGS=--num_workers 8 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir %DATA_DIR%  --use_val --val_size 0.2 --early_stopping_pat 9999 --early_stopping_start 9999 --disable_test --kd_params kd_params.json --kd_set %KD% --teacher_path resnet56_cifar100_73p18.pth --teacher_type resnet56"

"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%

set "KD=kAT+Vanilla"
set "ARGS=--num_workers 8 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir %DATA_DIR%  --use_val --val_size 0.2 --early_stopping_pat 9999 --early_stopping_start 9999 --disable_test --kd_params kd_params.json --kd_set %KD% --teacher_path resnet56_cifar100_73p18.pth --teacher_type resnet56"

"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%

set "KD=kAT_No_Mean"
set "ARGS=--num_workers 8 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir %DATA_DIR%  --use_val --val_size 0.2 --early_stopping_pat 9999 --early_stopping_start 9999 --disable_test --kd_params kd_params.json --kd_set %KD% --teacher_path resnet56_cifar100_73p18.pth --teacher_type resnet56"

"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%

set "KD=kAT_Mean_C_in"
set "ARGS=--num_workers 8 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir %DATA_DIR%  --use_val --val_size 0.2 --early_stopping_pat 9999 --early_stopping_start 9999 --disable_test --kd_params kd_params.json --kd_set %KD% --teacher_path resnet56_cifar100_73p18.pth --teacher_type resnet56"

"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%

set "KD=kAT_Mean_C_out"
set "ARGS=--num_workers 8 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir %DATA_DIR%  --use_val --val_size 0.2 --early_stopping_pat 9999 --early_stopping_start 9999 --disable_test --kd_params kd_params.json --kd_set %KD% --teacher_path resnet56_cifar100_73p18.pth --teacher_type resnet56"

"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%