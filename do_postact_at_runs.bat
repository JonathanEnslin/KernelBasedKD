set "DATA_DIR=final_runs"

set "KD=AT_Tuned"
set "ARGS=--num_workers 2 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir %DATA_DIR%  --kd_params kd_params.json --kd_set %KD% --teacher_path resnet56_cifar100_73p18.pth --teacher_type resnet56"

python kd_train.py %ARGS%
python kd_train.py %ARGS%
python kd_train.py %ARGS%
python kd_train.py %ARGS%
python kd_train.py %ARGS%

