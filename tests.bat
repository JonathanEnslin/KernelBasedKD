@REM python kd_train.py --num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir conf_runs --kd_params kd_params.json --kd_set AT+Vanilla_Tuned --teacher_path resnet56_cifar100_73p18.pth --teacher_type resnet56

python kd_train.py --num_workers 0 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir debug_runs --kd_params kd_params.json --kd_set AT_b_1000 --teacher_path resnet56_cifar100_73p18.pth --teacher_type resnet56

@REM python kd_train.py --num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir conf_runs --kd_params kd_params.json --kd_set AT+Vanilla --teacher_path resnet56_cifar100_73p18.pth --teacher_type resnet56 --use_val --val_size 0.2 --disable_val_until_epoch 0

@REM python kd_train.py --num_workers 4 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir conf_runs --kd_params kd_params.json --kd_set AT+Vanilla_Tuned --teacher_path resnet56_cifar100_73p18.pth --teacher_type resnet56 --use_val --val_size 0.2 --disable_val_until_epoch 0

@REM python at_test_file.py --dataset CIFAR100 --name "resnet20_cifar100_at_test_run" --device cuda --num-workers 4 --persistent-workers
