set "PYTHON_PATH=C:/Users/jonat/anaconda3/envs/COS700-ML-1/python.exe"

set "DATA_DIR=fat_initial_runs"

set "ARGS=--num_workers 2 --params params.json --param_set params3 --model_name resnet20 --checkpoint_freq 99999 --dataset CIFAR100 --device cuda --output_data_dir %DATA_DIR%"

"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%
"%PYTHON_PATH%" kd_train.py %ARGS%


