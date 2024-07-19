#!/bin/bash
#PBS -N rst20_cfr10
#PBS -q serial
#PBS -l select=1:ncpus=10:mem=16gb
#PBS -l walltime=24:00:00
#PBS -o /mnt/lustre/users/jenslin/cpu_training_1/KernelBasedKD/job_out/run3.out
#PBS -e /mnt/lustre/users/jenslin/cpu_training_1/KernelBasedKD/job_out/run3.err
#PBS -P CSCI1166
#PBS -m abe
#PBS -M u19103345@tuks.co.za
ulimit -s unlimited

echo 'Start of resnet20 training run 3'

cd /mnt/lustre/users/jenslin/cpu_training_1/KernelBasedKD

module load chpc/python/anaconda/3-2021.11

echo '============= Installed packages ============='
pip list
echo '=============================================='

pwd -P
source .CpuTorchEnv/bin/activate

echo '============= Running Python script ============='
python train.py --run_name resnet20_params3_CIFAR10_run3 --params ./params.json --param_set params3 --model_name resnet20 --use_val --val_size 0.1 --disable_test --early_stopping_patience 15 --early_stopping_start_epoch 90 --dataset CIFAR10 --device cpu --val_split_random_state 112 --track_best_after_epoch 80
echo '================================================='

echo 'End of test_job1'