#!/bin/bash
#PBS -N rst20_cfr10_5
#PBS -q serial
#PBS -l select=1:ncpus=10:mem=16gb
#PBS -l walltime=24:00:00
#PBS -o /mnt/lustre/users/jenslin/cpu_training_1/KernelBasedKD/job_out/run5.out
#PBS -e /mnt/lustre/users/jenslin/cpu_training_1/KernelBasedKD/job_out/run5.err
#PBS -P CSCI1166
#PBS -m abe
#PBS -M u19103345@tuks.co.za
ulimit -s unlimited

job_index=5
loop_count=3

echo "Start of resnet20 training run"

cd /mnt/lustre/users/jenslin/cpu_training_1/KernelBasedKD

module load chpc/python/anaconda/3-2021.11

echo '============= Installed packages ============='
pip list
echo '=============================================='

pwd -P
source .CpuTorchEnv/bin/activate

for i in $(seq 1 $loop_count)
do
    job_sub_index=$i
    run_name="resnet20_params3_CIFAR10_job${job_index}_${job_sub_index}"
    echo "============= Running Python script with run_name ${run_name} ============="
    python train.py \
        --run_name ${run_name} \
        --params ./params.json \
        --param_set params3 \
        --model_name resnet20 \
        --use_val \
        --val_size 0.1 \
        --disable_test \
        --early_stopping_patience 15 \
        --early_stopping_start_epoch 90 \
        --dataset CIFAR10 \
        --device cpu \
        --val_split_random_state 112 \
        --track_best_after_epoch 80
    echo '================================================='
done

echo "End of resnet20 training run"
