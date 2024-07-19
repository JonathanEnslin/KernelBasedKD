#!/bin/bash
#PBS -N cluster_test1_short
#PBS -q serial
#PBS -l select=1:ncpus=2:mem=4gb
#PBS -l walltime=00:02:00
#PBS -o /mnt/lustre/users/jenslin/firsttest/KernelBasedKD/job_out/test1.out
#PBS -e /mnt/lustre/users/jenslin/firsttest/KernelBasedKD/job_out/test1.err
#PBS -P CSCI1166
#PBS -m abe
#PBS -M u19103345@tuks.co.za
ulimit -s unlimited

echo 'Start of test_job1'

cd /mnt/lustre/users/jenslin/firsttest/KernelBasedKD

module load chpc/python/anaconda/3-2021.11

echo '============= Installed packages ============='
pip list
echo '=============================================='

pwd -P
source mytestenv/bin/activate

echo '============= Running Python script ============='
python dummyscript.py
echo '================================================='

echo 'End of test_job1'