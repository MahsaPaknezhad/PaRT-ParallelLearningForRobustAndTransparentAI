#!/bin/sh

#Baseline experiments

CURRENTDATE=`date +"%Y-%m-%d"`
LOG_FILE='../../logs/log_baseline_'$CURRENTDATE'.txt'

conda activate PaRT

mkdir -p ../results/Baseline/CIFAR100

cd ../src
cd CIFAR_Baseline

echo 'generating model'
python generate_model.py --json_file=$1
echo 'generate paths'
python generate_path.py --json_file=$1
echo 'start training'
python cifar.py --json_file=$1 2>&1 | tee ${LOG_FILE}






