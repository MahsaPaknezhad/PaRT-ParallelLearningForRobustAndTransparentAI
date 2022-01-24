#!/bin/sh

#Baseline experiments

CURRENTDATE=`date +"%Y-%m-%d"`
LOG_FILE='../../logs/log_parallel_'$CURRENTDATE'.txt'

conda activate PaRT

mkdir -p ../results/Parallel/5Tasks

cd ../src
cd 5Tasks_Parallel

echo 'generating model'
python generate_model.py --json_file=$1
echo 'generate paths'
python generate_path.py --json_file=$1
echo 'start training'
python 5tasks.py --json_file=$1 2>&1 | tee ${LOG_FILE} 

