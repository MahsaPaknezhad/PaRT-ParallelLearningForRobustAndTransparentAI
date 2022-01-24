#!/bin/sh


conda create -n PaRT python=3.8 pytorch=1.7 tensorboard torchvision scikit-learn pillow matplotlib numpy psutil scipy tqdm 
conda activate PaRT
conda install -c jmcmurray json
pip install argparse

mkdir -p ../results
mkdir -p ../logs


conda activate PaRT
python testEnv.py || echo "Virtual environment was NOT made successfully."













