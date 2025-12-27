#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=1
export PYTHONWARNINGS="ignore"

CFG="train_configs/gemkr_finetune.yaml"

CUDA_VISIBLE_DEVICES=4 python train.py --cfg-path $CFG
