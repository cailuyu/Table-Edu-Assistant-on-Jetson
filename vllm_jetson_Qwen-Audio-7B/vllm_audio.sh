#!/bin/bash
source /home/seeed/anaconda3/etc/profile.d/conda.sh
conda activate vllm
export HF_HUB_OFFLINE=1
vllm serve Qwen/Qwen2-Audio-7B-Instruct --max-num-seqs 1 --max-model-len 5500 --gpu-memory-utilization 0.7 --enforce-eager
