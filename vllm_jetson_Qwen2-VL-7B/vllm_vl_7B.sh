#!/bin/bash
source /home/cailuyu/data/mambaforge/etc/profile.d/conda.sh
conda activate vllm
export HF_HUB_OFFLINE=1
vllm serve Qwen/Qwen2-VL-7B-Instruct --max-num-seqs 1 --max-model-len 4000 --gpu-memory-utilization 0.35 --enforce-eager
