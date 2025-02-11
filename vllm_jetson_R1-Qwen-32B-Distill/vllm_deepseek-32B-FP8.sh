export HF_HUB_OFFLINE=1
vllm serve getfit/DeepSeek-R1-Distill-Qwen-32B-FP8-Dynamic --max-num-seqs 1 --max-model-len 10000 --gpu-memory-utilization 0.75
