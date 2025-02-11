export HF_HUB_OFFLINE=1
vllm serve Qwen/Qwen2-Audio-7B --max-num-seqs 3 --max-model-len 5500 --gpu-memory-utilization 0.6 --enforce-eager
