# 仅本机进程访问，绑定回环地址
export CUDA_VISIBLE_DEVICES=4,5,6,7

python -m vllm.entrypoints.openai.api_server \
  --model /data/huangbeining/qwen2.5-7b-it \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name qwen2.5-7b-it > logs/vllm_deploy_qwen2.5_7b_it_01.log 2>&1 &