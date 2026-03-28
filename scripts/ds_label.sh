python main.py --input data/video_info_label.csv \
    --column caption --max-items 10000 --batch-size 10 \
    --output output/ks_ds_label.csv \
    --base-url https://api.siliconflow.cn/v1 \
    --model deepseek-ai/DeepSeek-V3.2 > logs/ks_ds_label.log 2>&1 &