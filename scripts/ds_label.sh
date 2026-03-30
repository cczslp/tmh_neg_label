python main.py --input data/video_info_label.csv \
    --column caption --max-items 10000 --batch-size 10 \
    --output output/ks_ds_label.csv \
    --method api \
    --base-url https://api.siliconflow.cn/v1 \
    --model deepseek-ai/DeepSeek-V3.2 > logs/ks_ds_label.log 2>&1

python main.py --input data/douyin \
    --column 视频标题 --max-items 40000 --batch-size 10 \
    --output output/douyin_ds_label.csv \
    --method api \
    --base-url https://api.siliconflow.cn/v1 \
    --model deepseek-ai/DeepSeek-V3.2 > logs/douyin_ds_label.log 2>&1

# ds v2 test
python main.py --input output/ks_ds_test.xlsx \
    --column title --max-items 10000 --batch-size 10 \
    --output output/ks_ds_v2_test_label.xlsx \
    --method api \
    --base-url https://api.siliconflow.cn/v1 \
    --model deepseek-ai/DeepSeek-V3.2 > logs/ks_ds_v2_test_label.log 2>&1