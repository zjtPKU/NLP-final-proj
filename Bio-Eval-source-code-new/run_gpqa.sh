#!/bin/bash

set -x
# Prepare repository and environment
# git clone https://github.com/KOR-Bench/KOR-Bench.git
cd /map-vepfs/xinrun/KOR-Bench
# pip install -r requirements.txt
source /map-vepfs/miniconda3/bin/activate vllm

# Set HF_HOME to custom folder
# export HF_HOME=$(pwd)/cache


# gpt-4o-2024-08-06 / QwQ-32B-Preview / claude-3-5-sonnet-20241022 / o1-mini / gemini-1.5-pro-002 / DeepSeek-V2.5

export PYTHONPATH=$(pwd)
model_name=$1
# Run inference with openai api examples
case $model_name in
    "gpt-4o-2024-08-06"|"claude-3-5-sonnet-20241022"|"o1-mini"|"gemini-1.5-pro-002"|"DeepSeek-V2.5"|"doubao-pro-128k")
        # python infer/infer.py --config config/config_gpqa.yaml --split GPQA-data-processed-by-lnn-20241211-version2 --mode zero-shot --model_name $model_name --output_dir results/gpqa --num_worker 64
        # python infer/infer.py --config config/config_gpqa.yaml --split filtered-dk-data-toppr-struct-physics-chem-bio-2nd-filter --mode zero-shot --model_name $model_name --output_dir results/gpqa --num_worker 64
        python infer/infer.py --config config/config_gpqa.yaml --split toprp-with-confusion-options --mode zero-shot --model_name $model_name --output_dir results/gpqa --num_worker 64
        ;;
    "QwQ-32B-Preview")
        # python infer/infer.py --config config/config_gpqa.yaml --split GPQA-data-processed-by-lnn-20241211-version2 --mode zero-shot --model_name $model_name --output_dir results/gpqa --batch_size 1000 --use_accel --index $2 --world_size $3
        # python infer/infer.py --config config/config_gpqa.yaml --split filtered-dk-data-toppr-struct-physics-chem-bio-2nd-filter --mode zero-shot --model_name $model_name --output_dir results/gpqa --batch_size 1000 --use_accel --index $2 --world_size $3
        # python infer/infer.py --config config/config_gpqa.yaml --split only_one_correct_at_most_samples_filtered --mode gen_confusion_options --model_name $model_name --output_dir results/gpqa --batch_size 1000 --use_accel --index $2 --world_size $3
        python infer/infer.py --config config/config_gpqa.yaml --split toprp-with-confusion-options --mode zero-shot --model_name $model_name --output_dir results/gpqa --batch_size 1000 --use_accel --index $2 --world_size $3
        ;;
    "Qwen2.5-72B-Instruct")
        # python infer/infer.py --config config/config_gpqa.yaml --split all_correct_samples --mode tag-difficulty --model_name $model_name --output_dir results/gpqa/filter --batch_size 1000 --use_accel --index $2 --world_size $3
        # python infer/infer.py --config config/config_gpqa.yaml --split dk-data-toppr-struct-physics-chem-bio-2nd-filter --mode gpqa-filter --model_name $model_name --output_dir results/gpqa/filter --batch_size 1000 --use_accel --index $2 --world_size $3
        # python infer/infer.py --config config/config_gpqa.yaml --split filtered-dk-data-toppr-struct-physics-chem-bio-2nd-filter --mode zero-shot --model_name $model_name --output_dir results/gpqa --batch_size 1000 --use_accel --index $2 --world_size $3
        python infer/infer.py --config config/config_gpqa.yaml --split only_one_correct_at_most_samples_filtered --mode gen_confusion_options --model_name $model_name --output_dir results/gpqa --batch_size 1000 --use_accel --index $2 --world_size $3
        ;;
esac
# sleep 5

# Run evaluation
# python eval/eval.py results eval/results eval/results.csv
