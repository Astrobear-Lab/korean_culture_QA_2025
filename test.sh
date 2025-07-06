#!/usr/bin/env bash
set -e

# Phase 1: Prompting Experiment for Korean Culture QA
model_list=(
#    "skt/A.X-4.0-Light"
#    "kakaocorp/kanana-1.5-8b-instruct-2505"
#    "Qwen/Qwen3-8B"
    "/workspace/korean_culture_QA_2025/models/grpo_v1_A.X-4.0-Light/checkpoint-95_merged"
)


for model in "${model_list[@]}"
do
    echo "Running model: $model"
    nohup python run_phase1.py --model "$model" --use_test > test.log
done

# Phase 2: Scoring the Answers
python score_only_answer.py --json_path /workspace/korean_culture_QA_2025/results/phase1_checkpoint-95_merged_test_outputs.json