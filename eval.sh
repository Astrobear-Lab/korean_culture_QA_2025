#!/usr/bin/env bash
set -e

# Phase 1: Prompting Experiment for Korean Culture QA
model_list=(
#    "skt/A.X-4.0-Light"
#    "kakaocorp/kanana-1.5-8b-instruct-2505"
   "Qwen/Qwen3-8B"
)


for model in "${model_list[@]}"
do
    echo "Running model: $model"
    nohup python run_phase1.py --model "$model" >> run_phase1.log
done