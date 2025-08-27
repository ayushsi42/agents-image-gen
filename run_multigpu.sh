#!/bin/bash

BENCHMARK="eval_benchmark/GenAIBenchmark/genai_image_seed.json"
TOTAL_PARTS=8

for PART in $(seq 0 $((TOTAL_PARTS-1))); do
    CUDA_VISIBLE_DEVICES=$PART \
    python main_negative_prompt_multi.py \
        --benchmark_name $BENCHMARK \
        --part $PART \
        --total_parts $TOTAL_PARTS \
        "$@" \
        > run_part${PART}.log 2>&1 &
done

wait
echo "All 8 processes finished."
