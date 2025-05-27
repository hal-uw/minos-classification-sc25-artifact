#!/bin/bash

while getopts m:g:d:b:i:o: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        g) numgpu=${OPTARG};;
        d) datatype=${OPTARG};;
        b) batch_size=${OPTARG};;
        i) input_length=${OPTARG};;
        o) output_length=${OPTARG};;
    esac
done

# Set default values if not provided
batch_size=${batch_size:-8}
input_length=${input_length:-2048}
output_length=${output_length:-128}

echo "MODEL: $model"
echo "Batch size: $batch_size"
echo "Input length: $input_length"
echo "Output length: $output_length"

# Parse model name
model_org_name=(${model//// })
model_name=${model_org_name[1]}
tp=$numgpu

# Set environment and options
export VLLM_USE_TRITON_FLASH_ATTN=0

# Set backend based on GPU count
if [ $tp -eq 1 ]; then
    DIST_BE=" --enforce-eager "
else
    DIST_BE=" --distributed-executor-backend mp "
fi

# Set datatype options
if [[ $datatype == "float16" ]]; then
    DTYPE=" --dtype float16 "
elif [[ $datatype == "float8" ]]; then
    DTYPE=" --dtype float16 --quantization fp8 --kv-cache-dtype fp8 "
fi

# Benchmark settings
OPTION_LATENCY=" --gpu-memory-utilization 0.9 --enforce-eager "
n_warm=3
n_itr=5

# Create output directories
report_dir="reports_${datatype}"
report_summary_dir="${report_dir}/summary"
mkdir -p $report_dir
mkdir -p $report_summary_dir

# Define tool paths
tool_latency="/app/vllm/benchmarks/benchmark_latency.py"
tool_report="vllm_benchmark_report.py"

# Run latency benchmark
outjson=${report_dir}/${model_name}_latency_decoding_bs${batch_size}_in${input_length}_out${output_length}_${datatype}.json
outcsv=${report_summary_dir}/${model_name}_latency_report.csv

python3 $tool_latency \
    --model $model \
    --batch-size $batch_size \
    -tp $tp \
    --input-len $input_length \
    --output-len $output_length \
    --num-iters-warmup $n_warm \
    --num-iters $n_itr \
    --trust-remote-code \
    --output-json $outjson \
    $DTYPE $DIST_BE $OPTION_LATENCY