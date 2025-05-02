export TRANSFORMERS_CACHE=../../datasets/model_cache/data
export HF_HOME=../../datasets/model_cache
############ todo  need to create a new apptainer with ncu installed!!!!!!
SIF_IMAGE="/work/10009/yiwei357/ls6/vllm-openai_v0.6.1.sif"
NCU="ncu -f ./llama3_inf --target-processes all  --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed --csv --devices 0 "
CMD="timeout -s SIGINT 24h $NCU python3 off.py --output_len 5"

apptainer exec --nv $SIF_IMAGE bash -c "$CMD"