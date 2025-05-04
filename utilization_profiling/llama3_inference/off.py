#!/usr/bin/env python3
import os
os.environ['HF_HOME'] = "../../datasets/model_cache/data"
os.environ["TRANSFORMERS_CACHE"] = "../../datasets/model_cache/data"
os.environ["VLLM_CACHE_DIR"] = "../../datasets/model_cache/data"
import json
import argparse
from vllm import LLM, SamplingParams

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run inference with Meta-Llama model")
    parser.add_argument("--output_len", type=int, default=5,
                        help="Maximum number of tokens to generate")
    args = parser.parse_args()

    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    batch_size = 1
    tensor_parallel_size = 1
    trust_remote_code = True
    dtype = "float16"
    enforce_eager = True
    gpu_memory_utilization = 0.9
    output_file = "inference_result.json"

    print(f"loading models: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
        download_dir="../../datasets/model_cache/data"  # Explicitly set cache directory
    )

    prompt = "Hello, my name is"
    sampling_params = SamplingParams(
        max_tokens=args.output_len,
        temperature=0.0
    )

    outputs = llm.generate(prompt, sampling_params)
    results = []

    for output in outputs:
        result = {
            "prompt": output.prompt,
            "generated_text": output.outputs[0].text,
            "num_tokens": len(output.outputs[0].token_ids)
        }
        results.append(result)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"gentext: {results[0]['generated_text']}")

if __name__ == "__main__":
    main()