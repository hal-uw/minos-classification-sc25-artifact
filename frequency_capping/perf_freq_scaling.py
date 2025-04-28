import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib
from matplotlib.cm import get_cmap
import json
import time
import math
import re
import ast

# Style Configuration
MEDIUM_SIZE = 20
MARKER_SIZE = 15
plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rc('lines', markersize=MARKER_SIZE)
plt.rc('lines', linewidth=2)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


MARKER_EVERY = 100    

def extract_latency_vllm(filename):
    with open(filename, 'r') as file:
        for line in file:
            if "Avg latency:" in line:
                parts = line.strip().split("Avg latency:")
                if len(parts) > 1:
                    try:
                        latency = float(parts[1].strip().split()[0])
                        return latency
                    except ValueError:
                        pass  
    return None  # Return None if not found or parsing fails


def extract_throughputs(filename):
    pattern = re.compile(
        r'Avg prompt throughput:\s*([\d.]+)\s*tokens/s,\s*Avg generation throughput:\s*([\d.]+)\s*tokens/s'
    )
    matches = []
    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                matches.append((float(match.group(1)), float(match.group(2))))
    return matches[1] if len(matches) > 1 else (None, None)

def extract_latency_deepmd(filename):
    durations = []
    pattern = re.compile(r"total wall time\s*=\s*([0-9.]+)\s*s")

    with open(filename, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                try:
                    durations.append(float(match.group(1)))
                except ValueError:
                    pass

    if durations:
        return sum(durations) / len(durations)


def extract_latency_lsms(filename):
    with open(filename, 'r') as file:
        for line in file:
            if "LSMS Runtime =" in line:
                parts = line.strip().split("LSMS Runtime =")
                if len(parts) > 1:
                    try:
                        latency = float(parts[1].strip().split()[0])
                        return latency
                    except ValueError:
                        pass  # Ignore if it fails to convert
    return None

def extract_latency_gunrock(file_path):
    pattern = r"GPU Elapsed Time\s*:\s*([\d.]+)\s*\(ms\)"
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                return float(match.group(1))
    return None  # Return None if no match is found

def extract_latency_bfs(file_path):
    pattern = r"GPU Elapsed Time\s*:\s*([\d.]+)\s*\(ms\)"
    times = []
    for trial in range(5):
        with open(f"{file_path}_{trial}_stdout", 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    times.append(float(match.group(1)))

    if times:
        return sum(times) / len(times)
    
    return None  # Return None if no match is found

def extract_latency_openfold(filename):
    durations = []
    
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("training"):
                try:
                    # Extract the dictionary part from the line
                    dict_str = line.strip().split("training", 1)[1].strip()
                    log_data = ast.literal_eval(dict_str)
                    if 'duration' in log_data:
                        durations.append(float(log_data['duration']))
                except (ValueError, SyntaxError):
                    pass  # Skip lines that don't parse properly
    
    if durations:
        return sum(durations) / len(durations)
    return None


freq_data = {
    'app': [],
    'freq': [],
    'latency': [],
    'exec_time_inc_percent': []
}


apps = [
    "openfold",
    "deepmd",
    # "gunrock"
    # "llama2_bsz1",
    # "llama2_bsz32",
    # "llama3_bsz1",
    # "llama3_bsz32",
    # "lsms",
    # "gunrock"
]

mapps = [
    "lsms"
    "sssp_kron",
    "bfs_indo",
]

applabels = {
    "openfold" : "OpenFold",
    "deepmd"   : "DeepMD",
    "gunrock"  : "Gunrock PageRank \n(indochina)",
    "lsms"     : "LSMS",   
    "sssp_kron": "Gunrock SSSP (kron)",
    "bfs_indo" : "Gunrock BFS (indochina)",
}


for freq in range(1300,2200,100):
    for idx, app in enumerate(apps):
        latency = 0.0
        if app == "lsms":
            latency = extract_latency_lsms(f"{app}/{app}_{freq}_stdout")
            baseline_latency = extract_latency_lsms(f"{app}/{app}_2100_stdout")
        elif app == "deepmd":
            latency = extract_latency_deepmd(f"{app}/{app}_{freq}_stdout")
            baseline_latency = extract_latency_deepmd(f"{app}/{app}_2100_stdout")
        elif app == "openfold":
            latency = extract_latency_openfold(f"{app}/{app}_{freq}_stdout")
            baseline_latency = extract_latency_openfold(f"{app}/{app}_2100_stdout")
        elif app in ["llama2_bsz1", "llama2_bsz32", "llama3_bsz1", "llama3_bsz32"]:
            #latency = extract_latency_vllm(f"{app}/{app}_{freq}_stdout")
            prompt_throughput, token_throughput = extract_throughputs(f"{app}/{app}_{freq}_stdout")
            prompt_baseline, token_baseline = extract_throughputs(f"{app}/{app}_2100_stdout")
            prompt_per = 100*(prompt_baseline - prompt_throughput)/prompt_baseline
            token_per = 100*(token_baseline - token_throughput)/token_baseline
            freq_data['app'].append(f"{app}_prompt")
            freq_data['freq'].append(freq)
            freq_data['latency'].append(prompt_throughput)
            freq_data['exec_time_inc_percent'].append(prompt_per)
            freq_data['app'].append(f"{app}_token")
            freq_data['freq'].append(freq)
            freq_data['latency'].append(token_throughput)
            freq_data['exec_time_inc_percent'].append(token_per)
            continue
        elif app == "gunrock":
            latency = extract_latency_gunrock(f"{app}/{app}_{freq}_stdout")
        elif "bfs" in app or "sssp" in app:
            latency = extract_latency_bfs(f"{app}/{app}_{freq}")
            
        per_inc = 100*(latency - baseline_latency) / baseline_latency
        freq_data['app'].append(app)
        freq_data['freq'].append(freq)
        freq_data['latency'].append(latency)
        freq_data['exec_time_inc_percent'].append(per_inc)

df = pd.DataFrame(freq_data)
df.to_csv("latency_data.csv",index=False)   

fig, ax = plt.subplots(1,3, figsize=(12,5),sharey=True)

for idx, app in enumerate(apps):
    row_idx = idx // 3
    col_idx = idx % 3

    if "llama" in app:
        df_prompt = df[df['app'] == f"{app}_prompt"]
        df_token = df[df['app'] == f"{app}_token"]
        baseline_prompt = df_prompt.loc[df['freq'] == 2100, 'latency'].values[0]
        baseline_token = df_token.loc[df['freq'] == 2100, 'latency'].values[0]
        df_prompt['prompt_delta'] = ((baseline_prompt - df_prompt['latency'])  / baseline_prompt) * 100
        df_token['token_delta']  = ((baseline_token - df_token['latency']) / baseline_token)
        ax[idx].plot(df_prompt['freq'], df_prompt['prompt_delta'], marker='o', label='prefill', color='#fb8072')
        ax[idx].plot(df_token['freq'], df_token['token_delta'], marker='x', label='decode', color='#fb8072')
        ax[idx].legend()
    else:
        df_app = df[df['app'] == app]
        baseline_latency = df_app.loc[df['freq'] == 2100, 'latency'].values[0]
        df_app['percent_execution_time'] = ((df_app['latency'] - baseline_latency)  / baseline_latency) * 100
        ax[idx].plot(
            df_app['freq'], 
            df_app['percent_execution_time'], 
            marker='o', 
            color='#fb8072')

    ax[idx].set_title(applabels[app])
    ax[idx].set_xlabel('Frequency (MHz)')
    ax[idx].set_ylabel('% Increase in Exec Time')
    ax[idx].grid(True)
    ax[idx].set_xticks(range(1300, 2200, 100), minor=True)
    #ax[idx].set_ylim([0,40])

plt.tight_layout()

plt.savefig("latency_compute.pdf")


