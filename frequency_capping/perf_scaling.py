import subprocess

# List of your scripts
scripts = {
        'compute': "perf_freq_scaling.py",
        'memory' : "perf_freq_scaling_mem.py",
        'hybrid' : "perf_freq_scaling_hybrid.py",
}

for key, script in scripts.items():
    print(f"Plotting Perf Scaling for {key}...")
    try:
        subprocess.run(["python3", script], check=True)
        print(f"{key} workloads: SUCCESS")
    except subprocess.CalledProcessError:
        print(f"{key} workloads: FAILED")
