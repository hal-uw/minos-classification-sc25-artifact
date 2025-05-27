import pandas as pd
import json
import os
from io import StringIO


def process_cuda_file(filepath):
    # Read entire file content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find CSV header line
    header1 = '"ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"'
    header2 = '"ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream", "Block Size", "Grid Size", "Device", "CC", "Section Name", "Metric Name", "Metric Unit", "Metric Value"'
    # header_pos = content.find(header)
    header_pos = content.find(header1)
    if header_pos == -1:
        header_pos = content.find(header2)
    if header_pos == -1:
        print(f"Header not found in {filepath}")
        return None

    # Extract content starting from header
    csv_content = content[header_pos:]

    # Parse CSV using StringIO
    df = pd.read_csv(StringIO(csv_content))

    # Get time duration data
    time_data = df[df['Metric Name'] == 'gpu__time_duration.sum']
    sm_data = df[df['Metric Name'] ==
                 'sm__throughput.avg.pct_of_peak_sustained_elapsed']
    dram_data = df[df['Metric Name'] ==
                   'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed']

    if time_data.empty:
        print(f"No time data found in {filepath}")
        return None

    # Create kernel name to time mapping
    kernel_times = dict(
        zip(time_data['Kernel Name'], time_data['Metric Value']))

    # Calculate weighted average for all kernels
    total_time = sum(kernel_times.values())

    sm_weighted = sum(row['Metric Value'] * kernel_times.get(row['Kernel Name'], 0)
                      for _, row in sm_data.iterrows()) / total_time

    dram_weighted = sum(row['Metric Value'] * kernel_times.get(row['Kernel Name'], 0)
                        for _, row in dram_data.iterrows()) / total_time

    return {
        'Compute (SM) Throughput': round(sm_weighted, 2),
        'DRAM Throughput': round(dram_weighted, 2)
    }


def process_files(file_list):
    all_results = {}

    for filepath in file_list:
        # Use filename without extension as app name
        app_name = os.path.splitext(os.path.basename(filepath))[0]
        try:
            results = process_cuda_file(filepath)
            if results:
                all_results[app_name] = results
                print(
                    f"Processed: {app_name} - SM: {results['sm_throughput']}%, DRAM: {results['dram_throughput']}%")
        except Exception as e:
            print(f"Error processing {app_name}: {e}")

    # Save results to JSON file
    with open('utilization_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to cuda_results.json")
    return all_results


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__))+'/results')
    # Example usage with file list
    file_list = ["deepmd.csv", "gnn.csv", "pr_indochina.csv", "pr_att.csv",
                 "bfs_indochina.csv", "bfs_kron.csv",
                 "sssp_kron.csv", "sssp_indochina.csv",
                 "bc_kron.csv", "bc_indochina.csv",
                 "lammps.csv", "llama2_ft.csv", "llama3_inf.csv",
                 "m-psdns.csv", "milc.csv", "openfold.csv", "pannotia_copapers.csv", "pannotia_att.csv", "qmcpack.csv",
                 "resnet50.csv", "sgemm.csv"
                 ]
    process_files(file_list)
