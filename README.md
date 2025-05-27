# Minos: Systematically Classifying Performance and Power Characteristics of GPU Workloads on HPC Clusters

This artifact contains all scripts, configurations, and datasets needed to reproduce the results from the paper "Minos: Systematically Classifying Performance and Power Characteristics of GPU Workloads on HPC Clusters" submitted to SC'25.

Minos is a novel classification scheme that groups GPU workloads based on both power spike distributions and performance characteristics, enabling more efficient power and performance optimization in HPC clusters.

## Table of Contents

- [Minos: Systematically Classifying Performance and Power Characteristics of GPU Workloads on HPC Clusters](#minos-systematically-classifying-performance-and-power-characteristics-of-gpu-workloads-on-hpc-clusters)
  - [Table of Contents](#table-of-contents)
  - [Directory Structure](#directory-structure)
  - [Prerequisites](#prerequisites)
    - [Hardware Requirements](#hardware-requirements)
    - [Software Requirements](#software-requirements)
  - [Setup](#setup)
    - [1. Clone Repository](#1-clone-repository)
    - [2. Install Python Dependencies](#2-install-python-dependencies)
    - [3. Prepare Datasets](#3-prepare-datasets)
  - [Experiments](#experiments)
    - [Power-based Classification](#power-based-classification)
    - [Utilization-based Classification](#utilization-based-classification)
    - [Frequency Capping Analysis](#frequency-capping-analysis)
  - [Data Extraction \& Plotting](#data-extraction--plotting)
  - [Key Results](#key-results)
  - [Troubleshooting](#troubleshooting)
  - [Related Repositories](#related-repositories)
  - [Citation](#citation)
  - [Support](#support)

## Directory Structure

```
minos-classification-sc25-artifact/
├── datasets/                   # Input datasets for workloads
├── power_profiling/            # Power-based classification scripts
├── rocprofwrap/                # ROCm profiling wrapper
├── utilization_profiling/      # Utilization-based classification
├── frequency_capping/          # Frequency capping experiments
├── prepare_datasets.sh         # Dataset preparation script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Prerequisites

### Hardware Requirements
- **For Power Classification**: AMD MI210 or newer GPU with administrative privileges
- **For Utilization Classification**: NVIDIA Ampere A100 GPU or later generations
- **For Frequency Capping**: AMD MI300 GPU with frequency capping support

### Software Requirements
- ROCm 6.3.4+ (for AMD GPUs)
- CUDA Toolkit 11.4+ (for NVIDIA GPUs)
- Docker or Apptainer (Singularity)
- Python 3.8+

## Setup

### 1. Clone Repository
```bash
git clone --recursive https://github.com/hal-uw/minos-classification-sc25-artifact.git
cd minos-classification-sc25-artifact
```

### 2. Install Python Dependencies
```bash
python -m venv minos_venv
source ./minos_venv/bin/activate
pip3 install -r requirements.txt
```

### 3. Prepare Datasets
The dataset preparation is optional and modular. You can download all datasets or only specific ones needed for your experiments.
Download All Datasets (Default)
```bash 
mkdir -p datasets
# Edit prepare_datasets.sh line 7 to set your HF_TOKEN if needed
bash prepare_datasets.sh
```
Download Specific Datasets
```bash# Download only specific datasets (comma-separated)
bash prepare_datasets.sh --option qmcpack,gunrock
```
```bash
# Available datasets:
# - qmcpack: QMCPack quantum Monte Carlo datasets
# - gunrock: Graph analytics datasets (indochina, att, coPapersDBLP, kron)
# - openfold_aqlab: OpenFold AQLab protein datasets (~640GB)
# - openfold_mlcommons: OpenFold MLCommons datasets
# - gnn: Graph Neural Network datasets (IGBH-tiny)
# - llama2_ft: LLaMA2 fine-tuning model files
```
**Note**: 
- OpenFold's OpenProteinSet is large (640GB). Ensure sufficient disk space and download time.
- Some datasets require authentication tokens:
  - Edit prepare_datasets.sh line 7 to set your HF_TOKEN for Hugging Face datasets
  - Edit power_profiling/run_with_option.sh line 4 to set your HF_TOKEN for power profiling experiments
- AWS CLI is required for OpenFold datasets
- You can start with smaller datasets for initial testing

## Experiments

### Power-based Classification

This experiment reproduces **Figures 3 & 4** from the paper, showing hierarchical clustering based on power spike distributions.

**Expected Time**: ~240 minutes profiling + 15 minutes analysis

```bash
cd power_profiling

# Profile all workloads (or specify --selected_app for individual workloads)
bash run_with_option.sh --selected_app all

# Post-process data
bash post_processing.sh

# Generate figures
source ../minos_venv/bin/activate
python3 dendrogram.py
python3 cdf_by_category.py
```

### Utilization-based Classification

This experiment reproduces **Figure 5** from the paper, showing K-means clustering based on compute and memory utilization.

**Expected Time**: Up to 1 week for full profiling + 15 minutes analysis

```bash
cd utilization_profiling

# Profile workloads (warning: very time-intensive)
bash run_with_option.sh --selected_app all

# Post-process data
bash post_processing.sh

# Generate clustering plot
source ../minos_venv/bin/activate
python3 extract.py
python3 k-means.py
```

### Frequency Capping Analysis

This experiment reproduces **Figures 6-9** from the paper, showing how frequency capping affects workloads grouped by Minos.

**Expected Time**: ~200 minutes per workload × number of frequency caps

```bash
cd frequency_capping

# Run frequency capping experiments
python3 run_cap_freq.py "1300,1500,1700,1900,2100"

# Generate analysis plots
source ../minos_venv/bin/activate
python3 power_scaling.py
python3 perf_scaling.py
python3 case_study_openfold.py
```

## Data Extraction & Plotting

After completing experiments, generate all figures:

```bash
source ./minos_venv/bin/activate

# Power classification plots
python3 power_profiling/dendrogram.py
python3 power_profiling/cdf_by_category.py

# Utilization classification plots
python3 utilization_profiling/extract.py
python3 utilization_profiling/k-means.py

# Frequency capping analysis
python3 frequency_capping/power_scaling.py
python3 frequency_capping/perf_scaling.py
python3 frequency_capping/case_study_openfold.py
```

Figures will be saved in the respective experiment directories.


**Warning**: This will take several days to complete due to extensive profiling requirements.

## Key Results

This artifact demonstrates:

1. **Novel Power Classification** (C1): Groups workloads by power spike distributions into high-spike, low-spike, and hybrid categories
2. **Performance Classification** (C2): Clusters workloads as compute-intensive, memory-intensive, or hybrid based on utilization metrics
3. **Validation via Frequency Capping** (C3): Shows that workloads in the same power class exhibit similar responses to frequency caps
4. **Case Study Effectiveness** (C4): Demonstrates 90% reduction in profiling time for new workloads while maintaining prediction accuracy

## Troubleshooting

- **Permission Issues**: Ensure administrative privileges for frequency capping experiments
- **Memory Issues**: OpenFold requires substantial memory; adjust batch sizes if needed
- **Profiling Overhead**: Utilization profiling is very time-intensive; consider profiling subset of workloads for initial validation

## Related Repositories

- [OpenFold](https://github.com/aqlaboratory/openfold): Protein structure prediction
- [LAMMPS](https://github.com/lammps/lammps): Molecular dynamics simulator  
- [Gunrock](https://github.com/gunrock/gunrock): GPU graph processing library
- [Pannotia](https://github.com/pannotia/pannotia): GPU graph algorithms
- [MLPerf Inference](https://github.com/mlcommons/inference): ML benchmarking suite

## Citation

If you use this artifact, please cite our paper:

```bibtex

```

## Support

For questions about this artifact or the Minos classification system, please open an issue in this repository or contact the authors.