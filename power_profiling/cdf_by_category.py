import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib
import pickle
from pathlib import Path

# Style Configuration
MEDIUM_SIZE = 32
plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=MEDIUM_SIZE)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Split applications into categories
compute_bound = {
    "llama2_bsz1": "LLaMA2 \n(batch size 1)",
    "llama2_bsz32": "LLaMA2 \n(batch size 32)",
    "llama3_bsz1": "LLaMA3 \n(batch size 1)",
    "llama2_ft": "LLaMA2 FineTune\n(batch size 64)",
    "llama3_bsz32": "LLaMA3 \n(batch size 32)",
    "sd": "Stable Diffusion",
    "lulesh": "LULESH",
}

memory_bound = {
    "gunrock": "Gunrock \nPageRank",
}

hybrid = {
    "deepmd": "DeePMD",
    "deepseek": "DeepSeek-R1-\nDistill-Llama3-8B",
    "lsms": "LSMS",
    "openfold": "OpenFold"
}

# Color dictionary
colordict = {
    "llama2_bsz1": '#517aa3',
    "llama2_bsz32": '#e67e33',
    "llama3_bsz1": '#7a7a28',
    "llama3_bsz32": '#377eb8',     
    "llama2_ft": '#e41a1c',        
    "sd": '#281b6c',
    "gunrock": '#0d5f28',
    "deepmd": '#984ea3',           
    "deepseek": '#a65628',
    "lsms": '#4daf4a',             
    "openfold": '#ff7f00',         
    "lulesh": '#36887a',
}

# Marker style dictionary
markerdict = {
    "llama2_bsz1": 'o',
    "llama2_bsz32": '^',
    "llama3_bsz1": 'D',
    "llama3_bsz32": 's',
    "llama2_ft": '*',
    "sd": 'x',
    "gunrock": '+',
    "deepmd": 'v',
    "deepseek": '<',
    "lsms": '>',
    "openfold": 'p',
    "lulesh": 'h',
}

MARKER_EVERY = 100
CACHE_DIR = "vector_cache"  # Directory to store cached vectors

def format_func(value, tick_number):
    return f'{value:.1f}\u00D7'

def get_cached_data(app_name, frequency, base_dir, tdp=550):
    """Load cached vector data if available, otherwise calculate and cache it"""
    
    # Create cache directory if it doesn't exist
    Path(CACHE_DIR).mkdir(exist_ok=True)
    
    # Generate cache filename
    cache_file = Path(CACHE_DIR) / f"{app_name}_{frequency}_cache.pkl"
    
    # Check if cache file exists
    if cache_file.exists():
        print(f"Loading cached data for {app_name}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # If no cache, calculate the data
    data_file_path = os.path.join(base_dir, app_name, f"metrics_{app_name}_filtered.csv")
    
    try:
        power_data = pd.read_csv(data_file_path)['power_from_e']
        print(f"Successfully read data for {app_name}")
    except Exception as e:
        print(f"Error reading {data_file_path}: {e}")
        return None
    
    # Calculate power ratios
    power_ratios = power_data / tdp
    
    # Filter ratios to be within 0.5×TDP to 1.7×TDP range
    filtered_ratios = power_ratios[(power_ratios >= 0.5) & (power_ratios <= 1.7)]
    
    if len(filtered_ratios) == 0:
        print(f"No power data in range 0.5×TDP to 1.7×TDP for {app_name}")
        return None
    
    # Calculate CDF
    sorted_ratios = np.sort(filtered_ratios)
    cdf = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
    
    # Create result dictionary
    result = {
        'sorted_ratios': sorted_ratios,
        'cdf': cdf,
        'filtered_ratios': filtered_ratios,
        'total_data_points': len(power_data)
    }
    
    # Cache the result
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Cached data for {app_name}")
    return result

def plot_power_cdf_grouped_horizontal(base_dir, app_names, tdp=550):
    # Create horizontal layout with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    
    # Fixed frequency for this analysis
    frequency = "2100"
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    # Lists to store legend handles and labels for each subplot
    handles1, labels1 = [], []
    handles2, labels2 = [], []
    handles3, labels3 = [], []
    
    # Process each application
    for app_name in app_names:
        # Get cached or calculated data
        cached_data = get_cached_data(app_name, frequency, base_dir, tdp)
        
        if cached_data is None:
            continue
        
        # Extract data from cache
        sorted_ratios = cached_data['sorted_ratios']
        cdf = cached_data['cdf']
        filtered_ratios = cached_data['filtered_ratios']
        total_data_points = cached_data['total_data_points']
        
        # Get marker style and color
        marker = markerdict.get(app_name, 'o')
        color = colordict.get(app_name, '#000000')
        
        # Determine which subplot to use based on application category
        if app_name in compute_bound:
            ax = ax1
            display_name = compute_bound[app_name]
            # Store the handle and label for later legend creation
            line, = ax.plot(sorted_ratios, cdf,
                    color=color,
                    linewidth=2.0,
                    marker=marker,
                    markevery=MARKER_EVERY,
                    markersize=4)
            handles1.append(line)
            labels1.append(display_name)
        elif app_name in memory_bound:
            ax = ax2
            display_name = memory_bound[app_name]
            line, = ax.plot(sorted_ratios, cdf,
                    color=color,
                    linewidth=2.0,
                    marker=marker,
                    markevery=MARKER_EVERY,
                    markersize=4)
            handles2.append(line)
            labels2.append(display_name)
        elif app_name in hybrid:
            ax = ax3
            display_name = hybrid[app_name]
            line, = ax.plot(sorted_ratios, cdf,
                    color=color,
                    linewidth=2.0,
                    marker=marker,
                    markevery=MARKER_EVERY,
                    markersize=4)
            handles3.append(line)
            labels3.append(display_name)
        else:
            print(f"Warning: {app_name} not found in any category")
            continue
        
        # Print statistics
        print(f"Statistics for {app_name}:")
        print(f"Data in 0.5-1.7×TDP range: {len(filtered_ratios)/total_data_points*100:.2f}%")
        if len(filtered_ratios) > 0:
            print(f"Min power ratio in range: {min(filtered_ratios):.2f}× TDP")
            print(f"Max power ratio in range: {max(filtered_ratios):.2f}× TDP")
    
    # Add vertical lines at 1.0 TDP to each subplot
    red_line1 = ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5)
    red_line2 = ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5)
    red_line3 = ax3.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5)
    
    # Configure axes and labels
    ax1.set_title("High Spike Workloads")
    ax2.set_title("Low Spike Workloads")
    ax3.set_title("Hybrid Workloads")
    
    # Set x-axis limit explicitly from 0.5 to 1.7
    ax1.set_xlim(0.5, 1.7)
    ax2.set_xlim(0.5, 1.7)
    ax3.set_xlim(0.5, 1.7)
    
    # Format x-axis labels
    ax1.xaxis.set_major_formatter(FuncFormatter(format_func))
    ax2.xaxis.set_major_formatter(FuncFormatter(format_func))
    ax3.xaxis.set_major_formatter(FuncFormatter(format_func))
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Set common y-axis label
    fig.text(0.04, 0.5, "Cumulative Fraction", va='center', rotation='vertical')
    
    # Add x-axis labels
    ax1.set_xlabel('Power Ratio (×TDP)')
    ax2.set_xlabel('Power Ratio (×TDP)')
    ax3.set_xlabel('Power Ratio (×TDP)')
    
    # Create a 1.0x TDP reference line entry for the legend
    from matplotlib.lines import Line2D
    red_line = Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='1.0× TDP')
    
    # Add the reference line to all legends
    handles1.append(red_line)
    labels1.append('1.0× TDP')
    
    handles2.append(red_line)
    labels2.append('1.0× TDP')
    
    handles3.append(red_line)
    labels3.append('1.0× TDP')
    
    # Create legends below the x-axis
    # Adjust the bbox_to_anchor to position the legend below the plot
    ax1.legend(handles1, labels1,fontsize=MEDIUM_SIZE-6,framealpha=0, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
    ax2.legend(handles2, labels2,fontsize=MEDIUM_SIZE-6,framealpha=0, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=1)
    ax3.legend(handles3, labels3,fontsize=MEDIUM_SIZE-6, framealpha=0,loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)


    
    # Adjust layout to make room for legends at the bottom
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.1, top=0.92, bottom=0.4)  # Increase bottom margin for legends
    
    return plt

def main():
    # Specify base directory where data files are stored
    base_dir = "./results"  # Update this to your actual data directory
    
    # List of applications to process
    app_names = ["llama2_bsz1","llama2_bsz32","llama3_bsz32","llama3_bsz1", "pagerank_indochina", "lulesh", "sd-xl", "deepmd","lsms","llama2_ft","openfold"]
    
    # Create and save the horizontal plot
    plt = plot_power_cdf_grouped_horizontal(base_dir, app_names)
    plt.savefig('category_power_cdf_default.pdf', dpi=600, bbox_inches='tight')
    plt.savefig('category_power_cdf_default.png', dpi=600, bbox_inches='tight')
    
    print("\nCategory-based power CDF plot saved")
    plt.close()

if __name__ == "__main__":
    main()