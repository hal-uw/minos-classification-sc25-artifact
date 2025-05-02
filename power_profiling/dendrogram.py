import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
import os
import seaborn as sns
import json


filename_mapping = {}

# Application labels dictionary
app_display_names = {
    "llama2_bsz1": "LLaMA2 \nBSZ 1",
    "llama2_bsz8": "LLaMA2 \nBSZ 8",
    "llama2_bsz32": "LLaMA2 \nBSZ 32",
    "llama3_bsz1": "LLaMA3 \nBSZ 1",
    "llama3_bsz8": "LLaMA3 \nBSZ 8",
    "llama3_bsz32": "LLaMA3 \nBSZ 32",
    "gunrock": "PageRank",
    "lulesh": "LULESH",
    "sd": "SD-XL",
    "deepmd": "DeePMD",
    "deepseek": "DeepSeek-R1-\nDistill-Llama3-8B",
    "llama2_ft":"LLaMA2 FT\nBSZ 64",
    "lsms": "LSMS",
    "openfold":"OpenFold",
}

# Cache for power distribution results
results_cache = {}

def calculate_power_distribution(data_path, tdp=550, alpha=0.5, vectors_file="app_vectors.json"):
    """Calculate power distribution across different TDP ranges."""
    
    # Check if result is already in cache
    app_name = filename_mapping.get(data_path)
    label = app_display_names.get(app_name, app_name)
    
    # First check if we have persisted vectors file
    if os.path.exists(vectors_file):
        try:
            with open(vectors_file, 'r') as f:
                saved_vectors = json.load(f)
                
            if label in saved_vectors:
                print(f"Loading vector for {label} from {vectors_file}")
                return label, saved_vectors[label]
        except Exception as e:
            print(f"Error loading from {vectors_file}: {str(e)}")
    
    # Then check in-memory cache
    if label in results_cache:
        print(f"Using cached result for {label}")
        return label, results_cache[label]
    
    try:
        # Read CSV data
        filtered_data = data_path
        df = pd.read_csv(filtered_data)
        print(f"\nProcessing {filtered_data}...")

        # Get power data and apply filtering
        power_data = df['power_from_e']
        filtered_power = power_data.copy()

        # Calculate power ratios relative to TDP
        power_ratios = filtered_power / tdp

        # Define TDP ratio bins
        bins = np.arange(0.5, 2.05, 0.05)

        # Only take samples where power is greater than TDP
        excursion_samples = power_ratios[power_ratios >= 0.5]
        total_excursions = len(excursion_samples)

        if total_excursions == 0:
            # No excursions above TDP
            percentages = [0] * (len(bins)-1)
            results_cache[label] = percentages
            return label, percentages

        # Calculate percentage for each bin range
        percentages = []
        for i in range(len(bins)-1):
            count = len(excursion_samples[(excursion_samples >= bins[i]) &
                                          (excursion_samples < bins[i+1])])
            percentage = (count / total_excursions) * 100
            percentages.append(percentage)

        percentages = [round(p/100, 4) for p in percentages]

        # Print results
        print(f"{label}:")
        print(f"Distribution: {bins}")
        print(f"Normalized vector: {percentages}")
        
        # Cache results
        results_cache[label] = percentages
        
        # Save to persisted JSON file
        # First try to load existing file if it exists
        saved_vectors = {}
        if os.path.exists(vectors_file):
            try:
                with open(vectors_file, 'r') as f:
                    saved_vectors = json.load(f)
            except:
                pass
        
        # Update with new vector
        saved_vectors[label] = percentages
        
        # Save back to file
        try:
            with open(vectors_file, 'w') as f:
                json.dump(saved_vectors, f, indent=2)
            print(f"Saved vector for {label} to {vectors_file}")
        except Exception as e:
            print(f"Error saving to {vectors_file}: {str(e)}")
        
        return label, percentages

    except Exception as e:
        print(f"Error processing {data_path}: {str(e)}")
        return None


# Cache for distance matrices
distance_matrix_cache = {}

def plot_distance_matrix(data_dict, metric, output_file_prefix):
    """
    Plot and save the distance matrix as a heatmap using the specified distance metric.
    
    Parameters:
    data_dict: Dictionary containing application names and corresponding vectors
    metric: Distance metric method
    output_file_prefix: Prefix for output filenames
    """
    
    # Check if the distance matrix is already computed for this metric
    cache_key = f"{metric}_{','.join(data_dict.keys())}"
    if cache_key in distance_matrix_cache:
        print(f"Using cached distance matrix for {metric}")
        return distance_matrix_cache[cache_key]
    
    # Prepare data
    labels = list(data_dict.keys())
    vectors = np.array(list(data_dict.values()))
    
    # Calculate distance matrix
    distances = pdist(vectors, metric=metric)
    
    # Convert distance vector to square matrix
    distance_matrix = squareform(distances)
    
    # Save distance matrix to CSV
    csv_filename = f"{output_file_prefix}_{metric}_matrix.csv"
    distance_df = pd.DataFrame(distance_matrix, index=labels, columns=labels)
    distance_df.to_csv(csv_filename)
    print(f"Distance matrix saved as {csv_filename}")
    
    # Set matplotlib parameters
    MEDIUM_SIZE = 24
    plt.rcParams.update({
        'font.size': MEDIUM_SIZE,
        'axes.titlesize': MEDIUM_SIZE,
        'axes.labelsize': MEDIUM_SIZE,
        'xtick.labelsize': MEDIUM_SIZE-4,
        'ytick.labelsize': MEDIUM_SIZE-4,
        'legend.fontsize': MEDIUM_SIZE,
        'figure.titlesize': MEDIUM_SIZE,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    
    # Create figure for heatmap
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    ax = sns.heatmap(distance_matrix, 
                    xticklabels=labels, 
                    yticklabels=labels,
                    cmap="viridis", 
                    annot=True, 
                    fmt=".2f", 
                    linewidths=.5,
                    annot_kws={"size": MEDIUM_SIZE-8})
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Set title and labels
    plt.title(f"{metric.capitalize()} Distance Matrix")
    
    # Save figure
    heatmap_filename = f"{output_file_prefix}_{metric}_heatmap.pdf"
    plt.tight_layout()
    plt.savefig(heatmap_filename, bbox_inches='tight', dpi=600)
    print(f"Distance matrix heatmap saved as {heatmap_filename}")
    plt.close()
    
    # Cache the distance matrix
    distance_matrix_cache[cache_key] = distance_matrix
    
    return distance_matrix


# Cache for linkage matrices
linkage_matrix_cache = {}

def plot_dendrogram_with_metric(data_dict, metric, output_file, distance_matrix=None):
    """
    Plot hierarchical clustering dendrogram using specified distance metric

    Parameters:
    data_dict: Dictionary containing application names and corresponding vectors
    metric: Distance metric method
    output_file: Output filename
    distance_matrix: Optional pre-computed distance matrix
    """
    # Check if linkage matrix is already computed
    cache_key = f"{metric}_{','.join(data_dict.keys())}"
    if cache_key in linkage_matrix_cache:
        print(f"Using cached linkage matrix for {metric}")
        linkage_matrix = linkage_matrix_cache[cache_key]
    else:
        # Set matplotlib parameters
        MEDIUM_SIZE = 26
        plt.rcParams.update({
            'font.size': MEDIUM_SIZE,
            'axes.titlesize': MEDIUM_SIZE,
            'axes.labelsize': MEDIUM_SIZE,
            'xtick.labelsize': MEDIUM_SIZE,
            'ytick.labelsize': MEDIUM_SIZE,
            'legend.fontsize': MEDIUM_SIZE,
            'figure.titlesize': MEDIUM_SIZE,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        })
        
        # Prepare data
        labels = list(data_dict.keys())
        vectors = np.array(list(data_dict.values()))

        # Calculate distance matrix if not provided
        if distance_matrix is None:
            # Check if we have a cached distance matrix
            if cache_key in distance_matrix_cache:
                distances = squareform(distance_matrix_cache[cache_key])
            else:
                distances = pdist(vectors, metric=metric)
        else:
            # Convert square matrix to condensed form if necessary
            if len(distance_matrix.shape) == 2:
                distances = squareform(distance_matrix)
            else:
                distances = distance_matrix

        # Create hierarchical clustering
        linkage_matrix = hierarchy.linkage(distances, method='ward')
        
        # Save linkage matrix to npy file
        linkage_file = f"linkage_matrix_{metric}.npy"
        np.save(linkage_file, linkage_matrix)
        print(f"Linkage matrix saved as {linkage_file}")
        
        # Cache the linkage matrix
        linkage_matrix_cache[cache_key] = linkage_matrix

    # Create figure
    plt.figure(figsize=(14, 10))

    # Plot dendrogram
    labels = list(data_dict.keys())
    dendrogram = hierarchy.dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=45,
        leaf_font_size=MEDIUM_SIZE-2,
        color_threshold=0.7 * max(linkage_matrix[:, 2]),
        above_threshold_color='grey'
    )

    # Bold all text labels
    ax = plt.gca()
    xlabels = ax.get_xmajorticklabels()  # Get x-axis labels

    # Change font properties for all labels
    for label in xlabels:
        label.set_fontweight('bold')  # Add this line to make text bold

    plt.xlabel('Application',fontsize=MEDIUM_SIZE)
    plt.ylabel('Distance',fontsize=MEDIUM_SIZE)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, bbox_inches='tight', dpi=600)
    print(f"Dendrogram with {metric} distance saved as {output_file}")
    plt.close()


def main(vectors_file="app_vectors.json", skip_calculation=False):
    """
    Main function to process data and generate visualizations.
    
    Parameters:
    vectors_file: Path to the JSON file where application vectors are stored
    skip_calculation: If True, skip CSV processing and use only saved vectors
    """
    # Define filename mappings for the applications
    app_names = ["llama2_bsz1","llama2_bsz32","llama3_bsz32","llama3_bsz1", "pagerank_indochina", "lulesh", "sd-xl", "deepmd","lsms","llama2_ft","openfold"]
    prefix = "metrics_"
    suffix = "_filtered.csv"
    for app_name in app_names:
        filename = f"{prefix}{app_name}{suffix}"
        filename_mapping[filename] = app_name

    data_paths = list(filename_mapping.keys())

    # Check if we should load directly from JSON
    if skip_calculation and os.path.exists(vectors_file):
        try:
            with open(vectors_file, 'r') as f:
                results = json.load(f)
            print(f"Loaded all vectors from {vectors_file}")
        except Exception as e:
            print(f"Error loading from {vectors_file}, will calculate: {str(e)}")
            results = {}
            skip_calculation = False
    else:
        results = {}
    
    # If not skipping calculation or loading failed, process each data file
    if not skip_calculation:
        for path in data_paths:
            result = calculate_power_distribution(path, vectors_file=vectors_file)
            if result is not None:
                label, vector = result
                results[label] = vector

    # Create output directory for matrices if it doesn't exist
    os.makedirs("distance_matrices", exist_ok=True)
    
    # Generate distance matrices and dendrograms with different metrics
    metrics = ['cosine']
    for metric in metrics:
        # Plot and save distance matrix
        distance_matrix = plot_distance_matrix(results, metric, "distance_matrices/distance")
        # Generate dendrogram using the pre-computed distance matrix
        output_file = f'dendrogram_{metric}.pdf'
        plot_dendrogram_with_metric(results, metric, output_file, distance_matrix)



if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process application data and generate visualizations')
    parser.add_argument('--vectors-file', type=str, default='app_vectors.json', 
                        help='Path to JSON file for storing/loading application vectors')
    parser.add_argument('--skip-calculation', action='store_true', 
                        help='Skip CSV processing and use only saved vectors from the JSON file')
    parser.add_argument('--visualize-only', action='store_true',
                        help='Only create visualizations using existing data (alias for --skip-calculation)')
    
    args = parser.parse_args()
    
    # If visualize-only is set, also set skip-calculation
    if args.visualize_only:
        args.skip_calculation = True
    
    main(vectors_file=args.vectors_file, skip_calculation=args.skip_calculation)