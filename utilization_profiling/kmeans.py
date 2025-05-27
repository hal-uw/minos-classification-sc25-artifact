import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict
import matplotlib
from sklearn.metrics import silhouette_score
app_label_dict = {
    "deepmd": "DeePMD",
    "gnn": "GNN",
    "lammps": "LAMMPS",
    "llama2_ft": "Llama2 Fine-tuning",
    "milc": "MILC",
    "openfold": "OpenFold",
    "pannotia_att": "Pannotia PageRank, at&t",
    "bfs_indochina": "Gunrock BFS, indochina",
    "bc_indochina": "Gunrock BC, indochina",
    "sssp_indochina": "Gunrock SSSP, indochina",
    "llama3_inf": "Llama3 Inference",
    #   hollywood dataset
    ####################
    # "bfs_hollywood":"Gunrock Breadth-First Search on hollywood graph",
    # "bc_hollywood":"Gunrock Betweenness Centrality on hollywood graph",
    # "sssp_hollywood":"Gunrock Single Source Shortest Path on hollywood graph",
    #####################
    # kron dataset
    "bfs_kron": "Gunrock BFS, kron",
    "bc_kron": "Gunrock BC, kron",
    "sssp_kron": "Gunrock SSSP, kron",

    "pannotia_copapers": "Pannotia PageRank, coPapersDBLP",
    "pr_att": "Gunrock PageRank, at&t",
    ############
    # "pr_hollywood":"Gunrock PageRank on hollywood graph",
    #########
    "pr_indochina": "Gunrock PageRank, indochina",
    ###############
    # "pr_kron":"Gunrock PageRank on kron graph",
    # "pr_roadnet":"Gunrock PageRank on roadnet graph",
    ###############
    "m-psdns": "M-PSDNS",
    "resnet_50": "ResNet50 bsz256",
    # "resnet50_64":"ResNet50 bsz64",
    "sgemm": "SGEMM",
}


def load_data(file_path):
    """Load data from JSON file and convert to DataFrame."""
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    data = {}
    for key, value in json_data.items():
        data[key] = value['data'][0] if isinstance(
            value['data'][0], list) else value['data']

    columns = list(json_data.values())[0]['columns']
    return pd.DataFrame.from_dict(data, orient='index', columns=columns)


def filter_dataframe(df, app_label_dict):
    """Filter DataFrame to only include rows that exist in app_label_dict."""
    return df.loc[df.index.isin(app_label_dict.keys())]


def perform_kmeans(df, features, n_clusters):
    """Perform K-means clustering on selected features."""
    df_selected = df[features].copy()
    # print out the df_selected
    print("Selected features for clustering:")
    df_selected.head()
    # Create a copy to avoid SettingWithCopyWarning
    kmeans = KMeans(n_clusters=3, random_state=42)
    df.loc[:, 'Cluster'] = kmeans.fit_predict(df_selected)
    # calculate the silhouette score
    score = silhouette_score(df_selected, df['Cluster'])
    print(f"Silhouette Score: {score:}")
    return df, kmeans


def examine_clusters(df, kmeans, features):
    """
    Print information about clusters to help with manual assignment.
    """
    # Get cluster centers
    centers = kmeans.cluster_centers_
    compute_idx = features.index("Compute (SM) Throughput")
    memory_idx = features.index("DRAM Throughput")

    print("\nCluster Centers for manual categorization:")
    for i, center in enumerate(centers):
        print(
            f"Cluster {i}: Compute={center[compute_idx]:.2f}, Memory={center[memory_idx]:.2f}")

    # Print members of each cluster
    for cluster in range(kmeans.n_clusters):
        print(f"\nApplications in Cluster {cluster}:")
        cluster_members = df[df['Cluster'] == cluster].index.tolist()
        for app in cluster_members:
            print(f"  - {app} ({app_label_dict[app]})")


def assign_cluster_types(kmeans, cluster_assignments):
    """
    Manually assign cluster types based on provided mapping.

    Args:
        kmeans: The KMeans object
        cluster_assignments: A dictionary mapping cluster numbers to category types
                            ('M', 'C', or 'H')
    """
    # Convert the cluster_assignments to required maps
    cluster_category_map = {}
    cluster_full_name_map = {}
    color_map = {}

    full_names = {'M': 'Memory Bound',
                  'C': 'Compute Bound', 'H': 'Hybrid Bound'}
    colors = {'M': 'blue', 'C': 'red', 'H': 'green'}

    for cluster, category in cluster_assignments.items():
        cluster_category_map[cluster] = category
        cluster_full_name_map[cluster] = full_names[category]
        color_map[cluster] = colors[category]

    return cluster_category_map, cluster_full_name_map, color_map


def plot_clusters_with_categories(df, features, app_label_dict, cluster_category_map, cluster_full_name_map, color_map):
    """Plot clusters with category labels (M#, C#, H#) and two legends without distorting the plot."""
    # Create a figure with a larger size to accommodate both plot and legends
    fig = plt.figure(figsize=(14, 8))  # Wider and taller figure

    # Create the main axes for the scatter plot
    ax = plt.axes([0.1, 0.1, 0.8, 0.7])  # [left, bottom, width, height]
    ax.grid(True)

    # Define font settings
    MEDIUM_SIZE = 26
    plt.rcParams.update({
        # 'font.family': 'sans-serif',         # Use sans-serif fonts
        # 'font.sans-serif': ['Arial'],        # Specifically use Arial
        'font.size': MEDIUM_SIZE,            # Base font size for most elements
        'axes.titlesize': MEDIUM_SIZE,       # Title font size
        'axes.labelsize': MEDIUM_SIZE,       # Axis label font size
        'xtick.labelsize': MEDIUM_SIZE,      # X-axis tick label size
        'ytick.labelsize': MEDIUM_SIZE,      # Y-axis tick label size
        'legend.fontsize': MEDIUM_SIZE,      # Legend font size
        'figure.titlesize': MEDIUM_SIZE,     # Figure title font size
        'pdf.fonttype': 42,                  # Embed fonts in PDF
        'ps.fonttype': 42,                   # Embed fonts in PS
    })

    # Define academic-friendly colors
    academic_colors = {
        'M': '#4878D0',  # Medium blue for Memory Bound
        'C': '#EE6677',  # Muted red for Compute Bound
        'H': '#55A868'   # Muted green for Hybrid Bound
    }

    # Create counters for each category
    category_counters = defaultdict(int)

    # Store the scatter plots for legend
    scatter_plots = []

    # Create a scatter plot for each cluster with its own color
    for cluster, category in cluster_category_map.items():
        cluster_data = df[df['Cluster'] == cluster]
        color = academic_colors[category]
        scatter = ax.scatter(
            cluster_data[features[0]],
            cluster_data[features[1]],
            c=color,
            label=cluster_full_name_map[cluster],
            s=200,  # Larger point size
            alpha=0.8,  # More opaque
            edgecolors='black',  # Add black edge to make points stand out
            linewidths=0.5
        )
        scatter_plots.append(scatter)

    # Set axis labels and title
    ax.set_xlabel(features[0], fontsize=MEDIUM_SIZE)
    ax.set_ylabel(features[1], fontsize=MEDIUM_SIZE)
    ax.set_title('K-means Clustering Results', pad=20)

    # Calculate offsets for each point to avoid overlapping text
    cluster_centers = df.groupby('Cluster')[features].mean().reset_index()

    # Create a dictionary to store labels and app names for the second legend
    app_labels = {}

    # Add category-specific numbered labels with matching colors and offsets
    for idx, row in df.iterrows():
        category = row['Category']
        category_counters[category] += 1
        label = f"{category}{category_counters[category]}"
        color = academic_colors[category]

        # Store the label and app name for the second legend
        app_labels[label] = app_label_dict[idx]

        # Special positioning for M2 and M7
        if label in ["M9", "M2"]:
            dx, dy = 2.0, 2.0
        else:
            # Get the cluster center for this point
            cluster_id = row['Cluster']
            center_x = cluster_centers.loc[cluster_centers['Cluster']
                                           == cluster_id, features[0]].values[0]
            center_y = cluster_centers.loc[cluster_centers['Cluster']
                                           == cluster_id, features[1]].values[0]

            # Calculate offset direction (away from cluster center)
            dx = row[features[0]] - center_x
            dy = row[features[1]] - center_y

            # Normalize offset and apply a fixed distance
            offset_distance = 2.0
            if abs(dx) > 0.001 or abs(dy) > 0.001:  # Avoid division by zero
                magnitude = (dx**2 + dy**2)**0.5
                dx = dx / magnitude * offset_distance
                dy = dy / magnitude * offset_distance
            else:
                # If point is very close to center, use a default offset
                dx, dy = offset_distance, offset_distance

        # Add the label text near each point
        ax.text(row[features[0]] + dx, row[features[1]] + dy, label,
                ha='center', va='center',
                color='black',  # Use black text for better readability
                )

    # Set axis limits
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)

    # Add the first legend for categories
    # set the font size of the legend to 16

    legend1 = ax.legend(handles=scatter_plots, framealpha=0.0,
                        fontsize=MEDIUM_SIZE, loc='upper right')

    # Customize legend markers to be larger
    for handle in legend1.legend_handles:
        handle.set_sizes([200])

    # Add legend1 to the plot
    ax.add_artist(legend1)

    # Create a custom legend for app labels
    sorted_labels = sorted(app_labels.items())

    # Create a separate axis for the application legend
    # This is the key change - creating a dedicated space for the applications legend
    # [left, bottom, width, height]
    app_legend_ax = fig.add_axes([0.1, 0.0, 0.8, 0.1])
    app_legend_ax.axis('off')  # Hide the axes

    # Calculate how many columns we need for the legend
    num_items = len(sorted_labels)
    ncols = 2  # Default to 2 columns

    # Create custom handles and labels for the second legend
    handles = []
    labels = []

    for label, app_name in sorted_labels:
        # Create a proxy artist for the label
        category = label[0]  # First character indicates category (M, C, or H)
        color = academic_colors[category]

        # Create a marker with the appropriate color
        handle = plt.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=color, markeredgecolor='black',
                            markersize=10, label=label)

        handles.append(handle)
        labels.append(f"{label}: {app_name}")

    # Add the second legend to its dedicated axis
    bbox_to_anchor = (0.5, 0.5)
    legend2 = app_legend_ax.legend(handles=handles, labels=labels, framealpha=0.0,
                                   bbox_to_anchor=(1.2, 0.2), ncol=ncols, fontsize=MEDIUM_SIZE,
                                   title="Applications")
    app_legend_ax.set_title("", fontsize=MEDIUM_SIZE)

    # Adjust the figure spacing to make everything fit well
    # Adjust the rect parameter to avoid overlap
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    # Save the figure
    plt.savefig('kmeans_categorized.png', dpi=600, bbox_inches='tight')
    plt.savefig('kmeans_categorized.pdf', dpi=600, bbox_inches='tight')
    plt.show()


def print_cluster_info(df, kmeans, features, cluster_full_name_map):
    """Print information about the clusters."""
    for cluster in range(kmeans.n_clusters):
        print(f"\nCluster {cluster} ({cluster_full_name_map[cluster]}):")
        cluster_members = df[df['Cluster'] == cluster].index.tolist()
        print(", ".join(cluster_members))
        # Also print with app labels
        labeled_members = [
            f"{idx} ({app_label_dict[idx]})" for idx in cluster_members]
        print("Labels: " + ", ".join(labeled_members))

    cluster_averages = df.groupby('Cluster')[features].mean()
    print("\nCluster Averages:")
    print(cluster_averages)

    print("\nCluster Centers:")
    print(pd.DataFrame(kmeans.cluster_centers_, columns=features))


os.chdir(os.path.dirname(os.path.abspath(__file__))+'/results')
file_path = 'utilization_results.json'
features = ["Compute (SM) Throughput", "DRAM Throughput"]
n_clusters = 3

# Main execution flow
df = load_data(file_path)
# Filter the DataFrame to only include rows in app_label_dict
filtered_df = filter_dataframe(df, app_label_dict)
print(
    f"Original data had {len(df)} entries, filtered data has {len(filtered_df)} entries")

# Perform clustering on the filtered data
filtered_df, kmeans = perform_kmeans(filtered_df, features, n_clusters)

# Examine cluster information to help with manual assignment
examine_clusters(filtered_df, kmeans, features)

# Count items in each cluster to help with manual assignment
cluster_counts = filtered_df['Cluster'].value_counts().to_dict()
print("\nNumber of items in each cluster:")
for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster}: {count} items")

# Assign the clusters based on item count
cluster_assignments = {}
for cluster, count in cluster_counts.items():
    # Check if this cluster contains sgemm
    cluster_members = filtered_df[filtered_df['Cluster']
                                  == cluster].index.tolist()
    contains_sgemm = 'sgemm' in cluster_members

    if contains_sgemm:
        # If cluster contains sgemm, assign it to 'C' regardless of count
        cluster_assignments[cluster] = 'C'
    elif count == 8:  # The cluster with 8 items should be Memory Bound
        cluster_assignments[cluster] = 'M'
    elif count == 4:  # The cluster with 4 items should be Hybrid Bound
        cluster_assignments[cluster] = 'H'
    else:  # Any other cluster is Compute Bound
        cluster_assignments[cluster] = 'C'

print("\nCluster assignments based on item count:")
for cluster, category in cluster_assignments.items():
    print(f"Cluster {cluster} → {category} ({cluster_counts[cluster]} items)")

# Apply the manual assignments
cluster_category_map, cluster_full_name_map, color_map = assign_cluster_types(
    kmeans, cluster_assignments)
filtered_df['Category'] = filtered_df['Cluster'].map(cluster_category_map)

# Plot with the new category labels
plot_clusters_with_categories(filtered_df, features, app_label_dict,
                              cluster_category_map, cluster_full_name_map, color_map)

# Print cluster information
print_cluster_info(filtered_df, kmeans, features, cluster_full_name_map)
