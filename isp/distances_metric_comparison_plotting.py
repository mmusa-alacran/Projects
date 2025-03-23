import json
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the JSON file
with open('mutation_results_all.json', 'r') as file:
    data = json.load(file)

# Define a color map for individual files
color_map = plt.cm.get_cmap('tab20', len(data.keys()))

def plot_individual_files(data):
    for idx, (filename, results) in enumerate(data.items()):
        euclidean_distances = [result['euclidean_distance'] for result in results]
        cosine_distances_dim0 = [result['cosine_distance'][0] for result in results]
        cosine_distances_dim1 = [result['cosine_distance'][1] for result in results]

        min_euclidean = min(euclidean_distances)
        max_euclidean = max(euclidean_distances)
        min_cosine_dim0 = min(cosine_distances_dim0)
        max_cosine_dim0 = max(cosine_distances_dim0)
        min_cosine_dim1 = min(cosine_distances_dim1)
        max_cosine_dim1 = max(cosine_distances_dim1)

        # Create individual plots for each file
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot for cosine distance dimension 0
        axes[0].scatter(euclidean_distances, cosine_distances_dim0, color=color_map(idx))
        axes[0].set_title(f"{filename} - Cosine Distance Dim 0")
        axes[0].set_xlabel("Euclidean Distance")
        axes[0].set_ylabel("Cosine Distance Dim 0")
        axes[0].set_xlim(min_euclidean, max_euclidean)
        axes[0].set_ylim(min_cosine_dim0, max_cosine_dim0)

        # Plot for cosine distance dimension 1
        axes[1].scatter(euclidean_distances, cosine_distances_dim1, color=color_map(idx))
        axes[1].set_title(f"{filename} - Cosine Distance Dim 1")
        axes[1].set_xlabel("Euclidean Distance")
        axes[1].set_ylabel("Cosine Distance Dim 1")
        axes[1].set_xlim(min_euclidean, max_euclidean)
        axes[1].set_ylim(min_cosine_dim1, max_cosine_dim1)

        plt.tight_layout()
        plt.show()

def plot_combined_results(data):
    combined_euclidean = []
    combined_cosine_dim0 = []
    combined_cosine_dim1 = []

    for _, results in enumerate(data.values()):
        combined_euclidean.extend([result['euclidean_distance'] for result in results])
        combined_cosine_dim0.extend([result['cosine_distance'][0] for result in results])
        combined_cosine_dim1.extend([result['cosine_distance'][1] for result in results])

    # Compute global min and max for the combined plot
    min_euclidean = min(combined_euclidean)
    max_euclidean = max(combined_euclidean)
    min_cosine_dim0 = min(combined_cosine_dim0)
    max_cosine_dim0 = max(combined_cosine_dim0)
    min_cosine_dim1 = min(combined_cosine_dim1)
    max_cosine_dim1 = max(combined_cosine_dim1)

    # Create combined plots for all files
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Combined plot for cosine distance dimension 0
    scatter = axes[0].scatter(combined_euclidean, combined_cosine_dim0, c=np.arange(len(combined_euclidean)), cmap='tab20')
    axes[0].set_title("Euclidean VS Cosine [Dim 0]")
    axes[0].set_xlabel("Euclidean Distance")
    axes[0].set_ylabel("Cosine Distance Dim 0")
    axes[0].set_xlim(min_euclidean - 3, max_euclidean + 3)
    axes[0].set_ylim(min_cosine_dim0 - 0.1, max_cosine_dim0 + 0.1)

    # Combined plot for cosine distance dimension 1
    axes[1].scatter(combined_euclidean, combined_cosine_dim1, c=np.arange(len(combined_euclidean)), cmap='tab20')
    axes[1].set_title("Euclidean VS Cosine [Dim 1]")
    axes[1].set_xlabel("Euclidean Distance")
    axes[1].set_ylabel("Cosine Distance Dim 1")
    axes[1].set_xlim(min_euclidean - 3, max_euclidean + 3)
    axes[1].set_ylim(min_cosine_dim1 - 0.1, max_cosine_dim1 + 0.1)

    # Create a color bar for the combined plot
    # plt.colorbar(scatter, ax=axes, label='Mutation Iteration Index')
    
    plt.tight_layout()
    plt.show()

# First, plot individual files
# plot_individual_files(data)

# Then, plot combined results of all files
plot_combined_results(data)
