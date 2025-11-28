# -*- coding: utf-8 -*-
"""
Visualize Eigenvectors (Eigenfaces) and Eigenvalues (Variance Contribution) of MNIST
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from dataloader import MNISTLoader
from submission import PCA
from util import ensure_dir

def main():
    # 1. Load Data
    print("Loading MNIST dataset...")
    loader = MNISTLoader()
    dataset = loader.load()
    trainset = dataset["train"]
    
    # Convert to numpy array and flatten
    # shape: (N, 784)
    print("Preprocessing data...")
    X = np.stack(trainset["image1D"]).astype(np.float64)
    N, D = X.shape
    print(f"Data shape: {X.shape}")

    output_dir = Path("results/eigen_viz")
    ensure_dir(output_dir)
    pca_cache_path = output_dir / "pca_model.npz"

    if pca_cache_path.exists():
        print(f"Loading cached PCA model from {pca_cache_path}...")
        pca = PCA.from_pretrained(str(pca_cache_path))
    else:
        # 2. Perform Full PCA
        print("Performing Full PCA (this may take a moment)...")
        # We want all components to see the full spectrum
        n_components = min(N, D) 
        # This lines require 64GB+ RAM !!!
        pca = PCA(n_components=n_components, full_matrices=True)
        pca.fit(X)
        print(f"Saving PCA model to {pca_cache_path}...")
        pca.save_pretrained(str(pca_cache_path))
    
    print(f"Saving results to {output_dir}")

    # 3. Visualize First 64 Eigenvectors (Eigenfaces)
    print("Visualizing Eigenfaces...")
    n_row = 8
    n_col = 8
    n_eigenfaces = n_row * n_col
    
    # Get the first 64 components
    eigenfaces = pca.components_[:n_eigenfaces]
    
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(wspace=0.05, hspace=0.05) # Make it compact
    for i in range(n_eigenfaces):
        plt.subplot(n_row, n_col, i + 1)
        # Reshape to 28x28
        # Note: Eigenvectors can be negative, so we just visualize the pattern.
        # cmap='gray' maps low values to black, high to white.
        plt.imshow(eigenfaces[i].reshape(28, 28), cmap='gray')
        # plt.title(f"PC {i+1}", fontsize=8) # Removed title for compactness
        plt.axis('off')
    
    plt.suptitle("First 64 Eigenfaces (Principal Components)", fontsize=16, y=0.95)
    # plt.tight_layout() # tight_layout might conflict with custom subplots_adjust
    plt.savefig(output_dir / "eigenfaces.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved eigenfaces.png")

    # 4. Visualize Variance Contribution (Eigenvalues)
    print("Visualizing Variance Contribution...")
    
    explained_variance_ratio = pca.explained_variance_ratio_
    components_range = np.arange(1, len(explained_variance_ratio) + 1)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Linear Scale
    ax1.plot(components_range, explained_variance_ratio, linewidth=2)
    ax1.set_title("Variance Contribution (Linear Scale)")
    ax1.set_xlabel("Principal Component Index")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.grid(True, alpha=0.3)
    
    # Highlight the "elbow" or first few components
    ax1.scatter(components_range[:10], explained_variance_ratio[:10], color='red', s=20, zorder=5)

    # Plot 2: Log Scale
    ax2.plot(components_range, explained_variance_ratio, linewidth=2)
    ax2.set_yscale('log')
    ax2.set_title("Variance Contribution (Log Scale)")
    ax2.set_xlabel("Principal Component Index")
    ax2.set_ylabel("Explained Variance Ratio (log)")
    ax2.grid(True, alpha=0.3, which="both")
    
    plt.suptitle("Explained Variance Ratio by Principal Components", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "variance_contribution.png", dpi=300)
    plt.close()
    print("Saved variance_contribution.png")
    
    # Optional: Cumulative Variance Plot
    plt.figure(figsize=(10, 6))
    cumulative_variance = np.cumsum(explained_variance_ratio)
    plt.plot(components_range, cumulative_variance, linewidth=2)
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.grid(True, alpha=0.3)
    
    # Add lines for 90%, 95%, 99% variance
    for threshold in [0.90, 0.95, 0.99]:
        idx = np.where(cumulative_variance >= threshold)[0][0]
        plt.axvline(x=idx+1, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
        plt.text(idx+5, threshold-0.02, f"{threshold*100}% ({idx+1} comps)", color='r')
        
    # Add lines for explained variance of the first 2, 10, 20 components
    for threshold in [2, 10, 20]:
        if threshold <= len(explained_variance_ratio):
            cum_var = cumulative_variance[threshold - 1]
            plt.axvline(x=threshold, color='r', linestyle='--', alpha=0.5)
            plt.axhline(y=cum_var, color='r', linestyle='--', alpha=0.5)
            # plt.text(threshold+1, cum_var-0.02, f"First {threshold} comps: {cum_var*100:.2f}%", color='g')
            plt.text(threshold+5, cum_var-0.02, f"{cum_var*100:.2f}% ({threshold} comps)", color='r')
    
    plt.savefig(output_dir / "cumulative_variance.png", dpi=300)
    plt.close()
    print("Saved cumulative_variance.png")

    print("Done!")

if __name__ == "__main__":
    main()
