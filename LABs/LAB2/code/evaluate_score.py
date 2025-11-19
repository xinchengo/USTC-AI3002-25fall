import os
import argparse
import yaml
from pathlib import Path
import numpy as np
from sklearn.metrics import davies_bouldin_score

from dataloader import MNISTLoader
from submission import PCA, GMM
from autoencoder import AE
from util import ae_encode

def main():
    parser = argparse.ArgumentParser(description="Evaluate Davies-Bouldin Score")
    parser.add_argument("--results_path", type=str, required=True, help="Path to the results directory")
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    loader = MNISTLoader()
    dataset = loader.load()
    test_data = dataset["test"]
    test_data_raw1d = test_data["image1D"]
    
    # Normalize data if needed (usually GMM expects float data, and visualization.py didn't seem to normalize explicitly other than what might be in data_preprocess, but let's check submission.py's data_preprocess again. Wait, submission.py's data_preprocess just flattens. Usually pixel values are 0-255. PCA/GMM might handle it, but usually 0-1 is better. However, I should follow what train.py does.)
    
    # Let's check train.py to see if it normalizes data.
    
    # Load config
    config_path = Path(args.results_path) / "config.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dr_method = cfg.get('dr_method', 'pca')
    print(f"Dimensionality Reduction Method: {dr_method}")

    # Load models and predict labels
    if dr_method == 'pca':
        print("Loading PCA model...")
        pca_path = os.path.join(args.results_path, "pca.npz")
        if not os.path.exists(pca_path):
             print(f"Error: PCA model not found at {pca_path}")
             return
        pca = PCA.from_pretrained(pca_path)
        
        print("Transforming data with PCA...")
        test_data_reduced = pca.transform(test_data_raw1d)
        
        print("Loading GMM model...")
        gmm_path = os.path.join(args.results_path, "gmm")
        gmm = GMM.from_pretrained(gmm_path)
        
        print("Predicting clusters...")
        labels = gmm.predict(test_data_reduced)
        
    elif dr_method == 'autoencoder':
        print("Loading Autoencoder model...")
        # Use the same pretrained model as in train.py
        ae = AE.from_pretrained("H2O123h2o/mnist-autoencoder")
        
        print("Transforming data with Autoencoder...")
        # AE expects 2D images (N, 1, 28, 28) or similar, check train.py
        # train.py: traindata_raw = np.stack(trainset["image2D"])
        test_data_2d = np.stack(test_data["image2D"])
        test_data_reduced = ae_encode(ae, test_data_2d)
        
        print("Loading GMM model...")
        gmm_path = os.path.join(args.results_path, "gmm")
        gmm = GMM.from_pretrained(gmm_path)
        
        print("Predicting clusters...")
        labels = gmm.predict(test_data_reduced)
    else:
        print(f"Unknown dimensionality reduction method: {dr_method}")
        return
    
    # Calculate DB score
    print("Calculating Davies-Bouldin Score...")
    # The score is calculated on the ORIGINAL data (784 dim) as per README.
    # "我们在原始数据空间 (784 维) 中使用 davies_bouldin_score度量聚类的性能"
    
    # Note: davies_bouldin_score might be slow on full dataset.
    score = davies_bouldin_score(test_data_raw1d, labels)
    
    print(f"Davies-Bouldin Score: {score:.4f}")

if __name__ == "__main__":
    main()
