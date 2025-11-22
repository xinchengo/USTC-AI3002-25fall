# -*- coding: utf-8 -*-
"""
测试不同超参数组合对模型性能的影响
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用镜像站
import yaml
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import davies_bouldin_score
from tqdm import tqdm
import optuna

from util import set_seed, handle_results_path, ensure_dir
from dataloader import MNISTLoader
from submission import GMM, PCA

def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameter Search for PCA + GMM")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results_path", type=str, default="./results/hyperparams", help="Path to save results")
    parser.add_argument("--gmm_components", type=int, default=10, help="Number of GMM components (clusters)")
    parser.add_argument("--method", type=str, default="grid", choices=["grid", "bayes"], help="Search method: grid or bayes")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials for Bayesian optimization")
    return parser.parse_args()

def evaluate_model(params, traindata_raw, testdata_raw, args):
    """
    Train PCA and GMM with given params and evaluate on test set.
    Returns: dict with results
    """
    try:
        # 1. PCA
        pca = PCA(n_components=params['pca_components'])
        pca.fit(traindata_raw)
        traindata_pca = pca.transform(traindata_raw)
        testdata_pca = pca.transform(testdata_raw)
        
        # 2. GMM
        gmm = GMM(
            n_components=args.gmm_components,
            max_iter=params['gmm_max_iter'],
            tol=params['gmm_tol'],
            reg_covar=params['gmm_reg_covar'],
            random_state=args.seed
        )
        gmm.fit(traindata_pca)
        
        # 3. Evaluate
        labels = gmm.predict(testdata_pca)
        score = davies_bouldin_score(testdata_raw, labels)
        
        result_entry = params.copy()
        result_entry['db_score'] = score
        result_entry['converged'] = gmm.converged_
        result_entry['n_iter'] = gmm.n_iter_
        return result_entry
        
    except Exception as e:
        print(f"Error with params {params}: {e}")
        result_entry = params.copy()
        result_entry['db_score'] = float('inf')
        result_entry['error'] = str(e)
        return result_entry

def plot_results(df, output_dir):
    """
    Generate visualizations for hyperparameter search results.
    """
    output_dir = Path(output_dir)
    
    # 1. PCA Components vs DB Score
    if 'pca_components' in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Scatter plot of all trials
        plt.scatter(df['pca_components'], df['db_score'], alpha=0.4, color='gray', label='Individual Trials')
        
        # Calculate stats
        stats = df.groupby('pca_components')['db_score'].agg(['min', 'mean']).reset_index()
        
        # Plot Best (Min) score for each component count
        plt.plot(stats['pca_components'], stats['min'], 'r-o', linewidth=2, label='Best Score per Component')
        
        # Plot Mean score
        plt.plot(stats['pca_components'], stats['mean'], 'b--', label='Mean Score')
        
        # Highlight global best
        best_idx = df['db_score'].idxmin()
        plt.scatter(df.loc[best_idx, 'pca_components'], df.loc[best_idx, 'db_score'], 
                    color='gold', marker='*', s=300, edgecolors='black', zorder=10, label='Global Best')

        plt.xlabel('PCA Components')
        plt.ylabel('Davies-Bouldin Score (Lower is Better)')
        plt.title('Impact of PCA Components on Clustering Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "pca_vs_dbscore.png")
        plt.close()

    # 2. Reg Covar vs DB Score
    if 'gmm_reg_covar' in df.columns:
        plt.figure(figsize=(10, 6))
        # Use scatter plot for continuous variables
        plt.scatter(df['gmm_reg_covar'], df['db_score'], alpha=0.6, c='blue', label='Trial')
        
        # Highlight best point
        best_idx = df['db_score'].idxmin()
        plt.scatter(df.loc[best_idx, 'gmm_reg_covar'], df.loc[best_idx, 'db_score'], 
                    color='red', marker='*', s=200, label='Best')
        
        plt.xscale('log')
        plt.xlabel('GMM Regularization Covariance (log scale)')
        plt.ylabel('Davies-Bouldin Score')
        plt.title('Impact of Regularization on Clustering Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "regcovar_vs_dbscore.png")
        plt.close()

    # 3. Max Iter vs DB Score
    if 'gmm_max_iter' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['gmm_max_iter'], df['db_score'], alpha=0.6, c='green', label='Trial')
        
        # Highlight best point
        best_idx = df['db_score'].idxmin()
        plt.scatter(df.loc[best_idx, 'gmm_max_iter'], df.loc[best_idx, 'db_score'], 
                    color='red', marker='*', s=200, label='Best')

        plt.xlabel('GMM Max Iterations')
        plt.ylabel('Davies-Bouldin Score')
        plt.title('Impact of Max Iterations on Clustering Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "maxiter_vs_dbscore.png")
        plt.close()

def main():
    args = get_args()
    set_seed(args.seed)
    
    # Handle results path
    results_path = handle_results_path(args.results_path)
    ensure_dir(results_path)

    # Save args
    with open(results_path / "config.yaml", "w") as f:
        yaml.dump(vars(args), f)

    # Load Data
    print("Loading dataset...")
    dataloader = MNISTLoader()
    dataset = dataloader.load()
    
    trainset = dataset["train"].to_pandas()
    traindata_raw = np.vstack(trainset["image1D"].to_numpy())
    
    testset = dataset["test"].to_pandas()
    testdata_raw = np.vstack(testset["image1D"].to_numpy())

    print(f"Training data shape: {traindata_raw.shape}")
    print(f"Test data shape: {testdata_raw.shape}")

    results = []

    if args.method == "grid":
        # Define Hyperparameter Grid
        param_grid = {
            'pca_components': [20, 40, 60, 80],
            'gmm_max_iter': [50, 100],
            'gmm_tol': [1e-3, 1e-4],
            'gmm_reg_covar': [1e-6, 1e-4]
        }
        grid = ParameterGrid(param_grid)
        print(f"Starting Grid Search with {len(grid)} combinations...")
        
        for params in tqdm(grid):
            res = evaluate_model(params, traindata_raw, testdata_raw, args)
            results.append(res)

    elif args.method == "bayes":
        print(f"Starting Bayesian Optimization with {args.n_trials} trials...")
        
        def objective(trial):
            params = {
                'pca_components': trial.suggest_int('pca_components', 2, 200, log=True),
                'gmm_max_iter': trial.suggest_int('gmm_max_iter', 30, 150, step=10),
                'gmm_tol': trial.suggest_float('gmm_tol', 1e-6, 1e-2, log=True),
                'gmm_reg_covar': trial.suggest_float('gmm_reg_covar', 1e-12, 1e-2, log=True)
            }
            
            res = evaluate_model(params, traindata_raw, testdata_raw, args)
            
            # Store result in list for later analysis
            results.append(res)
            
            return res['db_score']

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=args.n_trials)
        
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    csv_path = results_path / "hyperparameter_search_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    json_path = results_path / "hyperparameter_search_results.json"
    df.to_json(json_path, orient="records", indent=4)
    print(f"Results saved to {json_path}")

    # Find best parameters
    if not df.empty and 'db_score' in df.columns:
        best_run = df.loc[df['db_score'].idxmin()]
        print("\n" + "="*50)
        print("Best Hyperparameters found:")
        print(best_run)
        print("="*50)
        
        # Save best params separately
        with open(results_path / "best_params.json", "w") as f:
            best_dict = best_run.to_dict()
            for k, v in best_dict.items():
                if isinstance(v, (np.integer, np.floating)):
                    best_dict[k] = float(v) if isinstance(v, np.floating) else int(v)
            json.dump(best_dict, f, indent=4)

    # Visualization
    print("Generating visualizations...")
    plot_results(df, results_path)
    print(f"Visualizations saved to {results_path}")

if __name__ == "__main__":
    main()

