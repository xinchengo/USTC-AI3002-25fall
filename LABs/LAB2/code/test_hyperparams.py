# -*- coding: utf-8 -*-
"""
Dedicated Hyperparameter Analysis: PCA Components & GMM Iterations/Tolerance
Optimized with Multithreading/Multiprocessing
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import yaml
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import davies_bouldin_score
from tqdm import tqdm
from joblib import Parallel, delayed

from util import set_seed, handle_results_path, ensure_dir
from dataloader import MNISTLoader
from submission import GMM, PCA

def get_parser():
    parser = argparse.ArgumentParser(description="Detailed Hyperparameter Analysis")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--results_path", type=str, default="./results/hyperparams_detailed", help="Path to save results")
    parser.add_argument("--gmm_components", type=int, default=10, help="Number of GMM components")
    parser.add_argument("--gmm_max_iter", type=int, default=1000, help="Max iterations for GMM")
    parser.add_argument("--n_trials", type=int, default=16, help="Number of trials per PCA setting")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs (-1 for all CPUs)")
    parser.add_argument("--plot_only", action="store_true", help="Skip training and only generate plots from existing results")
    parser.add_argument("--plot_dpi", type=int, default=300, help="DPI for saved plots")
    parser.add_argument("--gmm_tolerance", type=float, default=1e-5, help="Convergence tolerance for GMM")
    return parser

def run_single_trial(n_components, trial_idx, seed, traindata_pca, testdata_pca, testdata_raw, gmm_args):
    """
    Worker function to run a single GMM trial.
    """
    gmm = GMM(
        n_components=gmm_args['n_components'],
        max_iter=gmm_args['max_iter'],
        tol=1e-5, # Strict tolerance to capture full history
        reg_covar=1e-6,
        random_state=seed
    )
    
    trial_history = []
    
    def step_callback(model, it, improvement):
        # Predict on test set using current parameters
        try:
            labels = model.predict(testdata_pca)
            score = davies_bouldin_score(testdata_raw, labels)
        except Exception as e:
            score = float('nan')
        
        trial_history.append({
            'pca_components': n_components,
            'trial': trial_idx,
            'seed': seed,
            'iteration': it,
            'improvement': improvement,
            'db_score': score
        })

    gmm.fit(traindata_pca, callback=step_callback)
    return trial_history

def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    
    results_path = None

    if args.plot_only:
        # If plot only mode is enabled, load existing results
        if args.results_path is None:
            print("Error: --results_path must be specified in plot only mode.")
            return
        results_path = Path(args.results_path)

        # Load config from yaml file
        config_path = results_path / "config.yaml"
        if not config_path.exists():
            print(f"Error: Config file not found at {config_path}")
            return
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Re-parse arguments with config as defaults
        # This ensures: Config > Default, but CLI > Config
        parser.set_defaults(**config)
        args = parser.parse_args()
        
        print(f"Plot only mode: Loading results from: {results_path}")
        csv_path = results_path / "detailed_results.csv"
        if not csv_path.exists():
            print(f"Error: Results file not found at {csv_path}")
            return
        df = pd.read_csv(csv_path)
        
        # Infer pca_components_list from the dataframe
        if 'pca_components' in df.columns:
            pca_components_list = sorted(df['pca_components'].unique().tolist())
        else:
            print("Error: 'pca_components' column not found in results.")
            return
            
    else:
        # Normal mode: Run experiments and save results
        results_path = handle_results_path(args.results_path)
        ensure_dir(results_path)
        print(f"Results will be saved to: {results_path}")
        # Save config
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

        # Define PCA components range
        pca_components_list = list(range(2, 24, 1)) + list(range(24, 50, 2)) + list(range(50, 101, 5))
        
        all_results = []

        print(f"Starting Grid Search over PCA components: {pca_components_list}")
        print(f"Using {args.n_jobs} parallel jobs for trials.")
        
        for n_components in tqdm(pca_components_list, desc="PCA Loop"):
            # 1. Fit PCA once for this n_components
            pca = PCA(n_components=n_components)
            pca.fit(traindata_raw)
            traindata_pca = pca.transform(traindata_raw)
            testdata_pca = pca.transform(testdata_raw)
            
            # 2. Prepare args for parallel execution
            seeds = [np.random.randint(0, 100000) for _ in range(args.n_trials)]
            gmm_args = {
                'n_components': args.gmm_components,
                'max_iter': args.gmm_max_iter
            }
            
            # 3. Run trials in parallel
            # We use joblib to parallelize the trials
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(run_single_trial)(
                    n_components, 
                    i, 
                    seeds[i], 
                    traindata_pca, 
                    testdata_pca, 
                    testdata_raw, 
                    gmm_args
                ) for i in range(args.n_trials)
            )
            
            # 4. Collect results
            for res in results:
                all_results.extend(res)

        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Save raw data
        df.to_csv(results_path / "detailed_results.csv", index=False)
        print(f"Saved raw results to {results_path / 'detailed_results.csv'}")

    # ==========================================
    # Plotting
    # ==========================================
    print("Generating plots...")
    
    from matplotlib.ticker import MaxNLocator

    # 1. DBI vs PCA Components (at fixed tolerances)
    # tolerances = [1e-2, 1e-3, ... , until args.gmm_tolerance]
    tolerances = []
    tol = 1e-2
    while tol >= args.gmm_tolerance:
        tolerances.append(tol)
        tol /= 10
    
    plt.figure(figsize=(12, 8))
    
    for tol in tolerances:
        summary_data = []
        
        # Use groupby to handle trials cleanly
        # Group by PCA and Trial
        grouped = df.groupby(['pca_components', 'trial'])
        
        for name, group in grouped:
            pca_comp, trial = name
            group = group.sort_values('iteration')
            
            cutoff_row = group[group['improvement'] < tol]
            if not cutoff_row.empty:
                chosen_row = cutoff_row.iloc[0]
            else:
                chosen_row = group.iloc[-1]
            
            summary_data.append({
                'pca_components': pca_comp,
                'db_score': chosen_row['db_score']
            })
            
        summary_df = pd.DataFrame(summary_data)
        agg_df = summary_df.groupby('pca_components')['db_score'].agg(['mean', 'std']).reset_index()
        
        plt.errorbar(
            agg_df['pca_components'], 
            agg_df['mean'], 
            yerr=agg_df['std'], 
            label=f'tol={tol}',
            capsize=3,
            marker='x'
        )

    plt.xlabel('PCA Components')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('DBI vs PCA Components at Different Convergence Tolerances')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(results_path / "dbi_vs_pca_by_tol.png", dpi=args.plot_dpi, bbox_inches='tight')
    plt.close()

    # 2. DBI vs Tolerance
    plt.figure(figsize=(10, 6))
    
    selected_pcas = [10, 20, 40, 80]
    selected_pcas = [p for p in selected_pcas if p in pca_components_list]
    
    for pca_comp in selected_pcas:
        dbi_means = []
        dbi_stds = []
        pca_group = df[df['pca_components'] == pca_comp]
        
        for tol in tolerances:
            scores = []
            
            # Iterate over actual trials present in the data using groupby
            # This avoids the IndexError and is cleaner
            for trial_idx, trial_data in pca_group.groupby('trial'):
                trial_data = trial_data.sort_values('iteration')
                
                cutoff = trial_data[trial_data['improvement'] < tol]
                if not cutoff.empty:
                    scores.append(cutoff.iloc[0]['db_score'])
                else:
                    scores.append(trial_data.iloc[-1]['db_score'])
            
            if scores:
                dbi_means.append(np.mean(scores))
                dbi_stds.append(np.std(scores))
            else:
                dbi_means.append(np.nan)
                dbi_stds.append(np.nan)
            
        plt.errorbar(tolerances, dbi_means, yerr=dbi_stds, label=f'PCA={pca_comp}', marker='s', capsize=3)

    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.xlabel('Tolerance (log scale)')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('DBI vs Convergence Tolerance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_path / "dbi_vs_tol.png", dpi=args.plot_dpi, bbox_inches='tight')
    plt.close()

    # 3. DBI vs Iteration
    plt.figure(figsize=(10, 6))
    
    for pca_comp in selected_pcas:
        pca_data = df[df['pca_components'] == pca_comp]
        iter_stats = pca_data.groupby('iteration')['db_score'].agg(['mean', 'std']).reset_index()
        
        plt.plot(iter_stats['iteration'], iter_stats['mean'], label=f'PCA={pca_comp}')
        plt.fill_between(
            iter_stats['iteration'], 
            iter_stats['mean'] - iter_stats['std'], 
            iter_stats['mean'] + iter_stats['std'], 
            alpha=0.1
        )

    plt.xlabel('Iteration')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('DBI vs Iteration (Training Progress)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # Ensure x-axis ticks are integers
    plt.savefig(results_path / "dbi_vs_iter.png", dpi=args.plot_dpi, bbox_inches='tight')
    plt.close()

    print("Done!")

if __name__ == "__main__":
    main()
