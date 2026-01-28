#!/usr/bin/env python3
"""
Unified Experiment Conductor Tool

This module provides a unified interface for conducting experiments based on YAML configuration files.
It handles model training, evaluation, and statistical analysis in a standardized way.
"""

import argparse
import yaml
import subprocess
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import re


def run_training(model_params, checkpoint_dir):
    """
    Run model training with given parameters
    
    Args:
        model_params (dict): Parameters for model training
        checkpoint_dir (str): Directory to save checkpoints
        
    Returns:
        str: Path to the trained model
    """
    print(f"Starting training with params: {model_params}")
    
    # Prepare command arguments
    cmd = [
        sys.executable, "submission.py",
        "--num_episodes", str(model_params.get("num_episodes", model_params.get("num_episodes", 1000))),
        "--checkpoint", str(model_params.get("checkpoint_interval", model_params.get("checkpoint", 200))),
    ]
    
    # Add optional parameters
    if model_params.get("use_wandb", model_params.get("use_wandb", False)):
        cmd.extend(["--use_wandb"])
        if model_params.get("wandb_name"):
            cmd.extend(["--wandb_name", model_params.get("wandb_name")])
    
    if model_params.get("model_type"):
        cmd.extend(["--model-type", model_params["model_type"]])
    
    if "extra_specs" in model_params:
        cmd.extend(["--extra-specs", json.dumps(model_params["extra_specs"])])
    
    # Add additional parameters if they exist
    if "lr" in model_params:
        cmd.extend(["--lr", str(model_params["lr"])])
    if "gamma" in model_params:
        cmd.extend(["--gamma", str(model_params["gamma"])])
    if "reward_type" in model_params:
        cmd.extend(["--reward-type", str(model_params["reward_type"])])
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the training
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Training failed with error: {result.stderr}")
        return None
    
    print(f"Training completed successfully: {result.stdout[-500:]}")  # Print last 500 chars of output

    # Prefer the stable wandb symlink if it exists, otherwise parse stdout or fall back
    actual_checkpoint_dir = None

    wandb_name = model_params.get("wandb_name")
    if wandb_name:
        # Normalize wandb_name: dots to underscores (submission.py does this for directory names)
        normalized_name = re.sub(r'[^\w\-_]', '_', wandb_name)
        wandb_link = Path("checkpoints") / normalized_name
        if wandb_link.exists():
            actual_checkpoint_dir = str(wandb_link)
            print(f"Found wandb symlink: {actual_checkpoint_dir}")

    if actual_checkpoint_dir is None:
        match = re.search(r"Models will be saved to:\s*(.+)", result.stdout)
        if match:
            candidate = match.group(1).strip()
            if Path(candidate).exists():
                actual_checkpoint_dir = candidate
                print(f"Parsed checkpoint dir from output: {actual_checkpoint_dir}")

    if actual_checkpoint_dir is None:
        # If no wandb symlink or stdout parsing, use the checkpoint_dir we passed in
        if Path(checkpoint_dir).exists():
            actual_checkpoint_dir = checkpoint_dir
            print(f"Using checkpoint dir: {actual_checkpoint_dir}")
        else:
            print(f"Warning: Could not find checkpoint directory for {wandb_name or 'model'}")
            actual_checkpoint_dir = checkpoint_dir

    return actual_checkpoint_dir


def run_evaluation(eval_params, checkpoint_map=None):
    """
    Run model evaluation based on parameters
    
    Args:
        eval_params (dict): Parameters for evaluation
        
    Returns:
        dict: Evaluation results
    """
    print(f"Starting evaluation with params: {eval_params}")
    
    # Determine player paths
    player1_path = eval_params.get("player1_path")
    player2_path = eval_params.get("player2_path")
    
    # If player1 is a named model, resolve to checkpoint
    if eval_params.get("player1") and not player1_path:
        model_name = eval_params["player1"]
        checkpoint_name = eval_params.get("player1_checkpoint", "model_999.pth")
        mapped_dir = (checkpoint_map or {}).get(model_name)
        if mapped_dir:
            player1_path = os.path.join(mapped_dir, checkpoint_name)
        else:
            # Try to find wandb symlink by looking for common patterns
            # Model names like baseline_cnn_d3_w64 might have symlinks like baseline-cnn-d3-w64-late
            potential_symlink = model_name.replace("_", "-") + "-late"
            symlink_path = Path("checkpoints") / potential_symlink
            if symlink_path.exists():
                player1_path = os.path.join(str(symlink_path), checkpoint_name)
                checkpoint_map[model_name] = str(symlink_path)  # Cache for future use
                print(f"Found symlink for {model_name}: {symlink_path}")
            else:
                player1_path = f"checkpoints/{model_name}/{checkpoint_name}"
    
    # If player2 is a named model, resolve to checkpoint
    if eval_params.get("player2") and not player2_path:
        model_name = eval_params["player2"]
        checkpoint_name = eval_params.get("player2_checkpoint", "model_999.pth")
        mapped_dir = (checkpoint_map or {}).get(model_name)
        if mapped_dir:
            player2_path = os.path.join(mapped_dir, checkpoint_name)
        else:
            # Try to find wandb symlink by looking for common patterns
            potential_symlink = model_name.replace("_", "-") + "-late"
            symlink_path = Path("checkpoints") / potential_symlink
            if symlink_path.exists():
                player2_path = os.path.join(str(symlink_path), checkpoint_name)
                checkpoint_map[model_name] = str(symlink_path)  # Cache for future use
                print(f"Found symlink for {model_name}: {symlink_path}")
            else:
                player2_path = f"checkpoints/{model_name}/{checkpoint_name}"
    
    # If player2 is 'random', use baseline type
    if eval_params.get("player2") == "random":
        player2_path = "random"
        player2_type = "baseline"
    else:
        player2_type = "checkpoint"
    
    cmd = [
        sys.executable, "-m", "tests.evaluator",
        "--player1_path", player1_path,
        "--player1_type", "checkpoint",
        "--player2_path", player2_path,
        "--player2_type", player2_type,
        "--episodes", str(eval_params.get("episodes", 500))
    ]
    
    print(f"Running evaluation command: {' '.join(cmd)}")
    
    # Run the evaluation
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Evaluation failed with error: {result.stderr}")
        return {"error": result.stderr}
    
    print(f"Evaluation completed: {result.stdout[-500:]}")  # Print last 500 chars of output
    
    # Parse results from output
    output = result.stdout
    wins1 = 0
    wins2 = 0
    ties = 0
    
    # Extract win counts from output
    if "Player 1 (checkpoint) wins:" in output:
        import re
        wins_match = re.search(r"Player 1 \(checkpoint\) wins: (\d+)", output)
        if wins_match:
            wins1 = int(wins_match.group(1))
        
        wins2_match = re.search(r"Player 2 \([^)]*\) wins: (\d+)", output)
        if wins2_match:
            wins2 = int(wins2_match.group(1))
        
        ties_match = re.search(r"Ties: (\d+)", output)
        if ties_match:
            ties = int(ties_match.group(1))
    
    total_games = wins1 + wins2 + ties
    win_rate1 = wins1 / total_games if total_games > 0 else 0
    win_rate2 = wins2 / total_games if total_games > 0 else 0
    
    return {
        "player1_wins": wins1,
        "player2_wins": wins2, 
        "ties": ties,
        "total_games": total_games,
        "player1_win_rate": win_rate1,
        "player2_win_rate": win_rate2,
        "output": output
    }


def run_statistical_analysis(stats_params):
    """
    Run statistical analysis based on parameters
    
    Args:
        stats_params (dict): Parameters for statistical analysis
        
    Returns:
        dict: Statistical analysis results
    """
    print(f"Running statistical analysis with params: {stats_params}")
    
    # Prepare command arguments for statistical analysis
    cmd = [sys.executable, "-m", "tests.calculate_statistics"]
    
    # Add statistical test parameters if specified
    if stats_params.get("statistical_test") == "binomial_test":
        cmd.extend([
            "--experiment", "convergence",
            "--x1", str(stats_params.get("successes", 265)),
            "--n1", str(stats_params.get("trials", 500)),
            "--p_null", str(stats_params.get("null_prob", 0.5))
        ])
    elif stats_params.get("statistical_test") == "one_proportion_test":
        cmd.extend([
            "--experiment", "vs-random",
            "--x1", str(stats_params.get("successes", 485)),
            "--n1", str(stats_params.get("trials", 500)),
            "--p_null", str(stats_params.get("null_prob", 0.5))
        ])
    elif stats_params.get("statistical_test") == "two_proportion_z_test":
        cmd.extend([
            "--experiment", "comparison",
            "--x1", str(stats_params.get("x1", 160)),
            "--x2", str(stats_params.get("x2", 340)),
            "--n1", str(stats_params.get("n1", 500)),
            "--n2", str(stats_params.get("n2", 500))
        ])
    else:
        # Default: just run the statistics module to show examples
        pass
    
    print(f"Running statistical analysis command: {' '.join(cmd)}")
    
    # Run the statistical analysis
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Statistical analysis failed with error: {result.stderr}")
        return {"error": result.stderr}
    
    print(f"Statistical analysis completed: {result.stdout[-500:]}")  # Print last 500 chars
    
    return {
        "output": result.stdout,
        "error": result.stderr if result.stderr else None
    }


def conduct_experiment_phase(phase_config, results_dir, checkpoint_map=None, retry_failed=False):
    """
    Conduct a single experiment phase
    
    Args:
        phase_config (dict): Configuration for the phase
        results_dir (str): Directory to save results
        checkpoint_map (dict): Map of model names to checkpoint directories
        retry_failed (bool): If True, only run failed/missing experiments
        
    Returns:
        dict: Results of the phase
    """
    phase_name = phase_config["name"]
    print(f"Conducting phase: {phase_name}")
    
    phase_results = {}
    checkpoint_map = checkpoint_map if checkpoint_map is not None else {}
    
    # Load existing results if retry_failed is True
    phase_result_file = os.path.join(results_dir, f"{phase_name}_results.json")
    if retry_failed and os.path.exists(phase_result_file):
        with open(phase_result_file, 'r') as f:
            phase_results = json.load(f)
        print(f"Loaded existing results from {phase_result_file}")
    
    # Handle different types of phases
    if "models" in phase_config:
        # Training phase
        for model_config in phase_config["models"]:
            model_name = model_config["name"]
            model_params = model_config["parameters"]
            
            # Skip if retry_failed is True and this model already succeeded
            if retry_failed and model_name in phase_results:
                existing = phase_results[model_name]
                if existing.get("status") == "completed" and "checkpoint_dir" in existing:
                    print(f"Skipping {model_name} (already completed)")
                    checkpoint_map[model_name] = existing["checkpoint_dir"]
                    continue
            
            print(f"Training model: {model_name}")
            
            # Create checkpoint directory for this model
            checkpoint_dir = f"checkpoints/exp_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Run training
            model_path = run_training(model_params, checkpoint_dir)
            if model_path:
                phase_results[model_name] = {
                    "status": "completed",
                    "checkpoint_dir": model_path
                }
                checkpoint_map[model_name] = model_path
            else:
                phase_results[model_name] = {
                    "status": "failed",
                    "error": "Training failed"
                }
    
    elif "evaluations" in phase_config:
        # Evaluation phase
        for eval_config in phase_config["evaluations"]:
            eval_name = eval_config["name"]
            
            # Skip if retry_failed is True and this evaluation already succeeded
            if retry_failed and eval_name in phase_results:
                existing = phase_results[eval_name]
                if "error" not in existing and existing.get("total_games", 0) > 0:
                    print(f"Skipping {eval_name} (already completed)")
                    continue
            
            print(f"Running evaluation: {eval_name}")
            
            # Run evaluation
            eval_result = run_evaluation(eval_config, checkpoint_map)
            phase_results[eval_name] = eval_result
    
    elif "statistical_test" in phase_config:
        # Statistical analysis phase
        stats_result = run_statistical_analysis(phase_config)
        phase_results[f"stats_{phase_config.get('name', 'analysis')}"] = stats_result
    
    # Save phase results
    phase_result_file = os.path.join(results_dir, f"{phase_name}_results.json")
    with open(phase_result_file, 'w') as f:
        json.dump(phase_results, f, indent=2)
    
    return phase_results


def conduct_experiment(experiment_config, results_dir, retry_failed=False):
    """
    Conduct a complete experiment based on configuration
    
    Args:
        experiment_config (dict): Configuration for the experiment
        results_dir (str): Directory to save results
        retry_failed (bool): If True, only run failed/missing experiments
        
    Returns:
        dict: Overall results of the experiment
    """
    exp_name = experiment_config["name"]
    print(f"Starting experiment: {exp_name}")
    
    # Create results directory for this experiment
    exp_results_dir = os.path.join(results_dir, exp_name.replace(" ", "_").replace("/", "_"))
    os.makedirs(exp_results_dir, exist_ok=True)
    
    # Load existing experiment results if retry_failed
    exp_result_file = os.path.join(exp_results_dir, "experiment_results.json")
    if retry_failed and os.path.exists(exp_result_file):
        with open(exp_result_file, 'r') as f:
            experiment_results = json.load(f)
        print(f"Loaded existing experiment results from {exp_result_file}")
    else:
        experiment_results = {
            "experiment_name": exp_name,
            "start_time": datetime.now().isoformat(),
            "phases": {}
        }
    
    checkpoint_map = {}
    
    # Rebuild checkpoint_map from existing results
    if retry_failed:
        for phase_name, phase_data in experiment_results.get("phases", {}).items():
            for item_name, item_data in phase_data.items():
                if isinstance(item_data, dict) and "checkpoint_dir" in item_data:
                    checkpoint_map[item_name] = item_data["checkpoint_dir"]
                    print(f"Loaded checkpoint mapping: {item_name} -> {item_data['checkpoint_dir']}")

    # Execute each phase
    for phase_config in experiment_config.get("phases", []):
        phase_name = phase_config["name"]
        print(f"Starting phase: {phase_name}")
        
        phase_results = conduct_experiment_phase(phase_config, exp_results_dir, checkpoint_map, retry_failed)
        experiment_results["phases"][phase_name] = phase_results
    
    # Save overall experiment results
    exp_result_file = os.path.join(exp_results_dir, "experiment_results.json")
    with open(exp_result_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    experiment_results["end_time"] = datetime.now().isoformat()
    print(f"Experiment {exp_name} completed")
    
    return experiment_results


def main():
    parser = argparse.ArgumentParser(description="Unified Experiment Conductor Tool")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the experiment configuration YAML file")
    parser.add_argument("--results-dir", type=str, default="experiments/results",
                        help="Directory to save experiment results")
    parser.add_argument("--experiment", type=str, 
                        help="Specific experiment to run (if not specified, runs all experiments)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Only retry failed or missing experiments (skips successful ones)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Determine which experiments to run
    experiments_to_run = []
    if args.experiment:
        # Run specific experiment
        if args.experiment in config.get("experiments", {}):
            exp_config = config["experiments"][args.experiment]
            exp_config["name"] = args.experiment
            experiments_to_run.append(exp_config)
        else:
            print(f"Experiment '{args.experiment}' not found in configuration")
            sys.exit(1)
    else:
        # Run all experiments
        for exp_name, exp_config in config.get("experiments", {}).items():
            exp_config["name"] = exp_name
            experiments_to_run.append(exp_config)
    
    # Run experiments
    all_results = {}
    for exp_config in experiments_to_run:
        exp_name = exp_config["name"]
        print(f"\n{'='*50}")
        print(f"RUNNING EXPERIMENT: {exp_name}")
        if args.retry_failed:
            print(f"MODE: Retry failed experiments only")
        print(f"{'='*50}")
        
        exp_results = conduct_experiment(exp_config, args.results_dir, retry_failed=args.retry_failed)
        all_results[exp_name] = exp_results
    
    # Save overall results
    overall_results_file = os.path.join(args.results_dir, "overall_results.json")
    with open(overall_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll experiments completed. Results saved to {overall_results_file}")


if __name__ == "__main__":
    main()