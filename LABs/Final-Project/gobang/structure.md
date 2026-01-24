# Gobang AI Experiment Framework

This document describes the new unified experiment framework for the Gobang AI project.

## Directory Structure

```
gobang/
├── submission.py          # Model implementation
├── utils.py              # Utility functions
├── player.py             # Interactive player
├── tests/                # Testing and evaluation tools
│   ├── evaluator.py      # Model evaluation framework
│   ├── calculate_statistics.py  # Statistical analysis tools
│   └── conduct_experiment.py    # Unified experiment conductor (NEW!)
├── experiments/          # Experiment configurations and results
│   ├── plan.md           # Original experimental plan (moved from log.md)
│   ├── experiment_config.yaml  # YAML configuration for experiments (NEW!)
│   └── results/          # Experiment results directory
└── checkpoints/          # Trained model checkpoints
```

## New Experiment Framework

### 1. YAML Configuration

Experiments are now defined in `experiments/experiment_config.yaml` using a clear, structured format. This allows for easy modification and extension of experiments without changing code.

### 2. Unified Experiment Conductor

The new `tests/conduct_experiment.py` script provides a unified interface for running experiments:

```bash
# Run all experiments defined in the config
python -m tests.conduct_experiment --config experiments/experiment_config.yaml

# Run a specific experiment
python -m tests.conduct_experiment --config experiments/experiment_config.yaml --experiment baseline

# Specify custom results directory
python -m tests.conduct_experiment --config experiments/experiment_config.yaml --results-dir my_results
```

### 3. Configuration Schema

The YAML configuration supports:

- **Model training**: Define model architectures and training parameters
- **Evaluations**: Specify model vs model or model vs random comparisons
- **Statistical tests**: Define statistical analyses to run on results
- **Phased experiments**: Organize experiments in logical phases

## Benefits of the New Structure

1. **Maintainability**: Single tool for all experiments instead of multiple bloated scripts
2. **Flexibility**: Easy to modify experiments via configuration without code changes
3. **Reproducibility**: Clear, structured experiment definitions
4. **Scalability**: Easy to add new experiments or modify existing ones
5. **Organization**: Proper separation of concerns with dedicated directories

## Migration Notes

- Old scripts (`run_experiments.sh`, `rerun_evaluations.py`, `update_log_with_results.py`) have been removed
- The original experimental plan is now at `experiments/plan.md`
- All experiments are now conducted through the unified tool using YAML configs
- Results are saved in the `experiments/results/` directory