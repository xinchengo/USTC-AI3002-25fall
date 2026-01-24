#!/usr/bin/env python3
"""
Script to find the correct checkpoint paths based on hyperparameters.
"""

import os
import glob
import json

def find_checkpoint_by_hyperparams(target_params):
    """
    Find checkpoint directory based on hyperparameters.
    
    Args:
        target_params: dict with keys like depth, channels, batch_norm, action_injection, num_episodes
    
    Returns:
        Path to the checkpoint directory, or None if not found
    """
    checkpoint_dirs = glob.glob("checkpoints/cnn_gobang_model_*")
    
    for chkpt_dir in checkpoint_dirs:
        hyperparams_file = os.path.join(chkpt_dir, "hyperparameters.txt")
        if os.path.exists(hyperparams_file):
            with open(hyperparams_file, 'r') as f:
                lines = f.readlines()
                hyperparams = {}
                for line in lines:
                    if ': ' in line:
                        key, value = line.strip().split(': ', 1)
                        try:
                            # Try to evaluate the value as Python literal
                            hyperparams[key] = eval(value)
                        except:
                            # If eval fails, keep as string
                            hyperparams[key] = value
            
            # Check if this matches our target
            match = True
            for key, value in target_params.items():
                if key == 'extra_specs':
                    if 'extra_specs' not in hyperparams:
                        match = False
                        break
                    for spec_key, spec_value in value.items():
                        if spec_key not in hyperparams['extra_specs'] or hyperparams['extra_specs'][spec_key] != spec_value:
                            match = False
                            break
                elif key == 'num_episodes':
                    if 'num_episodes' not in hyperparams or hyperparams['num_episodes'] != value:
                        match = False
                        break
                else:
                    if key not in hyperparams or hyperparams[key] != value:
                        match = False
                        break
            
            if match:
                return chkpt_dir
    
    return None

def main():
    print("Finding checkpoint paths...")
    
    # Find baseline CNN (depth=3, channels=64, late injection, 1000 episodes)
    baseline_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'extra_specs': {
            'depth': 3,
            'channels': 64,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    baseline_path = find_checkpoint_by_hyperparams(baseline_params)
    print(f"Baseline CNN (d=3, w=64): {baseline_path}")
    
    # Find deep CNN (depth=5, channels=64, late injection, 1500 episodes)
    deep_params = {
        'model_type': 'cnn',
        'num_episodes': 1500,
        'extra_specs': {
            'depth': 5,
            'channels': 64,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    deep_path = find_checkpoint_by_hyperparams(deep_params)
    print(f"Deep CNN (d=5, w=64): {deep_path}")
    
    # Find depth=2 model
    d2_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'extra_specs': {
            'depth': 2,
            'channels': 64,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    d2_path = find_checkpoint_by_hyperparams(d2_params)
    print(f"Depth=2 model: {d2_path}")
    
    # Find depth=4 model
    d4_params = {
        'model_type': 'cnn',
        'num_episodes': 1200,
        'extra_specs': {
            'depth': 4,
            'channels': 64,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    d4_path = find_checkpoint_by_hyperparams(d4_params)
    print(f"Depth=4 model: {d4_path}")
    
    # Find depth=6 model
    d6_params = {
        'model_type': 'cnn',
        'num_episodes': 1800,
        'extra_specs': {
            'depth': 6,
            'channels': 64,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    d6_path = find_checkpoint_by_hyperparams(d6_params)
    print(f"Depth=6 model: {d6_path}")
    
    # Find width=16 model (depth=3)
    w16_d3_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'extra_specs': {
            'depth': 3,
            'channels': 16,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    w16_d3_path = find_checkpoint_by_hyperparams(w16_d3_params)
    print(f"Width=16 model (d=3): {w16_d3_path}")
    
    # Find width=32 model (depth=3)
    w32_d3_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'extra_specs': {
            'depth': 3,
            'channels': 32,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    w32_d3_path = find_checkpoint_by_hyperparams(w32_d3_params)
    print(f"Width=32 model (d=3): {w32_d3_path}")
    
    # Find width=128 model (depth=3)
    w128_d3_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'extra_specs': {
            'depth': 3,
            'channels': 128,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    w128_d3_path = find_checkpoint_by_hyperparams(w128_d3_params)
    print(f"Width=128 model (d=3): {w128_d3_path}")
    
    # Find width=256 model (depth=3)
    w256_d3_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'extra_specs': {
            'depth': 3,
            'channels': 256,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    w256_d3_path = find_checkpoint_by_hyperparams(w256_d3_params)
    print(f"Width=256 model (d=3): {w256_d3_path}")
    
    # Find width=16 model (depth=5)
    w16_d5_params = {
        'model_type': 'cnn',
        'num_episodes': 1500,
        'extra_specs': {
            'depth': 5,
            'channels': 16,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    w16_d5_path = find_checkpoint_by_hyperparams(w16_d5_params)
    print(f"Width=16 model (d=5): {w16_d5_path}")
    
    # Find width=32 model (depth=5)
    w32_d5_params = {
        'model_type': 'cnn',
        'num_episodes': 1500,
        'extra_specs': {
            'depth': 5,
            'channels': 32,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    w32_d5_path = find_checkpoint_by_hyperparams(w32_d5_params)
    print(f"Width=32 model (d=5): {w32_d5_path}")
    
    # Find width=128 model (depth=5)
    w128_d5_params = {
        'model_type': 'cnn',
        'num_episodes': 1500,
        'extra_specs': {
            'depth': 5,
            'channels': 128,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    w128_d5_path = find_checkpoint_by_hyperparams(w128_d5_params)
    print(f"Width=128 model (d=5): {w128_d5_path}")
    
    # Find width=256 model (depth=5)
    w256_d5_params = {
        'model_type': 'cnn',
        'num_episodes': 1500,
        'extra_specs': {
            'depth': 5,
            'channels': 256,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    w256_d5_path = find_checkpoint_by_hyperparams(w256_d5_params)
    print(f"Width=256 model (d=5): {w256_d5_path}")
    
    # Find batch normalization model
    bn_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'extra_specs': {
            'depth': 3,
            'channels': 64,
            'batch_norm': True,
            'action_injection': 'late'
        }
    }
    bn_path = find_checkpoint_by_hyperparams(bn_params)
    print(f"BatchNorm model: {bn_path}")
    
    # Find no injection model
    none_inj_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'extra_specs': {
            'depth': 3,
            'channels': 64,
            'batch_norm': False,
            'action_injection': 'none'
        }
    }
    none_inj_path = find_checkpoint_by_hyperparams(none_inj_params)
    print(f"No injection model: {none_inj_path}")
    
    # Find early injection model
    early_inj_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'extra_specs': {
            'depth': 3,
            'channels': 64,
            'batch_norm': False,
            'action_injection': 'early'
        }
    }
    early_inj_path = find_checkpoint_by_hyperparams(early_inj_params)
    print(f"Early injection model: {early_inj_path}")
    
    # Find FC injection model
    fc_inj_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'extra_specs': {
            'depth': 3,
            'channels': 64,
            'batch_norm': False,
            'action_injection': 'fc'
        }
    }
    fc_inj_path = find_checkpoint_by_hyperparams(fc_inj_params)
    print(f"FC injection model: {fc_inj_path}")
    
    # Find learning rate models
    lr_1e5_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'lr': 1e-5,
        'extra_specs': {
            'depth': 3,
            'channels': 64,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    lr_1e5_path = find_checkpoint_by_hyperparams(lr_1e5_params)
    print(f"Learning rate 1e-5 model: {lr_1e5_path}")
    
    lr_5e5_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'lr': 5e-5,
        'extra_specs': {
            'depth': 3,
            'channels': 64,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    lr_5e5_path = find_checkpoint_by_hyperparams(lr_5e5_params)
    print(f"Learning rate 5e-5 model: {lr_5e5_path}")
    
    lr_5e4_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'lr': 5e-4,
        'extra_specs': {
            'depth': 3,
            'channels': 64,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    lr_5e4_path = find_checkpoint_by_hyperparams(lr_5e4_params)
    print(f"Learning rate 5e-4 model: {lr_5e4_path}")
    
    lr_1e3_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'lr': 1e-3,
        'extra_specs': {
            'depth': 3,
            'channels': 64,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    lr_1e3_path = find_checkpoint_by_hyperparams(lr_1e3_params)
    print(f"Learning rate 1e-3 model: {lr_1e3_path}")
    
    # Find gamma models
    gamma_09_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'gamma': 0.9,
        'extra_specs': {
            'depth': 3,
            'channels': 64,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    gamma_09_path = find_checkpoint_by_hyperparams(gamma_09_params)
    print(f"Gamma 0.9 model: {gamma_09_path}")
    
    gamma_095_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'gamma': 0.95,
        'extra_specs': {
            'depth': 3,
            'channels': 64,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    gamma_095_path = find_checkpoint_by_hyperparams(gamma_095_params)
    print(f"Gamma 0.95 model: {gamma_095_path}")
    
    gamma_099_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'gamma': 0.99,
        'extra_specs': {
            'depth': 3,
            'channels': 64,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    gamma_099_path = find_checkpoint_by_hyperparams(gamma_099_params)
    print(f"Gamma 0.99 model: {gamma_099_path}")
    
    gamma_0999_params = {
        'model_type': 'cnn',
        'num_episodes': 1000,
        'gamma': 0.999,
        'extra_specs': {
            'depth': 3,
            'channels': 64,
            'batch_norm': False,
            'action_injection': 'late'
        }
    }
    gamma_0999_path = find_checkpoint_by_hyperparams(gamma_0999_params)
    print(f"Gamma 0.999 model: {gamma_0999_path}")

if __name__ == "__main__":
    main()