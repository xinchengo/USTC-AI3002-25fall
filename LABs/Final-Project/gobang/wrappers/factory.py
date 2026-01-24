"""
Factory module for creating wrapper instances based on type and parameters.
This allows test scripts to be truly wrapper-agnostic.
"""

from .checkpoint import CheckpointWrapper
from .random import RandomWrapper
from .baseline import BaselineWrapper


def create_wrapper(wrapper_type, **kwargs):
    """
    Factory function to create wrapper instances based on type.

    Args:
        wrapper_type (str): Type of wrapper to create ('checkpoint', 'random', 'baseline')
        **kwargs: Arguments to pass to the wrapper constructor

    Returns:
        An instance of the requested wrapper type
    """
    import os
    import json

    if wrapper_type == 'checkpoint':
        model_path = kwargs.get('model_path')
        board_size = kwargs.get('board_size', 12)
        bound = kwargs.get('bound', 5)

        if model_path and os.path.exists(model_path):
            # Try to find the hyperparameters file associated with this model
            # Look for hyperparameters.txt in the same directory as the model
            import os.path

            # Get the directory of the model file
            model_dir = os.path.dirname(model_path)

            # Look for hyperparameters.txt in the model directory
            hyperparams_path = os.path.join(model_dir, 'hyperparameters.txt')

            model_type = "default"
            extra_specs = {}

            if os.path.exists(hyperparams_path):
                # Read hyperparameters from file
                with open(hyperparams_path, 'r') as f:
                    lines = f.readlines()
                    hyperparams = {}
                    for line in lines:
                        if ': ' in line:
                            key, value = line.strip().split(': ', 1)
                            hyperparams[key] = value

                # Extract model type and extra specs
                model_type = hyperparams.get('model_type', 'default')

                # Parse extra_specs if present
                if 'extra_specs' in hyperparams:
                    try:
                        extra_specs = eval(hyperparams['extra_specs'])
                    except:
                        extra_specs = {}

                print(f"Loading checkpoint with model_type: {model_type}, extra_specs: {extra_specs}")

            return CheckpointWrapper(
                model_path=model_path,
                board_size=board_size,
                bound=bound,
                model_type=model_type,
                extra_specs=extra_specs
            )
        else:
            # Raise error if model path is invalid
            raise FileNotFoundError(f"Model path '{model_path}' does not exist for checkpoint wrapper.")

    elif wrapper_type == 'random':
        board_size = kwargs.get('board_size', 12)
        bound = kwargs.get('bound', 5)
        return RandomWrapper(board_size=board_size, bound=bound)

    elif wrapper_type == 'baseline':
        board_size = kwargs.get('board_size', 12)
        bound = kwargs.get('bound', 5)
        return BaselineWrapper(board_size=board_size, bound=bound)

    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}. Supported types: checkpoint, random, baseline")


__all__ = ['create_wrapper', 'BaseWrapper', 'CheckpointWrapper', 'RandomWrapper', 'BaselineWrapper']