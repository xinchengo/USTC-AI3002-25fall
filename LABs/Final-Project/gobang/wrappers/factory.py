"""
Factory module for creating wrapper instances based on type and parameters.
This allows test scripts to be truly wrapper-agnostic.
"""

from .checkpoint import CheckpointWrapper
from .random import RandomWrapper
from .baseline import BaselineWrapper
from .alpha_beta import AlphaBetaWrapper


def create_wrapper(wrapper_type, **kwargs):
    """
    Factory function to create wrapper instances based on type.

    Args:
        wrapper_type (str): Type of wrapper to create ('checkpoint', 'random', 'baseline', 'alpha_beta')
        **kwargs: Arguments to pass to the wrapper constructor

    Returns:
        An instance of the requested wrapper type
    """
    import os

    if wrapper_type == 'checkpoint':
        model_path = kwargs.get('model_path')
        board_size = kwargs.get('board_size', 12)
        bound = kwargs.get('bound', 5)

        if model_path and os.path.exists(model_path):
            return CheckpointWrapper(
                model_path=model_path,
                board_size=board_size,
                bound=bound
            )
        else:
            # If model path not provided or doesn't exist, fall back to baseline
            print(f"Model path not provided or doesn't exist for {wrapper_type}, falling back to baseline")
            return BaselineWrapper(board_size=board_size, bound=bound)

    elif wrapper_type == 'random':
        board_size = kwargs.get('board_size', 12)
        bound = kwargs.get('bound', 5)
        return RandomWrapper(board_size=board_size, bound=bound)

    elif wrapper_type == 'baseline':
        board_size = kwargs.get('board_size', 12)
        bound = kwargs.get('bound', 5)
        return BaselineWrapper(board_size=board_size, bound=bound)

    elif wrapper_type == 'alpha_beta':
        board_size = kwargs.get('board_size', 12)
        bound = kwargs.get('bound', 5)
        difficulty = kwargs.get('difficulty', 'medium')
        return AlphaBetaWrapper(board_size=board_size, bound=bound, difficulty=difficulty)

    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}. Supported types: checkpoint, random, baseline, alpha_beta")


__all__ = ['create_wrapper', 'BaseWrapper', 'CheckpointWrapper', 'RandomWrapper', 'BaselineWrapper', 'AlphaBetaWrapper']