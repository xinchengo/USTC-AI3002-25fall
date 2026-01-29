import torch
import pickle
import numpy as np
from typing import Tuple, Optional
from utils import device
from .base import BaseWrapper


# Import the required classes to handle pickle loading
from submission import GobangModel, Actor, Critic


class CheckpointWrapper(BaseWrapper):
    """
    A wrapper for loading and using checkpointed models.
    This wrapper allows loading both .pth (state dict) and .pkl (complete model) files.
    """

    def __init__(self, model_path: str, board_size: int = 12, bound: int = 5, model_type: str = "default", extra_specs: dict = None, use_deep: bool = False):
        """
        Initialize the checkpoint wrapper.

        Args:
            model_path: Path to the model file (.pth or .pkl)
            board_size: Size of the board (default 12)
            bound: Number of pieces in a row to win (default 5)
            model_type: Model architecture type (default "default")
            extra_specs: Extra specifications for model architecture (dict)
            use_deep: Whether to use deep architecture (for backward compatibility)
        """
        self.board_size = board_size
        self.bound = bound
        self.model_type = model_type
        self.extra_specs = extra_specs or {}

        # Determine if this is a deep model by checking hyperparameters or filename
        is_deep = use_deep or self.extra_specs.get('use_deep', False) or 'deep' in model_path.lower()

        # For backward compatibility, if model_type is default and use_deep is True, update extra_specs
        if model_type == "default" and is_deep:
            self.extra_specs['use_deep'] = True

        if model_path.endswith('.pkl'):
            # Load complete model object
            # Need to handle the case where the pickle was saved from __main__ context
            import sys
            # Map the __main__ module to submission temporarily
            original_main = sys.modules.get('__main__')
            submission_module = sys.modules.get('submission')

            # If we have access to the submission module, use it as __main__ for loading
            if submission_module:
                sys.modules['__main__'] = submission_module

            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            finally:
                # Restore the original __main__ module
                if original_main:
                    sys.modules['__main__'] = original_main
                elif '__main__' in sys.modules:
                    del sys.modules['__main__']

            # Ensure the loaded model has the correct properties
            self.model.to(device)
            self.model.eval()
        elif model_path.endswith('.pth'):
            # Load state dict and reconstruct model
            from submission import GobangModel
            
            # Load state dict first to inspect its keys
            state_dict = torch.load(model_path, map_location=device)
            
            # Detect architecture from state_dict keys
            has_backbone = any('backbone.' in key for key in state_dict.keys())
            has_transformer = any('transformer_encoder' in key for key in state_dict.keys())
            has_attention = any('attention' in key for key in state_dict.keys())
            use_deep = any('conv_blocks.4' in key for key in state_dict.keys()) or any('conv_blocks.6' in key for key in state_dict.keys())
            
            # Determine model type and extra specs
            actual_extra_specs = dict(self.extra_specs)  # Copy to avoid modifying original
            
            if has_transformer:
                # This is a transformer model
                if self.model_type == "default":
                    self.model_type = "transformer"
                actual_extra_specs['use_backbone'] = has_backbone
                
                # Detect number of transformer layers
                max_layer_idx = -1
                for key in state_dict.keys():
                    if 'transformer_encoder.layers.' in key:
                        # Extract layer index
                        parts = key.split('transformer_encoder.layers.')
                        if len(parts) > 1:
                            layer_idx_str = parts[1].split('.')[0]
                            if layer_idx_str.isdigit():
                                max_layer_idx = max(max_layer_idx, int(layer_idx_str))
                
                if max_layer_idx >= 0:
                    num_layers = max_layer_idx + 1
                    actual_extra_specs['num_layers'] = num_layers
                    print(f"Detected {num_layers} transformer layers")
                
            elif has_attention:
                # This is an attention model
                if self.model_type == "default":
                    self.model_type = "attention"
                actual_extra_specs['use_backbone'] = has_backbone
            elif use_deep:
                # This is a deep CNN model
                actual_extra_specs['use_deep'] = True
            
            # Detect action injection strategy for critic
            has_late_proj = 'critic.late_proj.weight' in state_dict
            if has_late_proj:
                actual_extra_specs['action_injection'] = 'late'
            
            # Detect CNN depth for CNN/default models
            if self.model_type in ["default", "cnn"]:
                max_conv_idx = -1
                for key in state_dict.keys():
                    if 'conv_blocks.' in key:
                        parts = key.split('conv_blocks.')
                        if len(parts) > 1:
                            idx_str = parts[1].split('.')[0]
                            if idx_str.isdigit():
                                max_conv_idx = max(max_conv_idx, int(idx_str))
                
                if max_conv_idx >= 0:
                    # Each depth level has 3 layers (Conv, BN optional, ReLU)
                    # Approximate depth based on highest index
                    estimated_depth = (max_conv_idx + 1) // 3
                    if estimated_depth > 0:
                        actual_extra_specs['depth'] = estimated_depth
            
            print(f"Auto-detected model architecture: type={self.model_type}, use_backbone={actual_extra_specs.get('use_backbone', False)}, extra_specs={actual_extra_specs}")
            
            self.model = GobangModel(board_size=board_size, bound=bound, model_type=self.model_type, extra_specs=actual_extra_specs)
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()
        else:
            raise ValueError(f"Unsupported model file format: {model_path}. Expected .pth or .pkl")
    
    def get_action(self, board_state: np.ndarray, temperature: float = 1.0) -> Tuple[int, int]:
        """
        Get an action from the model given the current board state.

        Args:
            board_state: Current board state as numpy array of shape (board_size, board_size)
            temperature: Temperature for action selection (higher = more random)

        Returns:
            Tuple of (row, col) representing the selected action
        """
        with torch.no_grad():
            # Get policy from the model
            policy, _ = self.model.actor(board_state), None  # We only need the policy

            if isinstance(policy, torch.Tensor):
                policy = policy.cpu().numpy()

            # Handle potential batch dimension
            if policy.ndim > 1:
                policy = policy[0]  # Take first item if batched

            # Apply temperature scaling
            if temperature != 1.0:
                policy = np.power(policy, 1.0 / temperature)
                policy = policy / np.sum(policy)

            # Mask illegal moves (occupied positions)
            # Flatten the board to match policy dimensions
            flat_board = board_state.flatten()
            mask = (flat_board != 0)  # Positions that are occupied

            # Ensure policy and mask have the same length
            if len(policy) != len(mask):
                raise ValueError(f"Policy length {len(policy)} does not match mask length {len(mask)}")

            masked_policy = np.copy(policy)
            masked_policy[mask] = 0  # Set occupied positions to 0 probability

            if np.sum(masked_policy) > 0:
                masked_policy = masked_policy / np.sum(masked_policy)
            else:
                # If all legal moves have zero probability, pick randomly among legal moves
                legal_moves = (flat_board == 0)  # Positions that are empty
                masked_policy = legal_moves.astype(float)
                masked_policy = masked_policy / np.sum(masked_policy)

            # Sample action based on policy
            action_idx = np.random.choice(len(masked_policy), p=masked_policy)
            row, col = action_idx // self.board_size, action_idx % self.board_size

            return int(row), int(col)
    
    def get_policy(self, board_state: np.ndarray) -> np.ndarray:
        """
        Get the policy distribution for the given board state.
        
        Args:
            board_state: Current board state as numpy array of shape (board_size, board_size)
            
        Returns:
            Policy distribution as numpy array of shape (board_size * board_size,)
        """
        with torch.no_grad():
            policy, _ = self.model.actor(board_state), None
            
            if isinstance(policy, torch.Tensor):
                policy = policy.cpu().numpy()
            
            # Mask illegal moves
            mask = (board_state != 0)
            masked_policy = np.copy(policy)
            masked_policy[mask.flatten()] = 0
            
            if np.sum(masked_policy) > 0:
                masked_policy = masked_policy / np.sum(masked_policy)
            
            return masked_policy
    
    def __call__(self, board_state: np.ndarray) -> np.ndarray:
        """
        Callable interface to get policy for a given board state.
        
        Args:
            board_state: Current board state
            
        Returns:
            Policy distribution
        """
        return self.get_policy(board_state)