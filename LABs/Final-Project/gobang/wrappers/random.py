import numpy as np
from typing import Tuple
import random


class RandomWrapper:
    """
    A wrapper for a random policy that selects moves uniformly at random
    from available legal moves.
    """
    
    def __init__(self, board_size: int = 12, bound: int = 5):
        """
        Initialize the random policy wrapper.
        
        Args:
            board_size: Size of the board (default 12)
            bound: Number of pieces in a row to win (default 5)
        """
        self.board_size = board_size
        self.bound = bound
    
    def get_action(self, board_state: np.ndarray, **kwargs) -> Tuple[int, int]:
        """
        Get a random legal action given the current board state.
        
        Args:
            board_state: Current board state as numpy array of shape (board_size, board_size)
            
        Returns:
            Tuple of (row, col) representing a random legal action
        """
        # Find all legal moves (empty positions)
        empty_positions = np.where(board_state == 0)
        empty_coords = list(zip(empty_positions[0], empty_positions[1]))
        
        if not empty_coords:
            # No legal moves available
            return -1, -1
        
        # Select a random legal move
        row, col = random.choice(empty_coords)
        return int(row), int(col)
    
    def get_policy(self, board_state: np.ndarray) -> np.ndarray:
        """
        Get a uniform random policy distribution for the given board state.
        
        Args:
            board_state: Current board state as numpy array of shape (board_size, board_size)
            
        Returns:
            Uniform policy distribution as numpy array of shape (board_size * board_size,)
        """
        # Create a policy where all legal moves have equal probability
        flat_board = board_state.flatten()
        legal_moves = (flat_board == 0).astype(float)
        
        if np.sum(legal_moves) > 0:
            # Normalize so that legal moves have equal probability
            legal_moves = legal_moves / np.sum(legal_moves)
        
        return legal_moves
    
    def __call__(self, board_state: np.ndarray) -> np.ndarray:
        """
        Callable interface to get policy for a given board state.
        
        Args:
            board_state: Current board state
            
        Returns:
            Uniform random policy distribution
        """
        return self.get_policy(board_state)