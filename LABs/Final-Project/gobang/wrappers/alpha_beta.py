"""
Alpha-Beta Wrapper for Gobang
Wrapper that exposes the alpha-beta pruning engine through the standard wrapper interface
"""

import numpy as np
from typing import Tuple
import sys
import os

# Add the parent directory to path to import base wrapper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from base import BaseWrapper

# Import the alpha-beta engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from baselines.alpha_beta import AlphaBetaEngine


class AlphaBetaWrapper(BaseWrapper):
    """
    Wrapper for the alpha-beta pruning baseline.
    Implements configurable-depth minimax search with alpha-beta pruning.
    """
    
    def __init__(self, board_size: int = 12, bound: int = 5, depth: int = 4):
        """
        Initialize the alpha-beta wrapper.
        
        Args:
            board_size: Size of the board (default 12)
            bound: Number of pieces in a row to win (default 5)
            depth: Search depth (default 4, configurable 2-10 for "弱智" levels)
        """
        self.board_size = board_size
        self.bound = bound
        self.depth = max(2, min(10, depth))  # Clamp depth to 2-10 range
        self.engine = AlphaBetaEngine(board_size, bound, self.depth)
    
    def get_action(self, board_state: np.ndarray, **kwargs) -> Tuple[int, int]:
        """
        Get an action using alpha-beta search.
        
        Args:
            board_state: Current board state as numpy array of shape (board_size, board_size)
                         Values: 0=empty, 1=black, 2=white
            **kwargs: Additional arguments (e.g., player)
            
        Returns:
            Tuple of (row, col) representing the selected action
        """
        # Determine which player we are
        # Count pieces: the player with fewer pieces should move next
        black_count = np.sum(board_state == 1)
        white_count = np.sum(board_state == 2)
        
        # If equal, assume we're black (player 1)
        # If black has more, we're white (player 2)
        player = kwargs.get('player', 1 if black_count <= white_count else 2)
        
        # Get the best move from the engine
        row, col = self.engine.get_best_move(board_state.copy(), player)
        
        return int(row), int(col)
    
    def get_policy(self, board_state: np.ndarray) -> np.ndarray:
        """
        Get a policy distribution based on alpha-beta evaluation.
        
        Args:
            board_state: Current board state as numpy array of shape (board_size, board_size)
            
        Returns:
            Policy distribution as numpy array of shape (board_size * board_size,)
        """
        # Create a policy array
        policy = np.zeros(self.board_size * self.board_size)
        
        # Determine which player we are
        black_count = np.sum(board_state == 1)
        white_count = np.sum(board_state == 2)
        player = 1 if black_count <= white_count else 2
        
        # Get legal moves
        legal_moves = np.where(board_state.flatten() == 0)[0]
        
        if len(legal_moves) == 0:
            return policy
        
        # Evaluate each legal move
        move_scores = []
        for move_idx in legal_moves:
            row = move_idx // self.board_size
            col = move_idx % self.board_size
            
            # Simulate the move
            test_board = board_state.copy()
            test_board[row, col] = player
            
            # Evaluate the position
            score = self.engine.evaluator.evaluate(test_board, player)
            move_scores.append(score)
        
        # Convert scores to probabilities
        # Shift scores to be positive
        move_scores = np.array(move_scores)
        min_score = np.min(move_scores)
        if min_score < 0:
            move_scores = move_scores - min_score + 1
        
        # Softmax-like distribution (with temperature for smoother distribution)
        temperature = 1.0
        exp_scores = np.exp(move_scores / (temperature * (np.max(move_scores) + 1)))
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Assign probabilities to the policy
        for i, move_idx in enumerate(legal_moves):
            policy[move_idx] = probabilities[i]
        
        return policy
    
    def __call__(self, board_state: np.ndarray) -> np.ndarray:
        """
        Callable interface to get policy for a given board state.
        
        Args:
            board_state: Current board state
            
        Returns:
            Policy distribution
        """
        return self.get_policy(board_state)
