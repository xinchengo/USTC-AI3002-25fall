"""
Wrapper for Alpha-Beta pruning baseline.
This implements the exact same algorithm as https://github.com/lihongxun945/gobang
"""

import numpy as np
from typing import Tuple
from .base import BaseWrapper

import sys
import os

# Add parent directory to path to import baselines
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines.alpha_beta import Board, minmax


class AlphaBetaWrapper(BaseWrapper):
    """
    A wrapper for alpha-beta pruning baseline.
    This implements the exact same algorithm as the React app https://github.com/lihongxun945/gobang
    
    Difficulty levels correspond to search depth:
    - weak (弱智): depth=2
    - easy (简单): depth=4  
    - medium (普通): depth=6
    - hard (困难): depth=8
    """
    
    # Difficulty to depth mapping
    DIFFICULTY_DEPTH = {
        'weak': 2,      # 弱智
        'easy': 4,      # 简单
        'medium': 6,    # 普通
        'hard': 8,      # 困难
    }
    
    def __init__(self, board_size: int = 15, bound: int = 5, difficulty: str = 'medium'):
        """
        Initialize the alpha-beta wrapper.
        
        Args:
            board_size: Size of the board (default 15 to match React app)
            bound: Number of pieces in a row to win (must be 5)
            difficulty: Difficulty level ('weak', 'easy', 'medium', 'hard')
        """
        assert bound == 5, "AlphaBetaWrapper only supports bound=5 (five in a row)"
        
        self.board_size = board_size
        self.bound = bound
        
        if difficulty not in self.DIFFICULTY_DEPTH:
            raise ValueError(f"Unknown difficulty: {difficulty}. "
                           f"Supported: {list(self.DIFFICULTY_DEPTH.keys())}")
        
        self.difficulty = difficulty
        self.depth = self.DIFFICULTY_DEPTH[difficulty]
        
        # Internal board for alpha-beta search
        # Note: The React app uses 1 for black (first player) and -1 for white
        self._board = None
    
    def _sync_board(self, board_state: np.ndarray, player_role: int):
        """
        Synchronize the internal board with the given board state.
        
        Args:
            board_state: Current board state where:
                - 0 = empty
                - 1 = player 1 (black in original game)
                - 2 = player 2 (white in original game)
            player_role: Which player the AI is (1 or 2)
        """
        # Create new board
        self._board = Board(size=self.board_size, first_role=1)
        
        # Count pieces to determine turn order
        count_1 = np.sum(board_state == 1)
        count_2 = np.sum(board_state == 2)
        
        # Reconstruct history by scanning the board
        # This is a simplification - we place pieces in order of count
        # For proper reconstruction, we'd need the actual move history
        pieces_1 = list(zip(*np.where(board_state == 1)))
        pieces_2 = list(zip(*np.where(board_state == 2)))
        
        # Place pieces alternating, starting with whoever has more or equal pieces
        i, j = 0, 0
        while i < len(pieces_1) or j < len(pieces_2):
            if i < len(pieces_1) and (j >= len(pieces_2) or i <= j):
                r, c = pieces_1[i]
                self._board.put(r, c, 1)  # Player 1 = black = 1
                i += 1
            if j < len(pieces_2) and i > j:
                r, c = pieces_2[j]
                self._board.put(r, c, -1)  # Player 2 = white = -1
                j += 1
    
    def _convert_role(self, player_role: int) -> int:
        """
        Convert player role from board representation to internal representation.
        
        Args:
            player_role: 1 or 2 (from board state)
        
        Returns:
            1 (black) or -1 (white) for internal use
        """
        return 1 if player_role == 1 else -1
    
    def _determine_ai_role(self, board_state: np.ndarray) -> int:
        """
        Determine which role the AI should play based on piece counts.
        
        Returns:
            1 or -1 (internal role representation)
        """
        count_1 = np.sum(board_state == 1)
        count_2 = np.sum(board_state == 2)
        
        # If equal pieces, AI plays black (1), otherwise plays the one with fewer pieces
        if count_1 <= count_2:
            return 1  # Play as black (player 1)
        else:
            return -1  # Play as white (player 2)
    
    def get_action(self, board_state: np.ndarray, **kwargs) -> Tuple[int, int]:
        """
        Get an action using alpha-beta pruning.
        
        Args:
            board_state: Current board state as numpy array of shape (board_size, board_size)
                - 0 = empty
                - 1 = player 1 (black)
                - 2 = player 2 (white)
            
        Returns:
            Tuple of (row, col) representing the selected action
        """
        # Check for empty board - play center
        if np.all(board_state == 0):
            center = self.board_size // 2
            return (center, center)
        
        # Determine which role to play
        ai_role = self._determine_ai_role(board_state)
        
        # Sync internal board
        self._sync_board(board_state, ai_role)
        
        # Run minimax search
        value, move, path = minmax(self._board, ai_role, self.depth, enable_vct=True)
        
        if move is None:
            # No valid move found, pick a random empty position
            empty_positions = np.where(board_state == 0)
            empty_coords = list(zip(empty_positions[0], empty_positions[1]))
            if empty_coords:
                import random
                return random.choice(empty_coords)
            return (-1, -1)
        
        return (int(move[0]), int(move[1]))
    
    def get_policy(self, board_state: np.ndarray) -> np.ndarray:
        """
        Get a policy distribution based on alpha-beta evaluation.
        
        For alpha-beta, this returns a one-hot vector for the best move.
        
        Args:
            board_state: Current board state as numpy array
            
        Returns:
            Policy distribution as numpy array of shape (board_size * board_size,)
        """
        # Get best action
        row, col = self.get_action(board_state)
        
        # Create one-hot policy
        policy = np.zeros(self.board_size * self.board_size)
        if row >= 0 and col >= 0:
            idx = row * self.board_size + col
            policy[idx] = 1.0
        else:
            # If no valid move, uniform over empty positions
            flat_board = board_state.flatten()
            legal_moves = (flat_board == 0).astype(float)
            if np.sum(legal_moves) > 0:
                policy = legal_moves / np.sum(legal_moves)
        
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
