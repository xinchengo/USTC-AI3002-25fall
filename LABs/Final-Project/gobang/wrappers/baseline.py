import numpy as np
from typing import Tuple
from .base import BaseWrapper
import random


class BaselineWrapper(BaseWrapper):
    """
    A wrapper for a simple baseline policy (e.g., alpha-beta pruning or heuristic-based).
    This is a simplified implementation for demonstration purposes.
    """
    
    def __init__(self, board_size: int = 12, bound: int = 5):
        """
        Initialize the baseline wrapper.
        
        Args:
            board_size: Size of the board (default 12)
            bound: Number of pieces in a row to win (default 5)
        """
        self.board_size = board_size
        self.bound = bound
    
    def get_action(self, board_state: np.ndarray, **kwargs) -> Tuple[int, int]:
        """
        Get an action using a simple heuristic approach.
        This is a basic implementation that looks for immediate wins/blocks.
        
        Args:
            board_state: Current board state as numpy array of shape (board_size, board_size)
            
        Returns:
            Tuple of (row, col) representing the selected action
        """
        # Find all legal moves (empty positions)
        empty_positions = np.where(board_state == 0)
        empty_coords = list(zip(empty_positions[0], empty_positions[1]))
        
        if not empty_coords:
            # No legal moves available
            return -1, -1
        
        # Simple heuristic: look for immediate wins or blocks
        for player in [1, 2]:  # Check for both players (self and opponent)
            for r, c in empty_coords:
                # Simulate placing a piece
                test_board = board_state.copy()
                test_board[r, c] = player
                
                # Check if this creates a win
                if self._check_win(test_board, r, c, player):
                    return int(r), int(c)
        
        # If no immediate win/block, select a reasonable move
        # For now, just pick a random move near existing pieces
        if len(empty_coords) > 0:
            # Prioritize moves near existing pieces
            prioritized_moves = []
            for r, c in empty_coords:
                # Check if adjacent to any existing piece
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                            if board_state[nr, nc] != 0:
                                prioritized_moves.append((r, c))
                                break
            
            if prioritized_moves:
                return random.choice(prioritized_moves)
            else:
                return random.choice(empty_coords)
        
        return random.choice(empty_coords)
    
    def _check_win(self, board: np.ndarray, r: int, c: int, player: int) -> bool:
        """
        Check if placing a piece at (r, c) creates a win for the player.
        """
        # Check horizontal
        count = 1
        # Left
        for i in range(1, self.bound):
            if c-i >= 0 and board[r, c-i] == player:
                count += 1
            else:
                break
        # Right
        for i in range(1, self.bound):
            if c+i < self.board_size and board[r, c+i] == player:
                count += 1
            else:
                break
        if count >= self.bound:
            return True
        
        # Check vertical
        count = 1
        # Up
        for i in range(1, self.bound):
            if r-i >= 0 and board[r-i, c] == player:
                count += 1
            else:
                break
        # Down
        for i in range(1, self.bound):
            if r+i < self.board_size and board[r+i, c] == player:
                count += 1
            else:
                break
        if count >= self.bound:
            return True
        
        # Check diagonal (top-left to bottom-right)
        count = 1
        # Top-left
        for i in range(1, self.bound):
            if r-i >= 0 and c-i >= 0 and board[r-i, c-i] == player:
                count += 1
            else:
                break
        # Bottom-right
        for i in range(1, self.bound):
            if r+i < self.board_size and c+i < self.board_size and board[r+i, c+i] == player:
                count += 1
            else:
                break
        if count >= self.bound:
            return True
        
        # Check diagonal (top-right to bottom-left)
        count = 1
        # Top-right
        for i in range(1, self.bound):
            if r-i >= 0 and c+i < self.board_size and board[r-i, c+i] == player:
                count += 1
            else:
                break
        # Bottom-left
        for i in range(1, self.bound):
            if r+i < self.board_size and c-i >= 0 and board[r+i, c-i] == player:
                count += 1
            else:
                break
        if count >= self.bound:
            return True
        
        return False
    
    def get_policy(self, board_state: np.ndarray) -> np.ndarray:
        """
        Get a policy distribution based on simple heuristics.
        
        Args:
            board_state: Current board state as numpy array of shape (board_size, board_size)
            
        Returns:
            Policy distribution as numpy array of shape (board_size * board_size,)
        """
        # Create a policy where all legal moves have equal probability initially
        flat_board = board_state.flatten()
        legal_moves = (flat_board == 0).astype(float)
        
        # Enhance probabilities for strategic moves
        for idx in range(len(legal_moves)):
            if legal_moves[idx] > 0:
                r, c = idx // self.board_size, idx % self.board_size
                # Give slight preference to center positions
                center_dist = abs(r - self.board_size//2) + abs(c - self.board_size//2)
                # Lower distance to center means higher priority
                legal_moves[idx] *= (self.board_size - center_dist) / self.board_size
        
        if np.sum(legal_moves) > 0:
            legal_moves = legal_moves / np.sum(legal_moves)
        
        return legal_moves
    
    def __call__(self, board_state: np.ndarray) -> np.ndarray:
        """
        Callable interface to get policy for a given board state.
        
        Args:
            board_state: Current board state
            
        Returns:
            Policy distribution
        """
        return self.get_policy(board_state)