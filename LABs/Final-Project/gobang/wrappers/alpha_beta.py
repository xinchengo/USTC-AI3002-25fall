"""
Alpha-Beta Wrapper for Gobang
Wrapper that exposes the exact lihongxun945/gobang minimax implementation
through the standard wrapper interface
"""

import numpy as np
from typing import Tuple
from .base import BaseWrapper
from baselines.alpha_beta import minmax, BoardEvaluator


class AlphaBetaWrapper(BaseWrapper):
    """
    Wrapper for the alpha-beta pruning baseline.
    Uses exact implementation from lihongxun945/gobang with iterative deepening,
    VCT (Variable-depth Continuous Threat) and VCF support.
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
        self.evaluator = BoardEvaluator(size=board_size, bound=bound)
    
    def get_action(self, board_state: np.ndarray, **kwargs) -> Tuple[int, int]:
        """
        Get an action using the exact minmax algorithm from lihongxun945/gobang.
        
        Args:
            board_state: Current board state as numpy array of shape (board_size, board_size)
                         Values: 0=empty, 1=black, 2=white
            **kwargs: Additional arguments (e.g., player, enable_vct)
            
        Returns:
            Tuple of (row, col) representing the selected action
        """
        # Determine which player we are
        black_count = np.sum(board_state == 1)
        white_count = np.sum(board_state == 2)
        
        # If equal, assume we're black (player 1)
        # If black has more, we're white (player -1)
        player = kwargs.get('player', 1 if black_count <= white_count else -1)
        enable_vct = kwargs.get('enable_vct', True)
        
        # Sync board state to evaluator
        # Convert from our format (0, 1, 2) to evaluator format (0, 1, -1)
        self._sync_board_to_evaluator(board_state, player)
        
        # Run minimax algorithm
        value, move, path = minmax(self.evaluator, player, self.depth, enable_vct)
        
        if move is None or len(move) == 0:
            # Fallback: return a random legal move
            empty_positions = np.where(board_state == 0)
            if len(empty_positions[0]) > 0:
                idx = np.random.randint(len(empty_positions[0]))
                return int(empty_positions[0][idx]), int(empty_positions[1][idx])
            return -1, -1
        
        return int(move[0]), int(move[1])
    
    def _sync_board_to_evaluator(self, board_state: np.ndarray, player: int):
        """
        Sync the board state to the evaluator's internal representation.
        
        Args:
            board_state: External board state (0=empty, 1=black, 2=white)
            player: Current player to move
        """
        # Reset evaluator
        self.evaluator = BoardEvaluator(size=self.board_size, bound=self.bound)
        
        # Replay moves in order
        # First, collect all pieces
        black_pieces = []
        white_pieces = []
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board_state[i][j] == 1:
                    black_pieces.append((i, j))
                elif board_state[i][j] == 2:
                    white_pieces.append((i, j))
        
        # Replay moves alternating between players
        # Assume black (1) plays first
        total_moves = len(black_pieces) + len(white_pieces)
        
        # Create interleaved move sequence
        moves = []
        for idx in range(max(len(black_pieces), len(white_pieces))):
            if idx < len(black_pieces):
                moves.append((black_pieces[idx], 1))
            if idx < len(white_pieces):
                moves.append((white_pieces[idx], -1))
        
        # Apply moves
        for (x, y), role in moves:
            self.evaluator.move(x, y, role)
    
    def get_policy(self, board_state: np.ndarray) -> np.ndarray:
        """
        Get a policy distribution based on the evaluation scores.
        
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
        player = 1 if black_count <= white_count else -1
        
        # Sync board to evaluator
        self._sync_board_to_evaluator(board_state, player)
        
        # Get valuable moves from evaluator
        moves = self.evaluator.get_moves(player, depth=0, only_three=False, only_four=False)
        
        if not moves:
            # If no moves, uniform over empty positions
            legal_moves = np.where(board_state.flatten() == 0)[0]
            if len(legal_moves) > 0:
                policy[legal_moves] = 1.0 / len(legal_moves)
            return policy
        
        # Evaluate each move and create a score-based distribution
        move_scores = []
        for x, y in moves:
            # Simulate move
            self.evaluator.move(x, y, player)
            score = self.evaluator.evaluate(player)
            self.evaluator.undo(x, y)
            
            move_scores.append((x * self.board_size + y, score))
        
        # Convert scores to probabilities using softmax
        if move_scores:
            positions = [pos for pos, score in move_scores]
            scores = np.array([score for pos, score in move_scores], dtype=np.float64)
            
            # Shift to positive
            if np.min(scores) < 0:
                scores = scores - np.min(scores) + 1
            
            # Softmax with temperature
            temperature = 1.0
            max_score = np.max(scores)
            if max_score > 0:
                exp_scores = np.exp(scores / (temperature * max_score))
                probabilities = exp_scores / np.sum(exp_scores)
                
                for pos, prob in zip(positions, probabilities):
                    policy[pos] = prob
        
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
