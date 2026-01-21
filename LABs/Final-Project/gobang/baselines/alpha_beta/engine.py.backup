"""
Alpha-Beta Minimax Search Engine for Gobang
Implements the minimax algorithm with alpha-beta pruning for efficient search
"""

import numpy as np
from typing import Tuple, List, Optional
from .evaluator import BoardEvaluator
import time


class AlphaBetaEngine:
    """
    Alpha-Beta pruning search engine for Gobang.
    Implements minimax with alpha-beta pruning and configurable depth.
    """
    
    def __init__(self, board_size: int = 12, bound: int = 5, depth: int = 4):
        """
        Initialize the alpha-beta engine.
        
        Args:
            board_size: Size of the board (default 12)
            bound: Number of pieces in a row to win (default 5)
            depth: Search depth (default 4, can be 2-10)
        """
        self.board_size = board_size
        self.bound = bound
        self.depth = depth
        self.evaluator = BoardEvaluator(board_size, bound)
        
        # Statistics
        self.nodes_searched = 0
        self.nodes_pruned = 0
    
    def get_best_move(self, board: np.ndarray, player: int) -> Tuple[int, int]:
        """
        Get the best move for the given player using alpha-beta search.
        
        Args:
            board: Current board state
            player: Player to move (1 or 2)
            
        Returns:
            Tuple of (row, col) representing the best move
        """
        self.nodes_searched = 0
        self.nodes_pruned = 0
        
        # Get legal moves
        legal_moves = self._get_legal_moves(board)
        
        if not legal_moves:
            return -1, -1
        
        # Quick win/block check
        quick_move = self._check_immediate_win_or_block(board, player)
        if quick_move is not None:
            return quick_move
        
        # Use alpha-beta search to find the best move
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Sort moves by heuristic value (most promising first)
        legal_moves = self._sort_moves(board, legal_moves, player)
        
        for move in legal_moves:
            row, col = move
            # Try this move
            board[row, col] = player
            
            # Evaluate with alpha-beta
            value = self._minimax(board, self.depth - 1, alpha, beta, False, player)
            
            # Undo the move
            board[row, col] = 0
            
            # Update best move
            if value > best_value:
                best_value = value
                best_move = move
            
            # Update alpha
            alpha = max(alpha, best_value)
        
        return best_move if best_move is not None else legal_moves[0]
    
    def _minimax(self, board: np.ndarray, depth: int, alpha: float, beta: float, 
                 is_maximizing: bool, player: int) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            board: Current board state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            is_maximizing: Whether this is a maximizing node
            player: The player we're optimizing for
            
        Returns:
            Evaluated score for this position
        """
        self.nodes_searched += 1
        
        # Terminal conditions
        if depth == 0:
            return self.evaluator.evaluate(board, player)
        
        # Check for game over
        if self._is_game_over(board):
            return self.evaluator.evaluate(board, player)
        
        legal_moves = self._get_legal_moves(board)
        if not legal_moves:
            return self.evaluator.evaluate(board, player)
        
        # Sort moves for better pruning
        current_player = player if is_maximizing else (3 - player)
        legal_moves = self._sort_moves(board, legal_moves, current_player)
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in legal_moves:
                row, col = move
                board[row, col] = player
                
                eval_score = self._minimax(board, depth - 1, alpha, beta, False, player)
                
                board[row, col] = 0
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    self.nodes_pruned += 1
                    break  # Beta cutoff
            
            return max_eval
        else:
            min_eval = float('inf')
            opponent = self.evaluator.get_opponent(player)
            for move in legal_moves:
                row, col = move
                board[row, col] = opponent
                
                eval_score = self._minimax(board, depth - 1, alpha, beta, True, player)
                
                board[row, col] = 0
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    self.nodes_pruned += 1
                    break  # Alpha cutoff
            
            return min_eval
    
    def _get_legal_moves(self, board: np.ndarray) -> List[Tuple[int, int]]:
        """
        Get all legal moves (empty positions).
        
        Args:
            board: Current board state
            
        Returns:
            List of (row, col) tuples for legal moves
        """
        empty_positions = np.where(board == 0)
        return list(zip(empty_positions[0], empty_positions[1]))
    
    def _get_neighbor_moves(self, board: np.ndarray, radius: int = 2) -> List[Tuple[int, int]]:
        """
        Get moves that are near existing pieces (for better move ordering).
        
        Args:
            board: Current board state
            radius: Radius around existing pieces to consider
            
        Returns:
            List of (row, col) tuples for moves near existing pieces
        """
        occupied_positions = np.where(board != 0)
        occupied_set = set(zip(occupied_positions[0], occupied_positions[1]))
        
        if not occupied_set:
            # If board is empty, return center position
            center = self.board_size // 2
            return [(center, center)]
        
        neighbor_moves = set()
        
        for row, col in occupied_set:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    new_row, new_col = row + dr, col + dc
                    if (0 <= new_row < self.board_size and 
                        0 <= new_col < self.board_size and 
                        board[new_row, new_col] == 0):
                        neighbor_moves.add((new_row, new_col))
        
        return list(neighbor_moves)
    
    def _sort_moves(self, board: np.ndarray, moves: List[Tuple[int, int]], 
                    player: int) -> List[Tuple[int, int]]:
        """
        Sort moves by heuristic value for better alpha-beta pruning.
        
        Args:
            board: Current board state
            moves: List of moves to sort
            player: Player to evaluate for
            
        Returns:
            Sorted list of moves (best first)
        """
        # If too many moves, only consider neighbors
        if len(moves) > 50:
            moves = self._get_neighbor_moves(board)
        
        move_scores = []
        for move in moves:
            row, col = move
            # Quick evaluation: place the piece and evaluate
            board[row, col] = player
            score = self.evaluator.evaluate(board, player)
            board[row, col] = 0
            move_scores.append((score, move))
        
        # Sort by score (descending)
        move_scores.sort(reverse=True, key=lambda x: x[0])
        
        return [move for score, move in move_scores]
    
    def _check_immediate_win_or_block(self, board: np.ndarray, 
                                      player: int) -> Optional[Tuple[int, int]]:
        """
        Check for immediate winning moves or blocking opponent's winning moves.
        
        Args:
            board: Current board state
            player: Current player
            
        Returns:
            (row, col) if there's an immediate win/block, None otherwise
        """
        opponent = self.evaluator.get_opponent(player)
        
        # First check for immediate wins
        for row, col in self._get_neighbor_moves(board, radius=2):
            board[row, col] = player
            if self.evaluator.check_win(board, row, col, player):
                board[row, col] = 0
                return (row, col)
            board[row, col] = 0
        
        # Then check for immediate blocks
        for row, col in self._get_neighbor_moves(board, radius=2):
            board[row, col] = opponent
            if self.evaluator.check_win(board, row, col, opponent):
                board[row, col] = 0
                return (row, col)
            board[row, col] = 0
        
        return None
    
    def _is_game_over(self, board: np.ndarray) -> bool:
        """
        Check if the game is over (someone won or board is full).
        
        Args:
            board: Current board state
            
        Returns:
            True if game is over, False otherwise
        """
        # Check if board is full
        if not np.any(board == 0):
            return True
        
        # Quick check for any winning patterns
        # This is a simplified check - could be optimized
        for player in [1, 2]:
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if board[row, col] == player:
                        if self.evaluator.check_win(board, row, col, player):
                            return True
        
        return False
