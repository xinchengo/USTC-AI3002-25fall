"""
Board evaluation function for Gobang AI
Evaluates board positions based on patterns and piece configurations
"""

import numpy as np
from typing import Tuple


class BoardEvaluator:
    """
    Evaluates board positions for Gobang using pattern-based heuristics
    """
    
    # Score values for different patterns
    FIVE = 100000        # Five in a row (win)
    FOUR = 10000         # Four in a row (strong attack)
    BLOCKED_FOUR = 1000  # Blocked four
    THREE = 1000         # Three in a row
    BLOCKED_THREE = 100  # Blocked three
    TWO = 100            # Two in a row
    BLOCKED_TWO = 10     # Blocked two
    ONE = 10             # Single piece
    
    def __init__(self, board_size: int = 12, bound: int = 5):
        """
        Initialize the evaluator.
        
        Args:
            board_size: Size of the board (default 12)
            bound: Number of pieces in a row to win (default 5)
        """
        self.board_size = board_size
        self.bound = bound
        
        # Directions: horizontal, vertical, diagonal /, diagonal \
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    @staticmethod
    def get_opponent(player: int) -> int:
        """
        Get the opponent player number.
        
        Args:
            player: Current player (1 or 2)
            
        Returns:
            Opponent player number (2 or 1)
        """
        return 3 - player
    
    def evaluate(self, board: np.ndarray, player: int) -> int:
        """
        Evaluate the board from the perspective of the given player.
        
        Args:
            board: Current board state
            player: Player to evaluate for (1 or 2)
            
        Returns:
            Score for the player (positive is good for player)
        """
        opponent = self.get_opponent(player)
        
        # Calculate scores for both players
        player_score = self._calculate_player_score(board, player)
        opponent_score = self._calculate_player_score(board, opponent)
        
        # Return the difference (player's advantage)
        return player_score - opponent_score
    
    def _calculate_player_score(self, board: np.ndarray, player: int) -> int:
        """
        Calculate the total score for a specific player.
        
        Args:
            board: Current board state
            player: Player to calculate score for
            
        Returns:
            Total score for the player
        """
        total_score = 0
        
        # Check all positions on the board
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board[row, col] == player:
                    # Check all directions from this piece
                    for direction in self.directions:
                        score = self._evaluate_line(board, row, col, direction, player)
                        total_score += score
        
        return total_score
    
    def _evaluate_line(self, board: np.ndarray, row: int, col: int, 
                       direction: Tuple[int, int], player: int) -> int:
        """
        Evaluate a line starting from a position in a given direction.
        
        Args:
            board: Current board state
            row: Starting row
            col: Starting column
            direction: Direction to check (dr, dc)
            player: Player to evaluate for
            
        Returns:
            Score for this line
        """
        dr, dc = direction
        opponent = self.get_opponent(player)
        
        # Count consecutive pieces in the forward direction
        count = 1
        blocked_start = False
        blocked_end = False
        
        # Count forward
        r, c = row + dr, col + dc
        while 0 <= r < self.board_size and 0 <= c < self.board_size:
            if board[r, c] == player:
                count += 1
            elif board[r, c] == opponent:
                blocked_end = True
                break
            else:  # Empty space
                break
            r, c = r + dr, c + dc
        
        # Check if blocked at the end
        if not (0 <= r < self.board_size and 0 <= c < self.board_size):
            blocked_end = True
        
        # Count backward (to get the full line)
        r, c = row - dr, col - dc
        while 0 <= r < self.board_size and 0 <= c < self.board_size:
            if board[r, c] == player:
                count += 1
            elif board[r, c] == opponent:
                blocked_start = True
                break
            else:  # Empty space
                break
            r, c = r - dr, c - dc
        
        # Check if blocked at the start
        if not (0 <= r < self.board_size and 0 <= c < self.board_size):
            blocked_start = True
        
        # Calculate score based on count and blocking status
        return self._get_score(count, blocked_start and blocked_end)
    
    def _get_score(self, count: int, blocked: bool) -> int:
        """
        Get the score for a given pattern.
        
        Args:
            count: Number of consecutive pieces
            blocked: Whether the pattern is blocked on both ends
            
        Returns:
            Score for this pattern
        """
        if count >= self.bound:
            return self.FIVE
        elif count == 4:
            return self.BLOCKED_FOUR if blocked else self.FOUR
        elif count == 3:
            return self.BLOCKED_THREE if blocked else self.THREE
        elif count == 2:
            return self.BLOCKED_TWO if blocked else self.TWO
        elif count == 1:
            return self.ONE
        else:
            return 0
    
    def check_win(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """
        Check if placing a piece at (row, col) creates a win for the player.
        
        Args:
            board: Current board state
            row: Row position
            col: Column position
            player: Player to check for
            
        Returns:
            True if this creates a win, False otherwise
        """
        if board[row, col] != player:
            return False
        
        # Check all four directions
        for dr, dc in self.directions:
            count = 1
            
            # Count in positive direction
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and board[r, c] == player:
                count += 1
                r, c = r + dr, c + dc
            
            # Count in negative direction
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and board[r, c] == player:
                count += 1
                r, c = r - dr, c - dc
            
            if count >= self.bound:
                return True
        
        return False
