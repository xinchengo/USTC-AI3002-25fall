"""
Board class for Gobang.
Ported from https://github.com/lihongxun945/gobang
"""

from .zobrist import Zobrist
from .cache import Cache
from .evaluate import Evaluate, FIVE
from .config import config


class Board:
    """Gobang board with evaluation support."""
    
    def __init__(self, size=15, first_role=1):
        self.size = size
        self.board = [[0] * size for _ in range(size)]
        self.first_role = first_role  # 1 for black, -1 for white
        self.role = first_role
        self.history = []
        self.zobrist = Zobrist(size)
        self.winner_cache = Cache()
        self.gameover_cache = Cache()
        self.evaluate_cache = Cache()
        self.valuable_moves_cache = Cache()
        self.evaluate_time = 0
        self.evaluator = Evaluate(size)
    
    def is_game_over(self):
        """Check if the game is over."""
        hash_val = self.hash()
        cached = self.gameover_cache.get(hash_val)
        if cached is not None:
            return cached
        
        if self.get_winner() != 0:
            self.gameover_cache.put(hash_val, True)
            return True
        
        # Check if board is full
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    self.gameover_cache.put(hash_val, False)
                    return False
        
        self.gameover_cache.put(hash_val, True)
        return True
    
    def get_winner(self):
        """Get the winner of the game."""
        hash_val = self.hash()
        cached = self.winner_cache.get(hash_val)
        if cached is not None:
            return cached
        
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    continue
                for dx, dy in directions:
                    count = 0
                    while (0 <= i + dx * count < self.size and
                           0 <= j + dy * count < self.size and
                           self.board[i + dx * count][j + dy * count] == self.board[i][j]):
                        count += 1
                    if count >= 5:
                        self.winner_cache.put(hash_val, self.board[i][j])
                        return self.board[i][j]
        
        self.winner_cache.put(hash_val, 0)
        return 0
    
    def get_valid_moves(self):
        """Get all valid moves."""
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves
    
    def put(self, i, j, role=None):
        """Place a piece on the board."""
        if role is None:
            role = self.role
        
        if self.board[i][j] != 0:
            return False
        
        self.board[i][j] = role
        self.history.append({'i': i, 'j': j, 'role': role})
        self.zobrist.toggle_piece(i, j, role)
        self.evaluator.move(i, j, role)
        self.role *= -1
        return True
    
    def undo(self):
        """Undo the last move."""
        if not self.history:
            return False
        
        last_move = self.history.pop()
        self.board[last_move['i']][last_move['j']] = 0
        self.role = last_move['role']
        self.zobrist.toggle_piece(last_move['i'], last_move['j'], last_move['role'])
        self.evaluator.undo(last_move['i'], last_move['j'])
        return True
    
    def get_valuable_moves(self, role, depth=0, only_three=False, only_four=False):
        """Get valuable moves based on evaluation."""
        hash_val = self.hash()
        prev = self.valuable_moves_cache.get(hash_val)
        if prev is not None:
            if (prev['role'] == role and prev['depth'] == depth and
                prev['only_three'] == only_three and prev['only_four'] == only_four):
                return prev['moves']
        
        moves = self.evaluator.get_moves(role, depth, only_three, only_four)
        
        # Add center point if not occupied
        if not only_three and not only_four:
            center = self.size // 2
            if self.board[center][center] == 0:
                moves.append((center, center))
        
        self.valuable_moves_cache.put(hash_val, {
            'role': role,
            'moves': moves,
            'depth': depth,
            'only_three': only_three,
            'only_four': only_four
        })
        return moves
    
    def hash(self):
        """Get Zobrist hash."""
        return self.zobrist.get_hash()
    
    def evaluate(self, role):
        """Evaluate the board for a role."""
        hash_val = self.hash()
        prev = self.evaluate_cache.get(hash_val)
        if prev is not None:
            if prev['role'] == role:
                return prev['score']
        
        winner = self.get_winner()
        if winner != 0:
            score = FIVE * winner * role
        else:
            score = self.evaluator.evaluate(role)
        
        self.evaluate_cache.put(hash_val, {'role': role, 'score': score})
        return score
    
    def reverse(self):
        """Create a reversed board (swap colors)."""
        new_board = Board(self.size, -self.first_role)
        for move in self.history:
            new_board.put(move['i'], move['j'], -move['role'])
        return new_board
    
    def display(self, extra_points=None):
        """Display the board."""
        if extra_points is None:
            extra_points = []
        extra_positions = {i * self.size + j for i, j in extra_points}
        
        result = ''
        for i in range(self.size):
            for j in range(self.size):
                position = i * self.size + j
                if position in extra_positions:
                    result += '? '
                    continue
                if self.board[i][j] == 1:
                    result += 'O '
                elif self.board[i][j] == -1:
                    result += 'X '
                else:
                    result += '- '
            result += '\n'
        return result
