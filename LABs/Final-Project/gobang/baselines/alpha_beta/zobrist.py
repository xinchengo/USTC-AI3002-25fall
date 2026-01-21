"""
Zobrist hashing for board positions.
Ported from https://github.com/lihongxun945/gobang
"""

import random


class Zobrist:
    """Zobrist hashing implementation for board positions."""
    
    # Use a fixed seed for reproducibility
    _SEED = 42
    
    def __init__(self, size):
        self.size = size
        self.zobrist_table = self._initialize_zobrist_table(size)
        self.hash = 0
    
    def _initialize_zobrist_table(self, size):
        """Initialize the Zobrist table with random values."""
        # Use a dedicated random generator with fixed seed for reproducibility
        rng = random.Random(self._SEED)
        
        table = []
        for i in range(size):
            row = []
            for j in range(size):
                row.append({
                    1: rng.getrandbits(64),   # black
                    -1: rng.getrandbits(64)   # white
                })
            table.append(row)
        return table
    
    def toggle_piece(self, x, y, role):
        """Toggle a piece at position (x, y)."""
        self.hash ^= self.zobrist_table[x][y][role]
    
    def get_hash(self):
        """Get the current hash value."""
        return self.hash
