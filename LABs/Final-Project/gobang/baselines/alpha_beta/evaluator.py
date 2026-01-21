"""
Board evaluation for Gobang - Exact translation from lihongxun945/gobang eval.js
Evaluates board positions based on pattern recognition and scoring
"""

import numpy as np
from typing import Tuple, Dict, Set, List
from .shape import Shape, get_shape_fast, is_five, is_four, get_all_shapes_of_point


# Exact score values from eval.js
FIVE = 10000000
BLOCK_FIVE = FIVE
FOUR = 100000
FOUR_FOUR = FOUR  # Double blocked four (双冲四)
FOUR_THREE = FOUR  # Blocked four + open three (冲四活三)
THREE_THREE = FOUR // 2  # Double three (双三)
BLOCK_FOUR = 1500
THREE = 1000
BLOCK_THREE = 150
TWO_TWO = 200  # Double open two (双活二)
TWO = 100
BLOCK_TWO = 15
ONE = 10
BLOCK_ONE = 1


def get_real_shape_score(shape: Shape) -> int:
    """
    Convert shape to actual score for empty position.
    CRITICAL: This maps potential shapes to actual scores.
    
    Args:
        shape: The shape that would be created
        
    Returns:
        Score for that position
    """
    if shape == Shape.FIVE:
        return FOUR
    elif shape == Shape.BLOCK_FIVE:
        return BLOCK_FOUR
    elif shape == Shape.FOUR:
        return THREE
    elif shape == Shape.FOUR_FOUR:
        return THREE
    elif shape == Shape.FOUR_THREE:
        return THREE
    elif shape == Shape.BLOCK_FOUR:
        return BLOCK_THREE
    elif shape == Shape.THREE:
        return TWO
    elif shape == Shape.THREE_THREE:
        return THREE_THREE // 10
    elif shape == Shape.BLOCK_THREE:
        return BLOCK_TWO
    elif shape == Shape.TWO:
        return ONE
    elif shape == Shape.TWO_TWO:
        return TWO_TWO // 10
    else:
        return 0


# Direction mappings
ALL_DIRECTIONS = [
    [0, 1],   # Horizontal
    [1, 0],   # Vertical
    [1, 1],   # Diagonal \
    [1, -1]   # Diagonal /
]


def direction_to_index(ox: int, oy: int) -> int:
    """Convert direction offset to index"""
    if ox == 0:
        return 0  # Horizontal
    if oy == 0:
        return 1  # Vertical
    if ox == oy:
        return 2  # Diagonal \
    return 3  # Diagonal /


class BoardEvaluator:
    """
    Evaluates Gobang board positions - exact translation from eval.js
    """
    
    def __init__(self, size: int = 15, bound: int = 5):
        """
        Initialize evaluator.
        
        Args:
            size: Board size (default 15, use 12 for our case)
            bound: Win condition - number in a row to win (default 5)
                   NOTE: Pattern detection is optimized for bound=5.
                   Other values will work for win detection but tactical 
                   evaluation may not be optimal.
        """
        self.size = size
        self.bound = bound
        
        # Board with walls: (size+2) x (size+2)
        # Walls are marked as 2, empty as 0, black as 1, white as -1
        self.board = np.zeros((size + 2, size + 2), dtype=np.int8)
        self.board[0, :] = 2  # Top wall
        self.board[-1, :] = 2  # Bottom wall
        self.board[:, 0] = 2  # Left wall
        self.board[:, -1] = 2  # Right wall
        
        # Score caches for each player
        self.black_scores = np.zeros((size, size), dtype=np.int32)
        self.white_scores = np.zeros((size, size), dtype=np.int32)
        
        # Shape cache: [role][direction][x][y] = shape
        self.shape_cache = {}
        for role in [1, -1]:
            self.shape_cache[role] = {}
            for direction in range(4):
                self.shape_cache[role][direction] = np.zeros((size, size), dtype=np.int8)
        
        self.history = []  # [(position, role), ...]
        
        # Zobrist hashing for position caching
        self._zobrist_table = None
        self._zobrist_hash = 0
        self._init_zobrist()
    
    def _init_zobrist(self):
        """Initialize Zobrist hash table"""
        import random
        random.seed(42)  # Fixed seed for reproducibility
        self._zobrist_table = {}
        for i in range(self.size):
            for j in range(self.size):
                for role in [1, -1]:
                    key = (i, j, role)
                    self._zobrist_table[key] = random.randint(0, 2**63 - 1)
    
    def hash(self) -> int:
        """Get Zobrist hash of current position"""
        return self._zobrist_hash
    
    def is_game_over(self) -> bool:
        """
        Check if game is over (someone won or board is full).
        
        Returns:
            True if game is over, False otherwise
        """
        # Check if board is full
        if not np.any(self.board[1:-1, 1:-1] == 0):
            return True
        
        # Check for winner by examining recent move
        if self.history:
            pos, role = self.history[-1]
            x, y = pos // self.size, pos % self.size
            # Create a simple board view for check_win
            simple_board = self.board[1:-1, 1:-1].copy()
            if self.check_win(simple_board, x, y, role):
                return True
        
        return False
    
    def reverse(self):
        """
        Create a reversed board (swap roles).
        
        Returns:
            New BoardEvaluator with swapped roles
        """
        new_evaluator = BoardEvaluator(self.size)
        for pos, role in self.history:
            x, y = pos // self.size, pos % self.size
            new_evaluator.move(x, y, -role)
        return new_evaluator
    
    @staticmethod
    def get_opponent(player: int) -> int:
        """Get opponent player number"""
        return -player
    
    def move(self, x: int, y: int, role: int):
        """
        Make a move and update scores.
        
        Args:
            x, y: Position (0-indexed)
            role: Player (1 for black, -1 for white)
        """
        # Clear shape cache for this position
        for d in range(4):
            self.shape_cache[role][d][x][y] = 0
            self.shape_cache[-role][d][x][y] = 0
        
        self.black_scores[x][y] = 0
        self.white_scores[x][y] = 0
        
        # Update board
        self.board[x + 1][y + 1] = role
        self.update_point(x, y)
        self.history.append((x * self.size + y, role))
        
        # Update Zobrist hash
        self._zobrist_hash ^= self._zobrist_table[(x, y, role)]
    
    def undo(self, x: int, y: int):
        """
        Undo a move.
        
        Args:
            x, y: Position (0-indexed)
        """
        role = self.board[x + 1][y + 1]
        self.board[x + 1][y + 1] = 0
        self.update_point(x, y)
        if self.history:
            self.history.pop()
        
        # Update Zobrist hash
        if role != 0:
            self._zobrist_hash ^= self._zobrist_table[(x, y, role)]
    
    def update_point(self, x: int, y: int):
        """
        Update scores around a position when it changes.
        
        Args:
            x, y: Position (0-indexed)
        """
        self.update_single_point(x, y, 1)
        self.update_single_point(x, y, -1)
        
        for ox, oy in ALL_DIRECTIONS:
            for sign in [1, -1]:
                for step in range(1, 6):
                    reach_edge = False
                    for role in [1, -1]:
                        nx, ny = x + sign * step * ox + 1, y + sign * step * oy + 1
                        
                        # Check wall
                        if self.board[nx][ny] == 2:
                            reach_edge = True
                            break
                        elif self.board[nx][ny] == -role:
                            continue
                        elif self.board[nx][ny] == 0:
                            self.update_single_point(nx - 1, ny - 1, role, [sign * ox, sign * oy])
                    
                    if reach_edge:
                        break
    
    def update_single_point(self, x: int, y: int, role: int, direction=None):
        """
        Update score for a single point.
        
        Args:
            x, y: Position (0-indexed)
            role: Player
            direction: Optional specific direction to update
        """
        if self.board[x + 1][y + 1] != 0:
            return
        
        # Temporarily place piece
        self.board[x + 1][y + 1] = role
        
        directions = [direction] if direction else ALL_DIRECTIONS
        shape_cache = self.shape_cache[role]
        
        # Clear cache for these directions
        for ox, oy in directions:
            int_direction = direction_to_index(ox, oy)
            shape_cache[int_direction][x][y] = Shape.NONE
        
        score = 0
        blockfour_count = 0
        three_count = 0
        two_count = 0
        
        # Calculate existing scores from other directions
        for int_direction in range(4):
            shape = shape_cache[int_direction][x][y]
            if shape > Shape.NONE:
                score += get_real_shape_score(shape)
                if shape == Shape.BLOCK_FOUR:
                    blockfour_count += 1
                if shape == Shape.THREE:
                    three_count += 1
                if shape == Shape.TWO:
                    two_count += 1
        
        # Calculate new shapes for specified directions
        for ox, oy in directions:
            int_direction = direction_to_index(ox, oy)
            shape, self_count = get_shape_fast(self.board, x, y, ox, oy, role)
            
            if shape:
                shape_cache[int_direction][x][y] = shape
                if shape == Shape.BLOCK_FOUR:
                    blockfour_count += 1
                if shape == Shape.THREE:
                    three_count += 1
                if shape == Shape.TWO:
                    two_count += 1
                
                # Check for combo shapes
                if blockfour_count >= 2:
                    shape = Shape.FOUR_FOUR
                elif blockfour_count and three_count:
                    shape = Shape.FOUR_THREE
                elif three_count >= 2:
                    shape = Shape.THREE_THREE
                elif two_count >= 2:
                    shape = Shape.TWO_TWO
                
                score += get_real_shape_score(shape)
        
        # Remove temporary piece
        self.board[x + 1][y + 1] = 0
        
        # Update score cache
        if role == 1:
            self.black_scores[x][y] = score
        else:
            self.white_scores[x][y] = score
        
        return score
    
    def evaluate(self, role: int) -> int:
        """
        Evaluate the entire board for a given role.
        
        Args:
            role: Player to evaluate for
            
        Returns:
            Score (positive is good for role)
        """
        black_score = np.sum(self.black_scores)
        white_score = np.sum(self.white_scores)
        
        return (black_score - white_score) if role == 1 else (white_score - black_score)
    
    def get_moves(self, role: int, depth: int = 0, only_three: bool = False, 
                  only_four: bool = False) -> List[Tuple[int, int]]:
        """
        Get valuable moves for current position.
        
        Args:
            role: Current player
            depth: Search depth
            only_three: VCT mode - only threes and fours
            only_four: VCF mode - only fours
            
        Returns:
            List of (x, y) positions
        """
        points = self._get_points(role, depth, only_three, only_four)
        moves_set = self._get_moves_from_points(role, depth, only_three, only_four, points)
        
        # Convert position indices to coordinates
        moves = [(pos // self.size, pos % self.size) for pos in moves_set]
        
        # Add center if empty and not in special modes
        if not only_three and not only_four:
            center = self.size // 2
            if self.board[center + 1][center + 1] == 0:
                if (center, center) not in moves:
                    moves.append((center, center))
        
        return moves
    
    def _get_points(self, role: int, depth: int, vct: bool, vcf: bool) -> Dict[Shape, Set[int]]:
        """Get all points organized by shape"""
        first = role if depth % 2 == 0 else -role
        
        points = {shape: set() for shape in Shape}
        
        # Scan all positions
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i + 1][j + 1] != 0:
                    continue
                
                for r in [role, -role]:
                    four_count = 0
                    blockfour_count = 0
                    three_count = 0
                    
                    for direction in range(4):
                        shape = self.shape_cache[r][direction][i][j]
                        if not shape:
                            continue
                        
                        point = i * self.size + j
                        
                        # VCF mode: only fours and fives
                        if vcf:
                            if r == first and not is_four(shape) and not is_five(shape):
                                continue
                            if r == -first and not is_five(shape):
                                continue
                        
                        # VCT mode: complex filtering
                        if vct:
                            if depth % 2 == 0:  # Attack
                                if depth == 0 and r != first:
                                    continue
                                if shape != Shape.THREE and not is_four(shape) and not is_five(shape):
                                    continue
                                if shape == Shape.THREE and r != first:
                                    continue
                                if depth > 0:
                                    if shape == Shape.THREE and len(get_all_shapes_of_point(self.shape_cache, i, j, r)) == 1:
                                        continue
                                    if shape == Shape.BLOCK_FOUR and len(get_all_shapes_of_point(self.shape_cache, i, j, r)) == 1:
                                        continue
                            else:  # Defense
                                if shape != Shape.THREE and not is_four(shape) and not is_five(shape):
                                    continue
                                if shape == Shape.THREE and r == -first:
                                    continue
                        
                        points[shape].add(point)
                        
                        if shape == Shape.FOUR:
                            four_count += 1
                        elif shape == Shape.BLOCK_FOUR:
                            blockfour_count += 1
                        elif shape == Shape.THREE:
                            three_count += 1
                        
                        # Check for combo shapes
                        union_shape = None
                        if four_count >= 2:
                            union_shape = Shape.FOUR_FOUR
                        elif blockfour_count and three_count:
                            union_shape = Shape.FOUR_THREE
                        elif three_count >= 2:
                            union_shape = Shape.THREE_THREE
                        
                        if union_shape:
                            points[union_shape].add(point)
        
        return points
    
    def _get_moves_from_points(self, role: int, depth: int, only_three: bool, 
                                only_four: bool, points: Dict[Shape, Set[int]]) -> Set[int]:
        """
        Filter and prioritize moves from points.
        Matches the priority logic from eval.js getMoves/_getMoves
        """
        fives = points[Shape.FIVE]
        block_fives = points[Shape.BLOCK_FIVE]
        if fives or block_fives:
            return fives | block_fives
        
        fours = points[Shape.FOUR]
        blockfours = points[Shape.BLOCK_FOUR]
        if only_four or fours:
            return fours | blockfours
        
        four_fours = points[Shape.FOUR_FOUR]
        if four_fours:
            return four_fours | blockfours
        
        threes = points[Shape.THREE]
        four_threes = points[Shape.FOUR_THREE]
        if four_threes:
            return four_threes | blockfours | threes
        
        three_threes = points[Shape.THREE_THREE]
        if three_threes:
            return three_threes | blockfours | threes
        
        if only_three:
            return blockfours | threes
        
        blockthrees = points[Shape.BLOCK_THREE]
        two_twos = points[Shape.TWO_TWO]
        twos = points[Shape.TWO]
        
        # Limit to 20 moves (config.pointsLimit from config.js)
        all_moves = list(blockfours | threes | blockthrees | two_twos | twos)
        return set(all_moves[:20])
    
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
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            
            # Count in positive direction
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and board[r, c] == player:
                count += 1
                r, c = r + dr, c + dc
            
            # Count in negative direction
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and board[r, c] == player:
                count += 1
                r, c = r - dr, c - dc
            
            if count >= self.bound:
                return True
        
        return False
