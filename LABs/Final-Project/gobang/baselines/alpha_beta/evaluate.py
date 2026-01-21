"""
Position evaluation for Gobang.
Ported from https://github.com/lihongxun945/gobang
"""

from .shape import shapes, get_shape_fast, is_five, is_four, get_all_shapes_of_point
from .config import config

# Score constants
FIVE = 10000000
BLOCK_FIVE = FIVE
FOUR = 100000
FOUR_FOUR = FOUR  # Double block four
FOUR_THREE = FOUR  # Block four + live three
THREE_THREE = FOUR // 2  # Double three
BLOCK_FOUR = 1500
THREE = 1000
BLOCK_THREE = 150
TWO_TWO = 200  # Double live two
TWO = 100
BLOCK_TWO = 15
ONE = 10
BLOCK_ONE = 1

# All directions
ALL_DIRECTIONS = [
    (0, 1),   # Horizontal
    (1, 0),   # Vertical
    (1, 1),   # Diagonal \
    (1, -1)   # Diagonal /
]


def direction_to_index(ox, oy):
    """Convert direction to index."""
    if ox == 0:
        return 0  # |
    if oy == 0:
        return 1  # -
    if ox == oy:
        return 2  # \
    return 3  # /


def get_real_shape_score(shape):
    """Get score for a shape (before placing a piece)."""
    if shape == shapes.FIVE:
        return FOUR
    elif shape == shapes.BLOCK_FIVE:
        return BLOCK_FOUR
    elif shape == shapes.FOUR:
        return THREE
    elif shape == shapes.FOUR_FOUR:
        return THREE
    elif shape == shapes.FOUR_THREE:
        return THREE
    elif shape == shapes.BLOCK_FOUR:
        return BLOCK_THREE
    elif shape == shapes.THREE:
        return TWO
    elif shape == shapes.THREE_THREE:
        return THREE_THREE // 10
    elif shape == shapes.BLOCK_THREE:
        return BLOCK_TWO
    elif shape == shapes.TWO:
        return ONE
    elif shape == shapes.TWO_TWO:
        return TWO_TWO // 10
    return 0


def coordinate_to_position(x, y, size):
    """Convert coordinate to position."""
    return x * size + y


def position_to_coordinate(position, size):
    """Convert position to coordinate."""
    return (position // size, position % size)


def is_line(a, b, size):
    """Check if two positions are in a line within max distance."""
    max_distance = config.in_line_distance
    x1, y1 = position_to_coordinate(a, size)
    x2, y2 = position_to_coordinate(b, size)
    return (
        (x1 == x2 and abs(y1 - y2) < max_distance) or
        (y1 == y2 and abs(x1 - x2) < max_distance) or
        (abs(x1 - x2) == abs(y1 - y2) and abs(x1 - x2) < max_distance)
    )


def has_in_line(p, arr, size):
    """Check if position p is in line with any position in arr."""
    for pos in arr:
        if is_line(p, pos, size):
            return True
    return False


class Evaluate:
    """Evaluator for Gobang board positions."""
    
    def __init__(self, size=15):
        self.size = size
        # Board with wall (size+2 x size+2), wall marked as 2
        self.board = [
            [2 if (i == 0 or j == 0 or i == size + 1 or j == size + 1) else 0
             for j in range(size + 2)]
            for i in range(size + 2)
        ]
        self.black_scores = [[0] * size for _ in range(size)]
        self.white_scores = [[0] * size for _ in range(size)]
        self._init_points()
        self.history = []  # Records [position, role]
    
    def _init_points(self):
        """Initialize shape cache and points cache."""
        # Cache: [role][direction][x][y] = shape
        self.shape_cache = {}
        for role in [1, -1]:
            self.shape_cache[role] = {}
            for direction in range(4):
                self.shape_cache[role][direction] = [
                    [shapes.NONE] * self.size for _ in range(self.size)
                ]
        
        # Points cache: points_cache[role][shape] = set of positions
        self.points_cache = {}
        for role in [1, -1]:
            self.points_cache[role] = {}
            for shape_val in [shapes.FIVE, shapes.BLOCK_FIVE, shapes.FOUR,
                             shapes.FOUR_FOUR, shapes.FOUR_THREE, shapes.THREE_THREE,
                             shapes.BLOCK_FOUR, shapes.THREE, shapes.BLOCK_THREE,
                             shapes.TWO_TWO, shapes.TWO, shapes.NONE]:
                self.points_cache[role][shape_val] = set()
    
    def move(self, x, y, role):
        """Make a move."""
        # Clear records
        for d in range(4):
            self.shape_cache[role][d][x][y] = 0
            self.shape_cache[-role][d][x][y] = 0
        self.black_scores[x][y] = 0
        self.white_scores[x][y] = 0
        
        # Update board and scores
        self.board[x + 1][y + 1] = role
        self._update_point(x, y)
        self.history.append((coordinate_to_position(x, y, self.size), role))
    
    def undo(self, x, y):
        """Undo a move."""
        self.board[x + 1][y + 1] = 0
        self._update_point(x, y)
        self.history.pop()
    
    def _get_points_in_line(self, role):
        """Get points that are in line with recent moves."""
        points_in_line = {
            shapes.FIVE: set(), shapes.BLOCK_FIVE: set(),
            shapes.FOUR: set(), shapes.FOUR_FOUR: set(),
            shapes.FOUR_THREE: set(), shapes.THREE_THREE: set(),
            shapes.BLOCK_FOUR: set(), shapes.THREE: set(),
            shapes.BLOCK_THREE: set(), shapes.TWO_TWO: set(),
            shapes.TWO: set(), shapes.NONE: set()
        }
        has_points_in_line = False
        
        last_points = [pos for pos, _ in self.history[-config.inline_count:]]
        processed = {}
        
        for r in [role, -role]:
            for point in last_points:
                x, y = position_to_coordinate(point, self.size)
                for ox, oy in ALL_DIRECTIONS:
                    for sign in [1, -1]:
                        for step in range(1, config.in_line_distance + 1):
                            nx, ny = x + sign * step * ox, y + sign * step * oy
                            position = coordinate_to_position(nx, ny, self.size)
                            
                            if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
                                break
                            if self.board[nx + 1][ny + 1] != 0:
                                continue
                            if processed.get(position) == r:
                                continue
                            processed[position] = r
                            
                            for direction in range(4):
                                shape = self.shape_cache[r][direction][nx][ny]
                                if shape:
                                    points_in_line[shape].add(coordinate_to_position(nx, ny, self.size))
                                    has_points_in_line = True
        
        if has_points_in_line:
            return points_in_line
        return None
    
    def get_points(self, role, depth, vct=False, vcf=False):
        """Get all valuable points."""
        first = role if depth % 2 == 0 else -role
        
        if config.only_in_line and len(self.history) >= config.inline_count:
            points_in_line = self._get_points_in_line(role)
            if points_in_line:
                return points_in_line
        
        points = {
            shapes.FIVE: set(), shapes.BLOCK_FIVE: set(),
            shapes.FOUR: set(), shapes.FOUR_FOUR: set(),
            shapes.FOUR_THREE: set(), shapes.THREE_THREE: set(),
            shapes.BLOCK_FOUR: set(), shapes.THREE: set(),
            shapes.BLOCK_THREE: set(), shapes.TWO_TWO: set(),
            shapes.TWO: set(), shapes.NONE: set()
        }
        
        last_points = [pos for pos, _ in self.history[-4:]]
        
        for r in [role, -role]:
            for i in range(self.size):
                for j in range(self.size):
                    four_count = 0
                    block_four_count = 0
                    three_count = 0
                    
                    for direction in range(4):
                        if self.board[i + 1][j + 1] != 0:
                            continue
                        shape = self.shape_cache[r][direction][i][j]
                        if not shape:
                            continue
                        
                        # VCF filter
                        if vcf:
                            if r == first and not is_four(shape) and not is_five(shape):
                                continue
                            if r == -first and is_five(shape):
                                continue
                        
                        point = i * self.size + j
                        
                        # VCT filter
                        if vct:
                            if depth % 2 == 0:
                                if depth == 0 and r != first:
                                    continue
                                if shape != shapes.THREE and not is_four(shape) and not is_five(shape):
                                    continue
                                if shape == shapes.THREE and r != first:
                                    continue
                                if depth == 0 and r != first:
                                    continue
                                if depth > 0:
                                    if shape == shapes.THREE and len(get_all_shapes_of_point(self.shape_cache, i, j, r)) == 1:
                                        continue
                                    if shape == shapes.BLOCK_FOUR and len(get_all_shapes_of_point(self.shape_cache, i, j, r)) == 1:
                                        continue
                            else:
                                if shape != shapes.THREE and not is_four(shape) and not is_five(shape):
                                    continue
                                if shape == shapes.THREE and r == -first:
                                    continue
                                if depth > 1:
                                    if shape == shapes.BLOCK_FOUR and len(get_all_shapes_of_point(self.shape_cache, i, j)) == 1:
                                        continue
                                    if shape == shapes.BLOCK_FOUR and not has_in_line(point, last_points, self.size):
                                        continue
                        
                        if vcf:
                            if not is_four(shape) and not is_five(shape):
                                continue
                        
                        # Skip low-value points that are not in line
                        if depth > 2 and shape in [shapes.TWO, shapes.TWO_TWO, shapes.BLOCK_THREE]:
                            if not has_in_line(point, last_points, self.size):
                                continue
                        
                        points[shape].add(point)
                        if shape == shapes.FOUR:
                            four_count += 1
                        elif shape == shapes.BLOCK_FOUR:
                            block_four_count += 1
                        elif shape == shapes.THREE:
                            three_count += 1
                        
                        # Check for union shapes
                        union_shape = None
                        if four_count >= 2:
                            union_shape = shapes.FOUR_FOUR
                        elif block_four_count and three_count:
                            union_shape = shapes.FOUR_THREE
                        elif three_count >= 2:
                            union_shape = shapes.THREE_THREE
                        
                        if union_shape:
                            points[union_shape].add(point)
        
        return points
    
    def _update_point(self, x, y):
        """Update scores around a changed position."""
        self._update_single_point(x, y, 1)
        self._update_single_point(x, y, -1)
        
        for ox, oy in ALL_DIRECTIONS:
            for sign in [1, -1]:
                for step in range(1, 6):
                    reach_edge = False
                    for role in [1, -1]:
                        nx, ny = x + sign * step * ox + 1, y + sign * step * oy + 1
                        if self.board[nx][ny] == 2:
                            reach_edge = True
                            break
                        elif self.board[nx][ny] == -role:
                            continue
                        elif self.board[nx][ny] == 0:
                            self._update_single_point(nx - 1, ny - 1, role, (sign * ox, sign * oy))
                    if reach_edge:
                        break
    
    def _update_single_point(self, x, y, role, direction=None):
        """Update score for a single point."""
        if self.board[x + 1][y + 1] != 0:
            return
        
        # Temporarily place the piece
        self.board[x + 1][y + 1] = role
        
        directions = [direction] if direction else ALL_DIRECTIONS
        shape_cache = self.shape_cache[role]
        
        # Clear cache
        for ox, oy in directions:
            shape_cache[direction_to_index(ox, oy)][x][y] = shapes.NONE
        
        score = 0
        block_four_count = 0
        three_count = 0
        two_count = 0
        
        # Calculate existing score
        for int_direction in range(4):
            shape = shape_cache[int_direction][x][y]
            if shape > shapes.NONE:
                score += get_real_shape_score(shape)
                if shape == shapes.BLOCK_FOUR:
                    block_four_count += 1
                if shape == shapes.THREE:
                    three_count += 1
                if shape == shapes.TWO:
                    two_count += 1
        
        for ox, oy in directions:
            int_direction = direction_to_index(ox, oy)
            shape, self_count = get_shape_fast(self.board, x, y, ox, oy, role)
            if not shape:
                continue
            
            shape_cache[int_direction][x][y] = shape
            if shape == shapes.BLOCK_FOUR:
                block_four_count += 1
            if shape == shapes.THREE:
                three_count += 1
            if shape == shapes.TWO:
                two_count += 1
            
            # Check for union shapes
            if block_four_count >= 2:
                shape = shapes.FOUR_FOUR
            elif block_four_count and three_count:
                shape = shapes.FOUR_THREE
            elif three_count >= 2:
                shape = shapes.THREE_THREE
            elif two_count >= 2:
                shape = shapes.TWO_TWO
            
            score += get_real_shape_score(shape)
        
        # Remove temporary piece
        self.board[x + 1][y + 1] = 0
        
        if role == 1:
            self.black_scores[x][y] = score
        else:
            self.white_scores[x][y] = score
        
        return score
    
    def evaluate(self, role):
        """Evaluate the entire board for a role."""
        black_score = sum(sum(row) for row in self.black_scores)
        white_score = sum(sum(row) for row in self.white_scores)
        
        if role == 1:
            return black_score - white_score
        else:
            return white_score - black_score
    
    def get_moves(self, role, depth, only_three=False, only_four=False):
        """Get valuable moves."""
        moves = list(self._get_moves(role, depth, only_three, only_four))
        return [(move // self.size, move % self.size) for move in moves]
    
    def _get_moves(self, role, depth, only_three=False, only_four=False):
        """Internal method to get moves."""
        points = self.get_points(role, depth, only_three, only_four)
        
        fives = points[shapes.FIVE]
        block_fives = points[shapes.BLOCK_FIVE]
        if fives or block_fives:
            return fives | block_fives
        
        fours = points[shapes.FOUR]
        block_fours = points[shapes.BLOCK_FOUR]
        if only_four or fours:
            return fours | block_fours
        
        four_fours = points[shapes.FOUR_FOUR]
        if four_fours:
            return four_fours | block_fours
        
        threes = points[shapes.THREE]
        four_threes = points[shapes.FOUR_THREE]
        if four_threes:
            return four_threes | block_fours | threes
        
        three_threes = points[shapes.THREE_THREE]
        if three_threes:
            return three_threes | block_fours | threes
        
        if only_three:
            return block_fours | threes
        
        block_threes = points[shapes.BLOCK_THREE]
        two_twos = points[shapes.TWO_TWO]
        twos = points[shapes.TWO]
        
        all_moves = list(block_fours | threes | block_threes | two_twos | twos)
        return set(all_moves[:config.points_limit])
