"""
Shape detection for Gobang patterns.
Ported from https://github.com/lihongxun945/gobang
"""

import re


# Shape constants
class Shapes:
    FIVE = 5
    BLOCK_FIVE = 50
    FOUR = 4
    FOUR_FOUR = 44  # Double block four
    FOUR_THREE = 43  # Block four + live three
    THREE_THREE = 33  # Double three
    BLOCK_FOUR = 40
    THREE = 3
    BLOCK_THREE = 30
    TWO_TWO = 22  # Double live two
    TWO = 2
    NONE = 0


shapes = Shapes()


# Patterns for matching shapes
patterns = {
    'five': re.compile(r'11111'),
    'blockfive': re.compile(r'211111|111112'),
    'four': re.compile(r'011110'),
    'block_four': re.compile(r'10111|11011|11101|211110|211101|211011|210111|011112|101112|110112|111012'),
    'three': re.compile(r'011100|011010|010110|001110'),
    'block_three': re.compile(r'211100|211010|210110|001112|010112|011012'),
    'two': re.compile(r'001100|011000|000110|010100|001010'),
}


def count_shape(board, x, y, offset_x, offset_y, role):
    """Count shapes in one direction."""
    opponent = -role
    
    inner_empty_count = 0  # Empty spaces inside the line
    temp_empty_count = 0
    self_count = 0
    total_length = 0
    
    side_empty_count = 0  # Empty spaces on the side
    no_empty_self_count = 0
    one_empty_self_count = 0
    
    # Search in the direction
    for i in range(1, 6):
        nx, ny = x + i * offset_x + 1, y + i * offset_y + 1
        current_role = board[nx][ny]
        
        if current_role == 2 or current_role == opponent:
            break
        
        if current_role == role:
            self_count += 1
            side_empty_count = 0
            if temp_empty_count:
                inner_empty_count += temp_empty_count
                temp_empty_count = 0
            if inner_empty_count == 0:
                no_empty_self_count += 1
                one_empty_self_count += 1
            elif inner_empty_count == 1:
                one_empty_self_count += 1
        
        total_length += 1
        
        if current_role == 0:
            temp_empty_count += 1
            side_empty_count += 1
        
        if side_empty_count >= 2:
            break
    
    if not inner_empty_count:
        one_empty_self_count = 0
    
    return {
        'self_count': self_count,
        'total_length': total_length,
        'no_empty_self_count': no_empty_self_count,
        'one_empty_self_count': one_empty_self_count,
        'inner_empty_count': inner_empty_count,
        'side_empty_count': side_empty_count
    }


def get_shape_fast(board, x, y, offset_x, offset_y, role):
    """
    Fast shape detection using position traversal.
    About 2x faster than string matching method.
    """
    # Skip empty nodes optimization
    if (board[x + offset_x + 1][y + offset_y + 1] == 0 and
        board[x - offset_x + 1][y - offset_y + 1] == 0 and
        board[x + 2 * offset_x + 1][y + 2 * offset_y + 1] == 0 and
        board[x - 2 * offset_x + 1][y - 2 * offset_y + 1] == 0):
        return (shapes.NONE, 1)
    
    self_count = 1
    total_length = 1
    shape = shapes.NONE
    
    left_empty = 0
    right_empty = 0
    no_empty_self_count = 1
    one_empty_self_count = 1
    
    left = count_shape(board, x, y, -offset_x, -offset_y, role)
    right = count_shape(board, x, y, offset_x, offset_y, role)
    
    self_count = left['self_count'] + right['self_count'] + 1
    total_length = left['total_length'] + right['total_length'] + 1
    no_empty_self_count = left['no_empty_self_count'] + right['no_empty_self_count'] + 1
    one_empty_self_count = max(
        left['one_empty_self_count'] + right['no_empty_self_count'],
        left['no_empty_self_count'] + right['one_empty_self_count']
    ) + 1
    right_empty = right['side_empty_count']
    left_empty = left['side_empty_count']
    
    if total_length < 5:
        return (shape, self_count)
    
    # Five
    if no_empty_self_count >= 5:
        if right_empty > 0 and left_empty > 0:
            return (shapes.FIVE, self_count)
        else:
            return (shapes.BLOCK_FIVE, self_count)
    
    if no_empty_self_count == 4:
        # Four
        right_ok = right_empty >= 1 or right['one_empty_self_count'] > right['no_empty_self_count']
        left_ok = left_empty >= 1 or left['one_empty_self_count'] > left['no_empty_self_count']
        if right_ok and left_ok:
            return (shapes.FOUR, self_count)
        elif not (right_empty == 0 and left_empty == 0):
            return (shapes.BLOCK_FOUR, self_count)
    
    if one_empty_self_count == 4:
        return (shapes.BLOCK_FOUR, self_count)
    
    # Three
    if no_empty_self_count == 3:
        if (right_empty >= 2 and left_empty >= 1) or (right_empty >= 1 and left_empty >= 2):
            return (shapes.THREE, self_count)
        else:
            return (shapes.BLOCK_THREE, self_count)
    
    if one_empty_self_count == 3:
        if right_empty >= 1 and left_empty >= 1:
            return (shapes.THREE, self_count)
        else:
            return (shapes.BLOCK_THREE, self_count)
    
    # Two
    if (no_empty_self_count == 2 or one_empty_self_count == 2) and total_length > 5:
        shape = shapes.TWO
    
    return (shape, self_count)


def is_five(shape):
    """Check if shape is a five."""
    return shape == shapes.FIVE or shape == shapes.BLOCK_FIVE


def is_four(shape):
    """Check if shape is a four."""
    return shape == shapes.FOUR or shape == shapes.BLOCK_FOUR


def get_all_shapes_of_point(shape_cache, x, y, role=None):
    """Get all shapes at a point."""
    roles = [role] if role else [1, -1]
    result = []
    for r in roles:
        for d in range(4):
            shape = shape_cache[r][d][x][y]
            if shape > 0:
                result.append(shape)
    return result
