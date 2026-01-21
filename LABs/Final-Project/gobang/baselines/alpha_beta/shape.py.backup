"""
Shape detection for Gobang - Direct translation from lihongxun945/gobang
Detects patterns like five, four, three, two using regex-style matching
"""

import re
from enum import IntEnum


class Shape(IntEnum):
    """Shape types matching the JavaScript implementation exactly"""
    FIVE = 5
    BLOCK_FIVE = 50
    FOUR = 4
    FOUR_FOUR = 44  # Double blocked four (双冲四)
    FOUR_THREE = 43  # Blocked four + open three (冲四活三)
    THREE_THREE = 33  # Double three (双三)
    BLOCK_FOUR = 40
    THREE = 3
    BLOCK_THREE = 30
    TWO_TWO = 22  # Double open two (双活二)
    TWO = 2
    NONE = 0


# Pattern definitions matching JavaScript regex patterns exactly
# 1 = current player, 2 = opponent/wall, 0 = empty
PATTERNS = {
    'five': re.compile(r'11111'),
    'blockfive': re.compile(r'211111|111112'),
    'four': re.compile(r'011110'),
    'blockFour': re.compile(r'10111|11011|11101|211110|211101|211011|210111|011112|101112|110112|111012'),
    'three': re.compile(r'011100|011010|010110|001110'),
    'blockThree': re.compile(r'211100|211010|210110|001112|010112|011012'),
    'two': re.compile(r'001100|011000|000110|010100|001010'),
}


def get_shape_fast(board, x, y, offset_x, offset_y, role):
    """
    Fast shape detection - translation of getShapeFast from shape.js
    
    Args:
        board: Board array with walls (size+2 x size+2), indexed from 1
        x, y: Position to check (0-indexed in original board)
        offset_x, offset_y: Direction to check
        role: Current player (1 or -1)
        
    Returns:
        tuple: (shape, self_count) where shape is Shape enum value
    """
    opponent = -role
    empty_count = 0
    self_count = 1
    opponent_count = 0
    
    # Quick optimization: skip if surrounded by empties
    if (board[x + offset_x + 1][y + offset_y + 1] == 0 and
        board[x - offset_x + 1][y - offset_y + 1] == 0 and
        board[x + 2 * offset_x + 1][y + 2 * offset_y + 1] == 0 and
        board[x - 2 * offset_x + 1][y - 2 * offset_y + 1] == 0):
        return Shape.NONE, self_count
    
    # TWO optimization - TWO is over 50% of patterns
    for i in range(-3, 4):
        if i == 0:
            continue
        nx, ny = x + i * offset_x + 1, y + i * offset_y + 1
        if nx < 0 or ny < 0 or nx >= len(board) or ny >= len(board[0]):
            continue
        current_role = board[nx][ny]
        if current_role == 2:
            opponent_count += 1
        elif current_role == role:
            self_count += 1
        elif current_role == 0:
            empty_count += 1
    
    if self_count == 2:
        if not opponent_count:
            return Shape.TWO, self_count
        else:
            return Shape.NONE, self_count
    
    # Build pattern string
    result_string = '1'
    empty_count = 0
    self_count = 1
    opponent_count = 0
    
    # Scan forward
    for i in range(1, 6):
        nx, ny = x + i * offset_x + 1, y + i * offset_y + 1
        current_role = board[nx][ny]
        if current_role == 2:
            result_string += '2'
        elif current_role == 0:
            result_string += '0'
        else:
            result_string += '1' if current_role == role else '2'
        
        if current_role == 2 or current_role == opponent:
            opponent_count += 1
            break
        if current_role == 0:
            empty_count += 1
        if current_role == role:
            self_count += 1
    
    # Scan backward
    for i in range(1, 6):
        nx, ny = x - i * offset_x + 1, y - i * offset_y + 1
        current_role = board[nx][ny]
        if current_role == 2:
            result_string = '2' + result_string
        elif current_role == 0:
            result_string = '0' + result_string
        else:
            result_string = ('1' if current_role == role else '2') + result_string
        
        if current_role == 2 or current_role == opponent:
            opponent_count += 1
            break
        if current_role == 0:
            empty_count += 1
        if current_role == role:
            self_count += 1
    
    # Match patterns in priority order
    shape = Shape.NONE
    if PATTERNS['five'].search(result_string):
        shape = Shape.FIVE
    elif PATTERNS['four'].search(result_string):
        shape = Shape.FOUR
    elif PATTERNS['blockfive'].search(result_string):
        shape = Shape.BLOCK_FIVE
    elif PATTERNS['blockFour'].search(result_string):
        shape = Shape.BLOCK_FOUR
    elif PATTERNS['three'].search(result_string):
        shape = Shape.THREE
    elif PATTERNS['blockThree'].search(result_string):
        shape = Shape.BLOCK_THREE
    elif PATTERNS['two'].search(result_string):
        shape = Shape.TWO
    
    return shape, self_count


def is_five(shape):
    """Check if shape is five in a row"""
    return shape == Shape.FIVE or shape == Shape.BLOCK_FIVE


def is_four(shape):
    """Check if shape is four (open or blocked)"""
    return shape in (Shape.FOUR, Shape.FOUR_FOUR, Shape.FOUR_THREE, Shape.BLOCK_FOUR)


def get_all_shapes_of_point(shape_cache, x, y, role):
    """
    Get all shapes at a given point for a role.
    
    Args:
        shape_cache: The shape cache structure [role][direction][x][y]
        x, y: Position to check
        role: Player role
        
    Returns:
        List of shapes at this point
    """
    shapes = []
    for direction in range(4):
        shape = shape_cache[role][direction][x][y]
        if shape != Shape.NONE:
            shapes.append(shape)
    return shapes
