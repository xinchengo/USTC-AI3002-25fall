"""
Shape detection for Gobang - Exact translation from lihongxun945/gobang
Uses COUNTING-BASED approach (getShapeFast), not regex
"""

from enum import IntEnum
from typing import Tuple


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


def count_shape(board, x, y, offset_x, offset_y, role):
    """
    Count pieces in one direction - exact translation of countShape from shape.js
    
    Args:
        board: Board with walls
        x, y: Starting position (0-indexed in original board)
        offset_x, offset_y: Direction to scan
        role: Current player
        
    Returns:
        dict with selfCount, totalLength, noEmptySelfCount, OneEmptySelfCount, 
        innerEmptyCount, sideEmptyCount
    """
    opponent = -role
    
    inner_empty_count = 0  # Empty spaces inside the pieces
    temp_empty_count = 0
    self_count = 0
    total_length = 0
    
    side_empty_count = 0  # Empty spaces on the side
    no_empty_self_count = 0
    one_empty_self_count = 0
    
    # Scan in the given direction
    for i in range(1, 6):  # Scan up to 5 positions
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
    Fast shape detection using counting - exact translation of getShapeFast from shape.js
    
    Args:
        board: Board array with walls (size+2 x size+2), indexed from 1
        x, y: Position to check (0-indexed in original board)
        offset_x, offset_y: Direction to check
        role: Current player (1 or -1)
        
    Returns:
        tuple: (shape, self_count) where shape is Shape enum value
    """
    # Quick optimization: skip if surrounded by empties
    if (board[x + offset_x + 1][y + offset_y + 1] == 0 and
        board[x - offset_x + 1][y - offset_y + 1] == 0 and
        board[x + 2 * offset_x + 1][y + 2 * offset_y + 1] == 0 and
        board[x - 2 * offset_x + 1][y - 2 * offset_y + 1] == 0):
        return Shape.NONE, 1
    
    self_count = 1
    total_length = 1
    shape = Shape.NONE
    
    left_empty = 0
    right_empty = 0
    no_empty_self_count = 1
    one_empty_self_count = 1
    
    # Count in both directions
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
        return shape, self_count
    
    # FIVE - exactly matches JavaScript logic
    if no_empty_self_count >= 5:
        if right_empty > 0 and left_empty > 0:
            return Shape.FIVE, self_count
        else:
            return Shape.BLOCK_FIVE, self_count
    
    # FOUR
    if no_empty_self_count == 4:
        # Check for open four or blocked four
        if ((right_empty >= 1 or right['one_empty_self_count'] > right['no_empty_self_count']) and
            (left_empty >= 1 or left['one_empty_self_count'] > left['no_empty_self_count'])):
            return Shape.FOUR, self_count
        elif not (right_empty == 0 and left_empty == 0):
            return Shape.BLOCK_FOUR, self_count
    
    if one_empty_self_count == 4:
        return Shape.BLOCK_FOUR, self_count
    
    # THREE
    if no_empty_self_count == 3:
        if ((right_empty >= 2 and left_empty >= 1) or 
            (right_empty >= 1 and left_empty >= 2)):
            return Shape.THREE, self_count
        else:
            return Shape.BLOCK_THREE, self_count
    
    if one_empty_self_count == 3:
        if right_empty >= 1 and left_empty >= 1:
            return Shape.THREE, self_count
        else:
            return Shape.BLOCK_THREE, self_count
    
    # TWO
    if (no_empty_self_count == 2 or one_empty_self_count == 2) and total_length > 5:
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
