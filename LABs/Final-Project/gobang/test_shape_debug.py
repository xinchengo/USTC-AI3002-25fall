"""
Unit tests for shape detection to debug issues
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from baselines.alpha_beta.shape import get_shape_fast, Shape


def test_six_in_a_row():
    """Test that 6 in a row is detected as FIVE"""
    print("Test: 6 in a row should be detected as FIVE")
    
    # Create board with walls
    board = np.zeros((14, 14), dtype=np.int8)
    board[0, :] = 2  # Top wall
    board[-1, :] = 2  # Bottom wall
    board[:, 0] = 2  # Left wall
    board[:, -1] = 2  # Right wall
    
    # Place 6 pieces horizontally: positions (7, 4-9) in board coords
    for col in range(4, 10):
        board[7][col] = 1
    
    # Test detection at empty position next to the line
    shape, count = get_shape_fast(board, 6, 4, 0, 1, 1)
    
    print(f"  Position (6, 4), horizontal direction")
    print(f"  Detected shape: {shape} (name: {Shape(shape).name if shape in Shape._value2member_map_ else 'UNKNOWN'})")
    print(f"  Expected: {Shape.FIVE} (FIVE)")
    print(f"  Count: {count}")
    
    assert shape == Shape.FIVE, f"Expected FIVE but got {Shape(shape).name if shape in Shape._value2member_map_ else shape}"
    print("  ✓ PASSED\n")


def test_five_in_a_row():
    """Test that exactly 5 in a row is detected as FIVE"""
    print("Test: Exactly 5 in a row should be detected as FIVE")
    
    board = np.zeros((14, 14), dtype=np.int8)
    board[0, :] = 2
    board[-1, :] = 2
    board[:, 0] = 2
    board[:, -1] = 2
    
    # Place exactly 5 pieces: positions (7, 4-8)
    for col in range(4, 9):
        board[7][col] = 1
    
    shape, count = get_shape_fast(board, 6, 4, 0, 1, 1)
    
    print(f"  Detected shape: {Shape(shape).name if shape in Shape._value2member_map_ else 'UNKNOWN'}")
    print(f"  Expected: FIVE")
    print(f"  Count: {count}")
    
    assert shape == Shape.FIVE, f"Expected FIVE but got {Shape(shape).name}"
    print("  ✓ PASSED\n")


def test_open_three():
    """Test that open three (活三) is detected"""
    print("Test: Open three should be detected")
    
    board = np.zeros((14, 14), dtype=np.int8)
    board[0, :] = 2
    board[-1, :] = 2
    board[:, 0] = 2
    board[:, -1] = 2
    
    # Pattern: _XXX_ (positions 5, 6, 7 occupied, 4 and 8 empty)
    board[7][5] = 1
    board[7][6] = 1
    board[7][7] = 1
    
    # Test at position 4 (empty before the three)
    shape, count = get_shape_fast(board, 6, 4, 0, 1, 1)
    
    print(f"  Position (6, 4) - empty before three pieces")
    print(f"  Detected shape: {Shape(shape).name if shape in Shape._value2member_map_ else 'UNKNOWN'}")
    print(f"  Expected: THREE")
    print(f"  Count: {count}")
    
    # Open three should be detected
    assert shape in (Shape.THREE, Shape.FOUR), f"Expected THREE or FOUR but got {Shape(shape).name if shape in Shape._value2member_map_ else shape}"
    print("  ✓ PASSED\n")


def test_open_four():
    """Test that open four (活四) is detected"""
    print("Test: Open four should be detected")
    
    board = np.zeros((14, 14), dtype=np.int8)
    board[0, :] = 2
    board[-1, :] = 2
    board[:, 0] = 2
    board[:, -1] = 2
    
    # Pattern: _XXXX_ (positions 5-8 occupied, 4 and 9 empty)
    for col in range(5, 9):
        board[7][col] = 1
    
    # Test at position 4 (empty before the four)
    shape, count = get_shape_fast(board, 6, 4, 0, 1, 1)
    
    print(f"  Position (6, 4) - empty before four pieces")
    print(f"  Detected shape: {Shape(shape).name if shape in Shape._value2member_map_ else 'UNKNOWN'}")
    print(f"  Expected: FOUR")
    print(f"  Count: {count}")
    
    assert shape == Shape.FOUR, f"Expected FOUR but got {Shape(shape).name if shape in Shape._value2member_map_ else shape}"
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    print("="*60)
    print("Shape Detection Unit Tests (Counting-Based)")
    print("="*60 + "\n")
    
    try:
        test_five_in_a_row()
        test_six_in_a_row()
        test_open_three()
        test_open_four()
        print("="*60)
        print("All tests PASSED!")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
