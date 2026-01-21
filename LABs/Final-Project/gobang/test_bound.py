"""
Test to verify bound parameter fixes and win detection
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from baselines.alpha_beta.evaluator import BoardEvaluator

def test_bound_10_detection():
    """Test that wins are detected correctly with bound=10"""
    print("Test: Win detection with bound=10")
    
    # Create evaluator with bound=10
    evaluator = BoardEvaluator(size=12, bound=10)
    
    # Create a board with 10 in a row diagonally
    # Positions (0,0), (1,1), (2,2), ..., (9,9)
    for i in range(10):
        evaluator.move(i, i, 1)
    
    # Check if win is detected
    win_detected = evaluator.check_win(evaluator.board[1:13, 1:13], 9, 9, 1)
    
    print(f"  Placed 10 pieces diagonally")
    print(f"  Win detected: {win_detected}")
    print(f"  Expected: True")
    
    assert win_detected, "Win should be detected with 10 in a row when bound=10"
    print("  ✓ PASSED\n")


def test_bound_5_detection():
    """Test that wins are detected correctly with bound=5"""
    print("Test: Win detection with bound=5")
    
    # Create evaluator with bound=5
    evaluator = BoardEvaluator(size=12, bound=5)
    
    # Create a board with 5 in a row horizontally
    for i in range(5):
        evaluator.move(5, i, 1)
    
    # Check if win is detected
    win_detected = evaluator.check_win(evaluator.board[1:13, 1:13], 5, 4, 1)
    
    print(f"  Placed 5 pieces horizontally")
    print(f"  Win detected: {win_detected}")
    print(f"  Expected: True")
    
    assert win_detected, "Win should be detected with 5 in a row when bound=5"
    print("  ✓ PASSED\n")


def test_six_in_row_bound_10():
    """Test that 6 in a row is NOT a win when bound=10"""
    print("Test: 6 in a row should NOT win when bound=10")
    
    # Create evaluator with bound=10
    evaluator = BoardEvaluator(size=12, bound=10)
    
    # Create a board with only 6 in a row
    for i in range(6):
        evaluator.move(5, i, 1)
    
    # Check if win is detected (should be False)
    win_detected = evaluator.check_win(evaluator.board[1:13, 1:13], 5, 5, 1)
    
    print(f"  Placed 6 pieces horizontally")
    print(f"  Win detected: {win_detected}")
    print(f"  Expected: False (need 10 for bound=10)")
    
    assert not win_detected, "Win should NOT be detected with only 6 in a row when bound=10"
    print("  ✓ PASSED\n")


def test_six_in_row_bound_5():
    """Test that 6 in a row IS a win when bound=5"""
    print("Test: 6 in a row should win when bound=5")
    
    # Create evaluator with bound=5
    evaluator = BoardEvaluator(size=12, bound=5)
    
    # Create a board with 6 in a row
    for i in range(6):
        evaluator.move(5, i, 1)
    
    # Check if win is detected (should be True)
    win_detected = evaluator.check_win(evaluator.board[1:13, 1:13], 5, 5, 1)
    
    print(f"  Placed 6 pieces horizontally")
    print(f"  Win detected: {win_detected}")
    print(f"  Expected: True (6 >= bound=5)")
    
    assert win_detected, "Win should be detected with 6 in a row when bound=5"
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    print("="*60)
    print("Bound Parameter Test Suite")
    print("="*60 + "\n")
    
    try:
        test_bound_5_detection()
        test_six_in_row_bound_5()
        test_bound_10_detection()
        test_six_in_row_bound_10()
        
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
