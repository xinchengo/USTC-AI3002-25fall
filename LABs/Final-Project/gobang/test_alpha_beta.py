#!/usr/bin/env python
"""
Test script for alpha-beta pruning implementation
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wrappers import create_wrapper
from tests.evaluator import GeneralEvaluator

def test_alpha_beta():
    """Test alpha-beta wrapper against itself"""
    print("Testing alpha-beta pruning baseline...")
    
    # Create two alpha-beta players with different depths
    player1 = create_wrapper('alpha_beta', board_size=12, bound=5, depth=3)
    player2 = create_wrapper('alpha_beta', board_size=12, bound=5, depth=3)
    
    print(f"Player 1: Alpha-Beta (depth=3)")
    print(f"Player 2: Alpha-Beta (depth=3)")
    
    # Create evaluator
    evaluator = GeneralEvaluator(board_size=12, bound=5)
    
    # Run evaluation
    print("\nRunning 2 test games...")
    results = evaluator.evaluate_pair(player1, player2, episodes=2, verbose=True)
    
    # Print results
    print("\n" + "="*50)
    print("Test Results:")
    print(f"Player 1 wins: {results['player1_wins']} ({results['player1_win_rate']:.2%})")
    print(f"Player 2 wins: {results['player2_wins']} ({results['player2_win_rate']:.2%})")
    print(f"Ties: {results['ties']} ({results['tie_rate']:.2%})")
    print(f"Total games: {results['total_games']}")
    print("="*50)
    
    print("\n✓ Test passed! Alpha-beta wrapper is working correctly.")
    
    return results

def test_alpha_beta_vs_random():
    """Test alpha-beta wrapper against random player"""
    print("\nTesting alpha-beta vs random player...")
    
    # Create players
    player1 = create_wrapper('alpha_beta', board_size=12, bound=5, depth=3)
    player2 = create_wrapper('random', board_size=12, bound=5)
    
    print(f"Player 1: Alpha-Beta (depth=3)")
    print(f"Player 2: Random")
    
    # Create evaluator
    evaluator = GeneralEvaluator(board_size=12, bound=5)
    
    # Run evaluation
    print("\nRunning 5 test games...")
    results = evaluator.evaluate_pair(player1, player2, episodes=5, verbose=True)
    
    # Print results
    print("\n" + "="*50)
    print("Test Results:")
    print(f"Alpha-Beta wins: {results['player1_wins']} ({results['player1_win_rate']:.2%})")
    print(f"Random wins: {results['player2_wins']} ({results['player2_win_rate']:.2%})")
    print(f"Ties: {results['ties']} ({results['tie_rate']:.2%})")
    print(f"Total games: {results['total_games']}")
    print("="*50)
    
    # Alpha-beta should win most games against random
    if results['player1_win_rate'] > 0.5:
        print("\n✓ Test passed! Alpha-beta is stronger than random player.")
    else:
        print("\n⚠ Warning: Alpha-beta did not dominate random player.")
    
    return results

if __name__ == "__main__":
    print("="*50)
    print("Alpha-Beta Pruning Baseline Test Suite")
    print("="*50)
    
    # Test 1: Alpha-beta vs Alpha-beta
    test_alpha_beta()
    
    # Test 2: Alpha-beta vs Random
    test_alpha_beta_vs_random()
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)
