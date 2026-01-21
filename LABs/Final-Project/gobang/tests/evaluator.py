import numpy as np
from typing import Tuple, Dict, Any
from wrappers import create_wrapper
from utils import Gobang
from tqdm import tqdm
import argparse
import os


class GeneralEvaluator:
    """
    General evaluator that can evaluate any pair of players/wrappers against each other.
    Supports evaluation between different types of strategies (checkpointed models, random, etc.)
    """
    
    def __init__(self, board_size: int = 12, bound: int = 5):
        """
        Initialize the evaluator.
        
        Args:
            board_size: Size of the board (default 12)
            bound: Number of pieces in a row to win (default 5)
        """
        self.board_size = board_size
        self.bound = bound
    
    def evaluate_pair(self, player1, player2, episodes: int = 1000, verbose: bool = False) -> Dict[str, Any]:
        """
        Evaluate two players against each other.
        
        Args:
            player1: First player (any wrapper with get_action method)
            player2: Second player (any wrapper with get_action method)
            episodes: Number of episodes to evaluate (default 1000)
            verbose: Whether to print progress (default False)
            
        Returns:
            Dictionary with evaluation results
        """
        results = {
            'player1_wins': 0,
            'player2_wins': 0,
            'ties': 0,
            'total_games': episodes,
            'player1_win_rate': 0.0,
            'player2_win_rate': 0.0,
            'tie_rate': 0.0
        }
        
        for episode in tqdm(range(episodes), desc="Evaluating", disable=not verbose):
            # Create a fresh game board for each episode
            game = Gobang(board_size=self.board_size, bound=self.bound, training=False)
            game.restart()
            
            # Alternate who goes first to be fair
            first_player = player1 if episode % 2 == 0 else player2
            second_player = player2 if episode % 2 == 0 else player1
            first_player_id = 1 if episode % 2 == 0 else 2
            second_player_id = 2 if episode % 2 == 0 else 1
            
            # Play the game
            winner = self._play_game(game, first_player, second_player, first_player_id, second_player_id)
            
            # Record results
            if winner == first_player_id:
                if episode % 2 == 0:
                    results['player1_wins'] += 1
                else:
                    results['player2_wins'] += 1
            elif winner == second_player_id:
                if episode % 2 == 0:
                    results['player2_wins'] += 1
                else:
                    results['player1_wins'] += 1
            else:  # Tie
                results['ties'] += 1
        
        # Calculate win rates
        results['player1_win_rate'] = results['player1_wins'] / episodes
        results['player2_win_rate'] = results['player2_wins'] / episodes
        results['tie_rate'] = results['ties'] / episodes
        
        return results
    
    def _play_game(self, game: Gobang, first_player, second_player, first_player_id: int, second_player_id: int) -> int:
        """
        Play a single game between two players.
        
        Args:
            game: Game instance
            first_player: Player that goes first
            second_player: Player that goes second
            first_player_id: ID of first player (1 or 2)
            second_player_id: ID of second player (1 or 2)
            
        Returns:
            Winner ID (1, 2) or 0 for tie
        """
        current_player = first_player
        current_player_id = first_player_id
        
        while True:
            # Get action from current player
            row, col = current_player.get_action(game.board)
            
            # Check if move is valid
            if row == -1 and col == -1:
                # No valid moves, game ends in tie
                return 0
            
            # Make the move
            game.board[row][col] = current_player_id
            game.action_space.remove((row, col))
            
            # Check for win
            black_conn, white_conn = game.count_max_connections(game.board)
            if (first_player_id == 1 and black_conn >= self.bound) or \
               (first_player_id == 2 and white_conn >= self.bound):
                return first_player_id
            elif (second_player_id == 1 and black_conn >= self.bound) or \
                 (second_player_id == 2 and white_conn >= self.bound):
                return second_player_id
            
            # Check for tie (board full)
            if len(game.action_space) == 0:
                return 0
            
            # Switch to other player
            if current_player is first_player:
                current_player = second_player
                current_player_id = second_player_id
            else:
                current_player = first_player
                current_player_id = first_player_id


def main():
    """Main function to run evaluation between different types of players."""
    parser = argparse.ArgumentParser(description='Evaluate different Gobang players against each other')
    parser.add_argument('--player1_path', type=str, default=None,
                       help='Path to player 1 model (.pth or .pkl file)')
    parser.add_argument('--player2_path', type=str, default=None,
                       help='Path to player 2 model (.pth or .pkl file)')
    parser.add_argument('--player1_type', type=str, choices=['checkpoint', 'random', 'baseline', 'alpha_beta'], default='checkpoint',
                       help='Type of player 1 (default: checkpoint)')
    parser.add_argument('--player2_type', type=str, choices=['checkpoint', 'random', 'baseline', 'alpha_beta'], default='random',
                       help='Type of player 2 (default: random)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes to evaluate (default: 100)')
    parser.add_argument('--board_size', type=int, default=12,
                       help='Size of the board (default: 12)')
    parser.add_argument('--bound', type=int, default=5,
                       help='Number of pieces in a row to win (default: 5)')
    parser.add_argument('--depth', type=int, default=4,
                       help='Search depth for alpha_beta players (default: 4, range: 2-10)')
    parser.add_argument('--difficulty', type=str, choices=['weak', 'easy', 'medium', 'hard'], default=None,
                       help='Difficulty level for alpha_beta players (weak=depth 2, easy=depth 4, medium=depth 6, hard=depth 8). Overrides --depth.')
    parser.add_argument('--verbose', action='store_true',
                       help='Print progress during evaluation')
    
    args = parser.parse_args()
    
    # Map difficulty to depth if specified
    if args.difficulty:
        difficulty_map = {
            'weak': 2,
            'easy': 4,
            'medium': 6,
            'hard': 8
        }
        args.depth = difficulty_map[args.difficulty]
        print(f"Using difficulty={args.difficulty} (depth={args.depth})")
    
    # Initialize evaluator
    evaluator = GeneralEvaluator(board_size=args.board_size, bound=args.bound)
    
    # Create player 1 using factory
    player1 = create_wrapper(
        wrapper_type=args.player1_type,
        model_path=args.player1_path,
        board_size=args.board_size,
        bound=args.bound,
        depth=args.depth
    )

    # Create player 2 using factory
    player2 = create_wrapper(
        wrapper_type=args.player2_type,
        model_path=args.player2_path,
        board_size=args.board_size,
        bound=args.bound,
        depth=args.depth
    )
    
    # Run evaluation
    print(f"Evaluating {args.player1_type} vs {args.player2_type} for {args.episodes} episodes...")
    results = evaluator.evaluate_pair(player1, player2, episodes=args.episodes, verbose=args.verbose)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Player 1 ({args.player1_type}) wins: {results['player1_wins']} ({results['player1_win_rate']:.2%})")
    print(f"Player 2 ({args.player2_type}) wins: {results['player2_wins']} ({results['player2_win_rate']:.2%})")
    print(f"Ties: {results['ties']} ({results['tie_rate']:.2%})")
    print(f"Total games: {results['total_games']}")


if __name__ == "__main__":
    main()