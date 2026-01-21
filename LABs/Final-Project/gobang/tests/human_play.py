"""
human VS AI models
Input your move in the format: 2,3

@author: Adapted from Junxiao Song's AlphaZero implementation
"""

from __future__ import print_function
import numpy as np
from wrappers.checkpoint import CheckpointWrapper
from wrappers.random import RandomWrapper
from utils import Gobang, _position_to_index, _index_to_position
import argparse
import os


class Human:
    """
    human player
    """

    def __init__(self, board_size=12):
        self.player = None
        self.board_size = board_size

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        """
        Get human player's move via console input.
        Input format: 2,3 (row, column)
        """
        try:
            location = input("Your move (row,col): ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            if len(location) != 2:
                raise ValueError("Invalid input length")
            move = _position_to_index(self.board_size, location[0], location[1])
        except Exception as e:
            move = -1
        if move == -1 or move not in [i * self.board_size + j for i in range(self.board_size) for j in range(self.board_size) if board[i][j] == 0]:
            print("Invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run(board_size=12, bound=5, model_path=None):
    """
    Run human vs AI game
    """
    try:
        # Create game board
        game_board = Gobang(board_size=board_size, bound=bound, training=False)
        game_board.restart()

        # Create AI player
        if model_path and os.path.exists(model_path):
            ai_wrapper = CheckpointWrapper(
                model_path=model_path,
                board_size=board_size,
                bound=bound
            )
            print(f"Using AI model: {model_path}")
        else:
            ai_wrapper = RandomWrapper(board_size=board_size, bound=bound)
            print("Using random AI (no model provided)")

        # Create human player
        human = Human(board_size=board_size)

        # Start playing
        print("Game started! Board size: {}x{}, Win condition: {} in a row".format(board_size, board_size, bound))
        print("Input your move in the format: row,col (e.g., 5,6)")
        print("Row and column numbers start from 0")
        print_board(game_board.board)

        # Play alternately
        current_player = 1  # Start with human player (typically 1 for black)
        while True:
            if current_player == 1:  # Human turn
                print("Your turn (Player 1 - Black)")
                move = human.get_action(game_board.board)
                x, y = _index_to_position(board_size, move)
                game_board.board[x][y] = 1
                game_board.action_space.remove((x, y))
            else:  # AI turn
                print("AI's turn (Player 2 - White)")
                # Get AI action
                x, y = ai_wrapper.get_action(game_board.board)
                if x != -1 and y != -1:
                    game_board.board[x][y] = 2
                    game_board.action_space.remove((x, y))
                else:
                    print("AI has no valid moves")
                    break

            print_board(game_board.board)

            # Check for win
            black_conn, white_conn = game_board.count_max_connections(game_board.board)
            if black_conn >= bound:
                print("Player 1 (Black) wins!")
                break
            elif white_conn >= bound:
                print("Player 2 (White) wins!")
                break
            elif len(game_board.action_space) == 0:
                print("Tie!")
                break

            # Switch player
            current_player = 3 - current_player  # Switch between 1 and 2

    except KeyboardInterrupt:
        print('\n\rquit')


def print_board(board):
    """
    Print the current board state in a readable format
    """
    board_size = len(board)
    print("  ", end="")
    for j in range(board_size):
        print("{:>3}".format(j), end="")
    print()
    for i in range(board_size):
        print("{:>2}".format(i), end="")
        for j in range(board_size):
            if board[i][j] == 0:
                print(" . ", end="")
            elif board[i][j] == 1:
                print(" X ", end="")  # Human player
            else:
                print(" O ", end="")  # AI player
        print()
    print()


def main():
    """Main function to start human vs AI game."""
    parser = argparse.ArgumentParser(description='Human vs AI Gobang Game')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to AI model (.pth or .pkl file)')
    parser.add_argument('--board_size', type=int, default=12,
                       help='Size of the board (default: 12)')
    parser.add_argument('--bound', type=int, default=5,
                       help='Number of pieces in a row to win (default: 5)')

    args = parser.parse_args()

    run(board_size=args.board_size, bound=args.bound, model_path=args.model_path)


if __name__ == "__main__":
    main()