from utils import *
from model_loader import get_model
from opponent_loader import get_opponent

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to the trained model for black pieces.')
    parser.add_argument('--opponent_path', type=str, default='opponent.pth', help='Path to the opponent model for white pieces.')
    args = parser.parse_args()
    
    # Define gaming settings.
    board_size = 12
    bound = 5
    num_episodes = 1000

    # Load trained models for black pieces and opponents for white pieces.
    model = get_model()
    opponent = get_opponent()

    model.eval()
    opponent.eval()

    # Start evaluation process.
    chess_board = Gobang(board_size=board_size, bound=bound, training=False)

    # Start testing with random noise (by setting random_response=True),
    # or testing with another trained model (by setting random_response=False).
    # Make sure that both the model (which represents black pieces) and opponent (which represents white pieces) are
    # loaded before the evaluation process.
    chess_board.evaluate_agent_performance(random_response=False, model=model, opponent=opponent, episodes=num_episodes)
