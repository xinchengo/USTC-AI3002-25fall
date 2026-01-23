#!/usr/bin/env python3
"""
Test script to evaluate the trained model against a random player
"""

import torch
import numpy as np
from utils import Gobang, device
from submission import GobangModel
import json
import os

def get_random_model(board_size=12, bound=5):
    """Create a random model for baseline comparison"""
    # Create a simple random model that generates uniform probabilities
    model = GobangModel(board_size=board_size, bound=bound, model_type="cnn", 
                        extra_specs={"depth": 1, "channels": 16})
    
    # Initialize with random weights to simulate a random policy
    for param in model.parameters():
        torch.nn.init.uniform_(param, -0.1, 0.1)
        
    model.eval()
    return model

def evaluate_trained_vs_random(trained_model_path, episodes=100):
    """Evaluate trained model vs random model"""
    # Load the trained model
    trained_specs = {
        "depth": 3,
        "channels": 64,
        "batch_norm": False,
        "action_injection": "late"
    }
    
    trained_model = GobangModel(
        board_size=12, 
        bound=5, 
        model_type="cnn", 
        extra_specs=trained_specs
    )
    
    # Load the trained weights
    state_dict = torch.load(trained_model_path, map_location=device)
    trained_model.load_state_dict(state_dict)
    trained_model.eval()
    
    # Create a random model
    random_model = get_random_model()
    
    # Set up the game environment
    chess_board = Gobang(board_size=12, bound=5, training=False)
    
    # Evaluate
    black_wins, white_wins, ties = 0, 0, 0
    
    print(f"Starting evaluation: Trained model (black) vs Random model (white) for {episodes} episodes")
    
    for episode in range(episodes):
        chess_board.restart()
        
        while True:
            color, end_up_gaming = chess_board.update_board(
                learning=False, 
                random_response=False  # Use the model instead of random
            )
            
            if end_up_gaming:
                if color == 1:  # Black wins
                    black_wins += 1
                elif color == 2:  # White wins
                    white_wins += 1
                else:  # Tie
                    ties += 1
                break
                
        if (episode + 1) % 20 == 0:
            print(f"Completed {episode + 1}/{episodes} episodes - "
                  f"Black: {black_wins}, White: {white_wins}, Ties: {ties}")
    
    print(f"\nEvaluation Results:")
    print(f"Black (Trained) wins: {black_wins} ({black_wins/episodes*100:.2f}%)")
    print(f"White (Random) wins: {white_wins} ({white_wins/episodes*100:.2f}%)")
    print(f"Ties: {ties} ({ties/episodes*100:.2f}%)")
    
    return black_wins, white_wins, ties

if __name__ == "__main__":
    # Path to the trained model
    model_path = "checkpoints/cnn_gobang_model_20260123-152733/final_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        # Try to find the most recent model
        import glob
        model_files = glob.glob("checkpoints/cnn_gobang_model_*/*/final_model.pth")
        if model_files:
            model_path = sorted(model_files)[-1]  # Get the most recent
            print(f"Using most recent model: {model_path}")
        else:
            print("No model files found!")
            exit(1)
    
    # Run evaluation
    black_wins, white_wins, ties = evaluate_trained_vs_random(model_path, episodes=50)