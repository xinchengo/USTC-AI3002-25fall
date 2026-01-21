# Gobang Project Structure

This document describes the structure of the enhanced Gobang project with extensible evaluation capabilities.

## Directory Structure

```
gobang/
├── submission.py          # Main model implementation with Actor-Critic architecture
├── utils.py              # Utility functions and game mechanics
├── model_loader.py       # Loading trained models for black pieces
├── opponent_loader.py    # Loading opponent models for white pieces
├── player.py             # Visualization of playing process
├── evaluator.py          # Original evaluator (kept for compatibility)
├── readme.md             # Project overview and methodology
├── log.md                # Training logs
├── loss_tracker.png      # Generated loss visualization
├── checkpoints/          # Saved model checkpoints
├── wandb/               # Weights & Biases logs
├── wrappers/            # Wrapper classes for different player types
│   ├── __init__.py
│   ├── checkpoint.py    # Wrapper for loading checkpointed models
│   └── random.py        # Wrapper for random policy
└── tests/               # Testing and evaluation scripts
    ├── __init__.py
    ├── human_play.py    # Human vs AI gameplay interface
    └── evaluator.py     # General evaluator for any pair of players
```

## Key Components

### Core Files
- **submission.py**: Implements the GobangModel with Actor-Critic architecture. Now exports both .pth (state dict) and .pkl (complete model) files.
- **utils.py**: Contains game mechanics, training loop, and utility functions. Updated to save both .pth and .pkl checkpoint files.
- **model_loader.py** / **opponent_loader.py**: Functions to load trained models for gameplay.

### Wrappers
Extensible wrapper system for different player types:

- **BaseWrapper**: Abstract base class defining the common interface for all wrappers
- **CheckpointWrapper**: Loads and interfaces with trained models (both .pth and .pkl formats)
- **RandomWrapper**: Implements a random policy for baseline comparison
- **BaselineWrapper**: Implements a simple baseline policy (e.g., alpha-beta pruning or heuristic-based)

### Tests
Enhanced evaluation system:

- **human_play.py**: Text-based interface for human vs AI gameplay (following AlphaZero_Gomoku format)
- **evaluator.py**: General evaluator supporting any pair of player types

## Extensibility

The new architecture supports easy addition of new player types:

1. Create a new wrapper in `wrappers/` directory
2. Implement the required interface (get_action, get_policy methods)
3. Use in evaluators without code changes

Future additions planned:
- `wrappers/baseline.py` for alpha-beta pruning baseline
- `baselines/` directory for various baseline implementations

## Usage Examples

### Training (now saves both .pth and .pkl)
```bash
python submission.py --num_episodes 1000 --checkpoint 500 --use_deep
```

### Evaluation between any two players
```bash
python tests/evaluator.py --player1_path checkpoints/model_999.pkl --player1_type checkpoint --player2_type random --episodes 100
```

### Human vs AI gameplay
```bash
python tests/human_play.py --model_path checkpoints/final_model.pkl --ai_player 1
```