# Gobang AI Project - QWEN Context

## Configs

- When encountering `[TODO(AGENT)]`, no matter what you are doing, add this into your plan, resolve it, then remove this label. The `[LINKUPDATE]` flag indicates the other affected files of this todo.

## Project Overview

This is a Gobang (Five in a Row) AI project that implements an Actor-Critic reinforcement learning approach to develop an agent capable of playing on a 12×12 board with a win condition of 5 in a row. The project addresses the challenge of scaling traditional Q-learning to larger state/action spaces by using deep learning techniques.

### Main Technologies
- Python 3.12+
- PyTorch for neural networks
- NumPy for numerical computations
- Tkinter for visualization
- Weights & Biases (wandb) for experiment tracking
- Matplotlib for loss visualization

### Architecture
- **Actor-Critic Framework**: Combines policy optimization (Actor) with value estimation (Critic)
- **CNN-based Models**: Convolutional neural networks for spatial pattern recognition on the board
- **Extensible Wrapper System**: Modular architecture for different player types (checkpoint, random, baseline)
- **Self-Play Training**: Agent learns by playing against itself

## Building and Running

### Prerequisites
- Python 3.12+
- Virtual environment with dependencies installed via `uv pip`

### Setup
```bash
# Activate the virtual environment
source .venv/bin/activate

# Install dependencies if needed
uv pip install scipy  # Additional dependency for statistics calculations
```

### Training Models
```bash
# Train regular CNN model for 1000 episodes with checkpoint every 200 episodes
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb

# Train deep CNN model for 1000 episodes with checkpoint every 200 episodes
python submission.py --num_episodes 1000 --checkpoint 200 --use_deep --use_wandb
```

### Evaluating Models
```bash
# Evaluate two models against each other
python -m tests.evaluator --player1_path checkpoints/model_999.pth --player1_type checkpoint --player2_path checkpoints/model_799.pth --player2_type checkpoint --episodes 500

# Evaluate against random player
python -m tests.evaluator --player1_path checkpoints/model_999.pth --player1_type checkpoint --player2_type random --episodes 100
```

### Visualization
```bash
# Visualize gameplay
python player.py
```

### Statistics Calculation
```bash
# Calculate statistical results for experiments
python calculate_statistics.py
```

## Development Conventions

### Code Structure
- **submission.py**: Main model implementation with Actor-Critic architecture
- **utils.py**: Game mechanics, training loop, and utility functions
- **wrappers/**: Extensible wrapper classes for different player types
- **tests/**: Evaluation and testing scripts
- **checkpoints/**: Saved model checkpoints
- **wandb/**: Weights & Biases experiment logs

### Model Architecture
- **Actor**: Outputs policy distribution over possible actions
- **Critic**: Estimates Q-values for state-action pairs
- **GobangModel**: Integrates Actor and Critic components

### Training Process
- Self-play training where the agent plays against itself
- Legal move enforcement through masking invalid positions
- Reward shaping based on connection lengths
- Periodic checkpoint saving

### Evaluation Methodology
- Fair evaluation by alternating who plays first (black/white)
- Statistical significance testing for performance comparisons
- Multiple evaluation metrics (win rates, tie rates)

## Key Files and Components

### Core Implementation
- `submission.py`: Main model definitions (Actor, Critic, GobangModel)
- `utils.py`: Game mechanics, training loop, helper functions
- `model_loader.py` / `opponent_loader.py`: Model loading utilities

### Wrappers System
- `wrappers/base.py`: Abstract base class for all wrappers
- `wrappers/checkpoint.py`: Wrapper for loading trained models
- `wrappers/random.py`: Wrapper for random policy
- `wrappers/baseline.py`: Wrapper for baseline policy
- `wrappers/factory.py`: Factory for creating wrapper instances

### Evaluation System
- `tests/evaluator.py`: General evaluator for any pair of players
- `tests/human_play.py`: Interface for human vs AI gameplay

### Experimental Scripts
- `calculate_statistics.py`: Statistical calculations for experiment results
- `log.md`: Experimental plan and results documentation
- `structure.md`: Project structure documentation

## Project Goals

1. **Scale Traditional RL**: Move beyond basic Q-learning to handle larger board sizes
2. **Deep Learning Integration**: Use CNNs to achieve generalization on unknown states
3. **Legal Move Enforcement**: Ensure all moves are valid through policy masking
4. **Efficient Response Generation**: Use same model for both black and white moves
5. **Extensible Architecture**: Support different player types through wrapper system
6. **Rigorous Evaluation**: Statistical testing to validate improvements

## Experimental Results

The project includes comprehensive evaluation comparing:
- Regular vs deep CNN architectures
- Model convergence testing (800 vs 1000 epoch models)
- Performance against random and baseline policies
- Statistical significance of improvements

Statistical calculations are performed by the `calculate_statistics.py` script to ensure reproducibility and accuracy.