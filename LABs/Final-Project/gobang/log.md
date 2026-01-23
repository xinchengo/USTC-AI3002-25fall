# Plan

Plan for the gobang Actor-Critic Reinforcement Learning task

## Guide for AI agents

- The Python environment is a `uv` virtual environment located within `USTC-AI3002-25fall` (the repository root);
- You should `source .venv/bin/activate` before running `python`
- REMEMBER TO Use `uv pip` for package management
- scripts in `tests/` folder are intended to be called with `python -m tests.<...> <...>` inside the `gobang` directory
- You should follow my plan to automatically execute scripts for my experiment
- Numerical calculations should be done by executing `calculate_statistics.py`, you should extend this file if necessary.

For this file, complete this file in a consistent, rigorous format.

For every experiments, reread the following lines:

- Every experiment, three parts: steps, script, and results
- Fill in the blanks, and the format should follow the contents and consistent with other already-filled-in experiment
- Numerical calculation are done by modifying and executing `calculate_statistics.py`
- Include all WanDB links for easier introspection

## Experimental Plan Overview

We aim to explore the connection between the model performance and:

- **Model depth**: Comprehensive depth exploration (depths 2, 3, 4, 5, 6) with width=64
- **Model width**: Comprehensive width exploration (channels 16, 32, 64, 128, 256) at various depths
- **Batch normalization**: With vs without batch normalization
- **Action injection in critic**: Early (input channel) vs Late (before last conv layer) vs FC (before FC layer) vs None
- **Learning rate**: Impact of different learning rates (1e-5, 5e-5, 1e-4, 5e-4, 1e-3) on convergence
- **Discount factor (gamma)**: Impact of different discount factors (0.9, 0.95, 0.99, 0.999) on long-term value estimation

**Notes**:
- We use the `cnn` model type exclusively (not `default` which is for backward compatibility)
- Baseline/unspecified uses `late` injection (positioned before the last conv layer)
- Late injection is a reference architecture for future work on shared backbone/separate head models
- Larger models use more training epochs to ensure convergence
- Comprehensive depth and width exploration to understand scaling properties

We employ multiple metrics to evaluate the model performance:

- **Competition**: Models compete against each other in tournaments
- **Convergence**: Speed of convergence measured by comparing checkpoints
- **Scalability**: How performance scales with depth and width

---

## Part 1: Baseline Experiments

### 1.1 Baseline Model Training (CNN: width=64, depth=3 and 5)

**Objective**: Train baseline CNN models with `cnn` model type and establish parameter counts.

**Steps**:
1. Train baseline CNN (depth=3, width=64) for 1000 episodes with `late` injection
2. Train deep CNN (depth=5, width=64) for 1500 episodes with `late` injection (more epochs for larger model)
3. Record model parameter counts
4. Save checkpoints every 200 episodes

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

# Train baseline CNN (depth=3, width=64, late injection - baseline)
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "baseline-cnn-d3-w64-late" --model-type cnn --extra-specs '{"depth": 3, "channels": 64, "batch_norm": false, "action_injection": "late"}'

# Train deep CNN (depth=5, width=64, late injection)
python submission.py --num_episodes 1500 --checkpoint 200 --use_wandb --wandb_name "deep-cnn-d5-w64-late" --model-type cnn --extra-specs '{"depth": 5, "channels": 64, "batch_norm": false, "action_injection": "late"}'
```

**Results**:
- Baseline CNN (d=3, w=64) checkpoint folder: `<TO BE FILLED IN>`
- Deep CNN (d=5, w=64) checkpoint folder: `<TO BE FILLED IN>`
- Baseline CNN parameter count: `<TO BE FILLED IN>`
- Deep CNN parameter count: `<TO BE FILLED IN>`
- Baseline CNN WandB link: `<TO BE FILLED IN>`
- Deep CNN WandB link: `<TO BE FILLED IN>`

---

### 1.2 Convergence Test: Baseline CNN (d=3, w=64)

**Objective**: Test if baseline CNN has converged by comparing epoch 1000 vs epoch 800.

**Steps**:
1. Evaluate model_999.pth vs model_799.pth over 500 episodes
2. Calculate winning rate of epoch 1000 model
3. Perform hypothesis test: $H_0: p=0.5$ vs $H_1: p \ne 0.5$ at $\alpha=0.01$

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

python -m tests.evaluator --player1_path <BASELINE_CHECKPOINT_FOLDER>/model_999.pth --player1_type checkpoint --player2_path <BASELINE_CHECKPOINT_FOLDER>/model_799.pth --player2_type checkpoint --episodes 500

python calculate_statistics.py  # Calculate p-value for convergence test
```

**Results**:
$$
\begin{gather}
H_0:p=0.5 \leftrightarrow H_1:p\ne 0.5\\
X=\sum_{i=1}^{500} X_i = \text{<TO BE FILLED IN>}\\
\text{p-value} = \text{<TO BE FILLED IN>}\\
\text{Conclusion: <TO BE FILLED IN>}
\end{gather}
$$

---

### 1.3 Convergence Test: Deep CNN (d=5, w=64)

**Objective**: Test if deep CNN has converged by comparing epoch 1500 vs epoch 1300.

**Steps**:
1. Evaluate model_1499.pth vs model_1299.pth over 500 episodes
2. Calculate winning rate of epoch 1500 model
3. Perform hypothesis test: $H_0: p=0.5$ vs $H_1: p \ne 0.5$ at $\alpha=0.01$

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

python -m tests.evaluator --player1_path <DEEP_CHECKPOINT_FOLDER>/model_1499.pth --player1_type checkpoint --player2_path <DEEP_CHECKPOINT_FOLDER>/model_1299.pth --player2_type checkpoint --episodes 500

python calculate_statistics.py  # Calculate p-value for convergence test
```

**Results**:
$$
\begin{gather}
H_0:p=0.5 \leftrightarrow H_1:p\ne 0.5\\
X=\sum_{i=1}^{500} X_i = \text{<TO BE FILLED IN>}\\
\text{p-value} = \text{<TO BE FILLED IN>}\\
\text{Conclusion: <TO BE FILLED IN>}
\end{gather}
$$

---

### 1.4 Performance vs Random Policy

**Objective**: Test if baseline CNN significantly outperforms random policy.

**Steps**:
1. Evaluate baseline CNN vs random policy over 500 episodes
2. Perform hypothesis test: $H_0: p=0.5$ vs $H_1: p > 0.5$ at $\alpha=0.01$ (one-tailed)

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

python -m tests.evaluator --player1_path <BASELINE_CHECKPOINT_FOLDER>/model_999.pth --player1_type checkpoint --player2_path random --player2_type baseline --episodes 500

python calculate_statistics.py  # Calculate p-value
```

**Results**:
$$
\begin{gather}
H_0:p=0.5 \leftrightarrow H_1:p > 0.5\\
\text{Baseline CNN wins: } \text{<TO BE FILLED IN>}\\
\text{p-value} = \text{<TO BE FILLED IN>}\\
\text{Conclusion: <TO BE FILLED IN>}
\end{gather}
$$

---

### 1.5 Depth Comparison: Baseline (d=3) vs Deep (d=5)

**Objective**: Test if deep CNN significantly outperforms baseline CNN.

**Steps**:
1. Evaluate baseline CNN (d=3, epoch 999) vs deep CNN (d=5, epoch 1499) over 500 episodes
2. Perform two-proportion Z-test at $\alpha=0.01$

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

python -m tests.evaluator --player1_path <BASELINE_CHECKPOINT_FOLDER>/model_999.pth --player1_type checkpoint --player2_path <DEEP_CHECKPOINT_FOLDER>/model_1499.pth --player2_type checkpoint --episodes 500

python calculate_statistics.py  # Calculate Z-test
```

**Results**:
$$
\begin{gather}
H_0:p_{\text{baseline}}=p_{\text{deep}} \leftrightarrow H_1:p_{\text{baseline}} \ne p_{\text{deep}}\\
\hat p_{\text{baseline}} = \text{<TO BE FILLED IN>}, \quad \hat p_{\text{deep}} = \text{<TO BE FILLED IN>}\\
Z = \text{<TO BE FILLED IN>}\\
\text{p-value} = \text{<TO BE FILLED IN>}\\
\text{Conclusion: <TO BE FILLED IN>}
\end{gather}
$$

---

## Part 2: Comprehensive Depth Exploration

**Objective**: Test models with different depths (2, 3, 4, 5, 6) at fixed width=64 to understand depth scaling.

### 2.1 Depth Series Training (width=64, late injection)

**Steps**:
1. Train CNN with depth=2 for 1000 episodes
2. Train CNN with depth=3 for 1000 episodes (already trained as baseline, reuse)
3. Train CNN with depth=4 for 1200 episodes
4. Train CNN with depth=5 for 1500 episodes (already trained, reuse)
5. Train CNN with depth=6 for 1800 episodes
6. All models use width=64, no batch norm, late injection

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

# Depth=2 (shallow)
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "depth-d2-w64-late" --model-type cnn --extra-specs '{"depth": 2, "channels": 64, "batch_norm": false, "action_injection": "late"}'

# Depth=3 (baseline - reuse from Part 1)

# Depth=4 (medium-deep)
python submission.py --num_episodes 1200 --checkpoint 200 --use_wandb --wandb_name "depth-d4-w64-late" --model-type cnn --extra-specs '{"depth": 4, "channels": 64, "batch_norm": false, "action_injection": "late"}'

# Depth=5 (deep - reuse from Part 1)

# Depth=6 (very deep)
python submission.py --num_episodes 1800 --checkpoint 200 --use_wandb --wandb_name "depth-d6-w64-late" --model-type cnn --extra-specs '{"depth": 6, "channels": 64, "batch_norm": false, "action_injection": "late"}'
```

**Results**:
| Depth | Episodes | Checkpoint | Params | WandB Link |
|-------|----------|-----------|--------|-----------|
| 2 | 1000 | `<FILL>` | `<FILL>` | `<FILL>` |
| 3 | 1000 | `<BASELINE_CHECKPOINT>` | `<FILL>` | `<FILL>` |
| 4 | 1200 | `<FILL>` | `<FILL>` | `<FILL>` |
| 5 | 1500 | `<DEEP_CHECKPOINT>` | `<FILL>` | `<FILL>` |
| 6 | 1800 | `<FILL>` | `<FILL>` | `<FILL>` |

---

### 2.2 Depth Scaling Tournament

**Objective**: Compare all depth variants in round-robin tournament to analyze scaling effects.

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

# d=2 vs d=3
python -m tests.evaluator --player1_path <D2_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <BASELINE_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

# d=3 vs d=4
python -m tests.evaluator --player1_path <BASELINE_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <D4_CHECKPOINT>/model_1199.pth --player2_type checkpoint --episodes 500

# d=4 vs d=5
python -m tests.evaluator --player1_path <D4_CHECKPOINT>/model_1199.pth --player1_type checkpoint --player2_path <DEEP_CHECKPOINT>/model_1499.pth --player2_type checkpoint --episodes 500

# d=5 vs d=6
python -m tests.evaluator --player1_path <DEEP_CHECKPOINT>/model_1499.pth --player1_type checkpoint --player2_path <D6_CHECKPOINT>/model_1799.pth --player2_type checkpoint --episodes 500

# d=2 vs d=5 (extreme comparison)
python -m tests.evaluator --player1_path <D2_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <DEEP_CHECKPOINT>/model_1499.pth --player2_type checkpoint --episodes 500

# d=2 vs d=6 (extreme comparison)
python -m tests.evaluator --player1_path <D2_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <D6_CHECKPOINT>/model_1799.pth --player2_type checkpoint --episodes 500

python calculate_statistics.py  # Calculate tournament statistics
```

**Results**:
| Matchup | Player 1 Win Rate | Player 2 Win Rate | p-value | Winner |
|---------|-------------------|-------------------|---------|--------|
| d=2 vs d=3 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| d=3 vs d=4 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| d=4 vs d=5 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| d=5 vs d=6 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| d=2 vs d=5 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| d=2 vs d=6 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |

**Depth Scaling Analysis**:
- Optimal depth: `<TO BE FILLED IN>`
- Performance trend: `<TO BE FILLED IN>` (improving/degrading/plateauing)
- Parameter scaling vs performance: `<TO BE FILLED IN>`

---

## Part 3: Comprehensive Width Exploration

**Objective**: Test models with different widths (16, 32, 64, 128, 256) at fixed depth=3 and depth=5.

### 3.1 Width Series Training - Shallow (depth=3, late injection)

**Steps**:
1. Train CNN with channels=16 for 1000 episodes
2. Train CNN with channels=32 for 1000 episodes
3. Train CNN with channels=64 for 1000 episodes (baseline, reuse)
4. Train CNN with channels=128 for 1000 episodes
5. Train CNN with channels=256 for 1000 episodes

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

# Width=16 (very narrow)
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "width-d3-ch16-late" --model-type cnn --extra-specs '{"depth": 3, "channels": 16, "batch_norm": false, "action_injection": "late"}'

# Width=32 (narrow)
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "width-d3-ch32-late" --model-type cnn --extra-specs '{"depth": 3, "channels": 32, "batch_norm": false, "action_injection": "late"}'

# Width=64 (medium - reuse baseline)

# Width=128 (wide)
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "width-d3-ch128-late" --model-type cnn --extra-specs '{"depth": 3, "channels": 128, "batch_norm": false, "action_injection": "late"}'

# Width=256 (very wide)
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "width-d3-ch256-late" --model-type cnn --extra-specs '{"depth": 3, "channels": 256, "batch_norm": false, "action_injection": "late"}'
```

**Results**:
| Channels | Episodes | Checkpoint | Params | WandB Link |
|----------|----------|-----------|--------|-----------|
| 16 | 1000 | `<FILL>` | `<FILL>` | `<FILL>` |
| 32 | 1000 | `<FILL>` | `<FILL>` | `<FILL>` |
| 64 | 1000 | `<BASELINE_CHECKPOINT>` | `<FILL>` | `<FILL>` |
| 128 | 1000 | `<FILL>` | `<FILL>` | `<FILL>` |
| 256 | 1000 | `<FILL>` | `<FILL>` | `<FILL>` |

---

### 3.2 Width Series Training - Deep (depth=5, late injection)

**Steps**:
1. Train CNN with channels=16 for 1500 episodes
2. Train CNN with channels=32 for 1500 episodes
3. Train CNN with channels=64 for 1500 episodes (deep baseline, reuse)
4. Train CNN with channels=128 for 1500 episodes
5. Train CNN with channels=256 for 1500 episodes

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

# Width=16 (very narrow)
python submission.py --num_episodes 1500 --checkpoint 200 --use_wandb --wandb_name "width-d5-ch16-late" --model-type cnn --extra-specs '{"depth": 5, "channels": 16, "batch_norm": false, "action_injection": "late"}'

# Width=32 (narrow)
python submission.py --num_episodes 1500 --checkpoint 200 --use_wandb --wandb_name "width-d5-ch32-late" --model-type cnn --extra-specs '{"depth": 5, "channels": 32, "batch_norm": false, "action_injection": "late"}'

# Width=64 (medium - reuse deep baseline)

# Width=128 (wide)
python submission.py --num_episodes 1500 --checkpoint 200 --use_wandb --wandb_name "width-d5-ch128-late" --model-type cnn --extra-specs '{"depth": 5, "channels": 128, "batch_norm": false, "action_injection": "late"}'

# Width=256 (very wide)
python submission.py --num_episodes 1500 --checkpoint 200 --use_wandb --wandb_name "width-d5-ch256-late" --model-type cnn --extra-specs '{"depth": 5, "channels": 256, "batch_norm": false, "action_injection": "late"}'
```

**Results**:
| Channels | Episodes | Checkpoint | Params | WandB Link |
|----------|----------|-----------|--------|-----------|
| 16 | 1500 | `<FILL>` | `<FILL>` | `<FILL>` |
| 32 | 1500 | `<FILL>` | `<FILL>` | `<FILL>` |
| 64 | 1500 | `<DEEP_CHECKPOINT>` | `<FILL>` | `<FILL>` |
| 128 | 1500 | `<FILL>` | `<FILL>` | `<FILL>` |
| 256 | 1500 | `<FILL>` | `<FILL>` | `<FILL>` |

---

### 3.3 Width Scaling Tournament - Shallow (depth=3)

**Objective**: Compare all width variants at depth=3 to analyze width scaling.

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

# ch=16 vs ch=32
python -m tests.evaluator --player1_path <CH16_D3_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <CH32_D3_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

# ch=32 vs ch=64
python -m tests.evaluator --player1_path <CH32_D3_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <BASELINE_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

# ch=64 vs ch=128
python -m tests.evaluator --player1_path <BASELINE_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <CH128_D3_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

# ch=128 vs ch=256
python -m tests.evaluator --player1_path <CH128_D3_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <CH256_D3_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

# ch=16 vs ch=256 (extreme)
python -m tests.evaluator --player1_path <CH16_D3_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <CH256_D3_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

python calculate_statistics.py
```

**Results**:
| Matchup | Player 1 Win Rate | Player 2 Win Rate | p-value | Winner |
|---------|-------------------|-------------------|---------|--------|
| ch=16 vs ch=32 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| ch=32 vs ch=64 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| ch=64 vs ch=128 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| ch=128 vs ch=256 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| ch=16 vs ch=256 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |

---

### 3.4 Width Scaling Tournament - Deep (depth=5)

**Objective**: Compare all width variants at depth=5 to analyze width scaling in deeper networks.

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

# ch=16 vs ch=32
python -m tests.evaluator --player1_path <CH16_D5_CHECKPOINT>/model_1499.pth --player1_type checkpoint --player2_path <CH32_D5_CHECKPOINT>/model_1499.pth --player2_type checkpoint --episodes 500

# ch=32 vs ch=64
python -m tests.evaluator --player1_path <CH32_D5_CHECKPOINT>/model_1499.pth --player1_type checkpoint --player2_path <DEEP_CHECKPOINT>/model_1499.pth --player2_type checkpoint --episodes 500

# ch=64 vs ch=128
python -m tests.evaluator --player1_path <DEEP_CHECKPOINT>/model_1499.pth --player1_type checkpoint --player2_path <CH128_D5_CHECKPOINT>/model_1499.pth --player2_type checkpoint --episodes 500

# ch=128 vs ch=256
python -m tests.evaluator --player1_path <CH128_D5_CHECKPOINT>/model_1499.pth --player1_type checkpoint --player2_path <CH256_D5_CHECKPOINT>/model_1499.pth --player2_type checkpoint --episodes 500

# ch=16 vs ch=256 (extreme)
python -m tests.evaluator --player1_path <CH16_D5_CHECKPOINT>/model_1499.pth --player1_type checkpoint --player2_path <CH256_D5_CHECKPOINT>/model_1499.pth --player2_type checkpoint --episodes 500

python calculate_statistics.py
```

**Results**:
| Matchup | Player 1 Win Rate | Player 2 Win Rate | p-value | Winner |
|---------|-------------------|-------------------|---------|--------|
| ch=16 vs ch=32 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| ch=32 vs ch=64 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| ch=64 vs ch=128 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| ch=128 vs ch=256 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| ch=16 vs ch=256 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |

**Width Scaling Analysis**:
- Optimal width at depth=3: `<TO BE FILLED IN>`
- Optimal width at depth=5: `<TO BE FILLED IN>`
- Width scaling trend: `<TO BE FILLED IN>` (improving/plateauing/diminishing returns)

---

## Part 4: Batch Normalization

### 4.1 Batch Normalization Training

**Objective**: Test effect of batch normalization on convergence and performance.

**Steps**:
1. Train CNN with batch norm (depth=3, channels=64, late injection) for 1000 episodes
2. Compare with baseline (without batch norm) from Part 1.1

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

# With batch normalization (d=3, ch=64, late injection)
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "bn-d3-ch64-late" --model-type cnn --extra-specs '{"depth": 3, "channels": 64, "batch_norm": true, "action_injection": "late"}'
```

**Results**:
- BatchNorm checkpoint: `<TO BE FILLED IN>`
- BatchNorm params: `<TO BE FILLED IN>`
- WandB link: `<TO BE FILLED IN>`

---

### 4.2 Batch Normalization Performance Test

**Objective**: Compare batch norm vs no batch norm at depth=3, width=64.

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

python -m tests.evaluator --player1_path <BATCHNORM_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <BASELINE_CHECKPOINT_FOLDER>/model_999.pth --player2_type checkpoint --episodes 500

python calculate_statistics.py  # Statistical test
```

**Results**:
$$
\begin{gather}
H_0:p_{\text{bn}}=p_{\text{no-bn}} \leftrightarrow H_1:p_{\text{bn}} \ne p_{\text{no-bn}}\\
\text{BatchNorm win rate: } \text{<TO BE FILLED IN>}\\
\text{No-BatchNorm win rate: } \text{<TO BE FILLED IN>}\\
Z = \text{<TO BE FILLED IN>}\\
\text{p-value} = \text{<TO BE FILLED IN>}\\
\text{Conclusion: <TO BE FILLED IN>}
\end{gather}
$$

---

## Part 5: Action Injection Methods

### 5.1 Training with Different Action Injections

**Objective**: Test different action injection strategies in the Critic (baseline uses late injection, positioned before last conv layer).

**Steps**:
1. Train with no injection (standard approach)
2. Train with early injection (extra input channel)
3. Train with late injection before last conv layer (baseline reference)
4. Train with FC injection (before fully connected layer)
5. All use depth=3, channels=64, no batch norm
6. Train for 1000 episodes

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

# No injection
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "action-inject-none" --model-type cnn --extra-specs '{"depth": 3, "channels": 64, "batch_norm": false, "action_injection": "none"}'

# Early injection
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "action-inject-early" --model-type cnn --extra-specs '{"depth": 3, "channels": 64, "batch_norm": false, "action_injection": "early"}'

# Late injection (baseline reference - reuse from Part 1.1)

# FC injection
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "action-inject-fc" --model-type cnn --extra-specs '{"depth": 3, "channels": 64, "batch_norm": false, "action_injection": "fc"}'
```

**Results**:
- No injection: checkpoint=`<TO BE FILLED IN>`, params=`<FILL>`, WandB=`<TO BE FILLED IN>`
- Early injection: checkpoint=`<TO BE FILLED IN>`, params=`<FILL>`, WandB=`<TO BE FILLED IN>`
- Late injection (before last conv): checkpoint=`<BASELINE_CHECKPOINT_FOLDER>`, params=`<FILL>`, WandB=`<TO BE FILLED IN>` (baseline reference)
- FC injection: checkpoint=`<TO BE FILLED IN>`, params=`<FILL>`, WandB=`<TO BE FILLED IN>`

---

### 5.2 Action Injection Tournament

**Objective**: Compare all action injection methods in round-robin tournament.

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

# None vs Early
python -m tests.evaluator --player1_path <NONE_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <EARLY_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

# None vs Late
python -m tests.evaluator --player1_path <NONE_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <BASELINE_CHECKPOINT_FOLDER>/model_999.pth --player2_type checkpoint --episodes 500

# None vs FC
python -m tests.evaluator --player1_path <NONE_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <FC_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

# Early vs Late
python -m tests.evaluator --player1_path <EARLY_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <BASELINE_CHECKPOINT_FOLDER>/model_999.pth --player2_type checkpoint --episodes 500

# Early vs FC
python -m tests.evaluator --player1_path <EARLY_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <FC_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

# Late vs FC
python -m tests.evaluator --player1_path <BASELINE_CHECKPOINT_FOLDER>/model_999.pth --player1_type checkpoint --player2_path <FC_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

python calculate_statistics.py  # Tournament statistics
```

**Results**:
| Matchup | Player 1 Win Rate | Player 2 Win Rate | p-value | Winner |
|---------|-------------------|-------------------|---------|--------|
| None vs Early | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| None vs Late | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| None vs FC | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| Early vs Late | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| Early vs FC | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| Late vs FC | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |

**Conclusion**: `<TO BE FILLED IN>`

---

## Part 6: Learning Rate Exploration

**Objective**: Test models with different learning rates (1e-5, 5e-5, 1e-4, 5e-4, 1e-3) to understand impact on convergence and final performance.

### 6.1 Learning Rate Series Training

**Steps**:
1. Train CNN with learning rate=1e-5 for 1000 episodes
2. Train CNN with learning rate=5e-5 for 1000 episodes
3. Train CNN with learning rate=1e-4 for 1000 episodes (baseline reference, similar to default)
4. Train CNN with learning rate=5e-4 for 1000 episodes
5. Train CNN with learning rate=1e-3 for 1000 episodes
6. All use depth=3, channels=64, no batch norm, late injection

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

# Learning rate=1e-5 (very low)
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "lr-1e-5-d3-ch64-late" --model-type cnn --extra-specs '{"depth": 3, "channels": 64, "batch_norm": false, "action_injection": "late"}' --lr 1e-5

# Learning rate=5e-5 (low)
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "lr-5e-5-d3-ch64-late" --model-type cnn --extra-specs '{"depth": 3, "channels": 64, "batch_norm": false, "action_injection": "late"}' --lr 5e-5

# Learning rate=1e-4 (baseline reference - reuse from Part 1.1)

# Learning rate=5e-4 (high)
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "lr-5e-4-d3-ch64-late" --model-type cnn --extra-specs '{"depth": 3, "channels": 64, "batch_norm": false, "action_injection": "late"}' --lr 5e-4

# Learning rate=1e-3 (very high)
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "lr-1e-3-d3-ch64-late" --model-type cnn --extra-specs '{"depth": 3, "channels": 64, "batch_norm": false, "action_injection": "late"}' --lr 1e-3
```

**Results**:
| Learning Rate | Episodes | Checkpoint | Params | WandB Link | Convergence Quality |
|---|----------|-----------|--------|-----------|-------------|
| 1e-5 | 1000 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 5e-5 | 1000 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 1e-4 | 1000 | `<BASELINE_CHECKPOINT>` | `<FILL>` | `<FILL>` | `<FILL>` (baseline) |
| 5e-4 | 1000 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 1e-3 | 1000 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |

---

### 6.2 Learning Rate Performance Tournament

**Objective**: Compare models trained with different learning rates to identify optimal learning rate.

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

# 1e-5 vs 5e-5
python -m tests.evaluator --player1_path <LR1E5_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <LR5E5_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

# 5e-5 vs 1e-4
python -m tests.evaluator --player1_path <LR5E5_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <BASELINE_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

# 1e-4 vs 5e-4
python -m tests.evaluator --player1_path <BASELINE_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <LR5E4_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

# 5e-4 vs 1e-3
python -m tests.evaluator --player1_path <LR5E4_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <LR1E3_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

# 1e-5 vs 1e-3 (extreme comparison)
python -m tests.evaluator --player1_path <LR1E5_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <LR1E3_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

python calculate_statistics.py  # Tournament statistics
```

**Results**:
| Matchup | Player 1 Win Rate | Player 2 Win Rate | p-value | Winner |
|---------|-------------------|-------------------|---------|--------|
| 1e-5 vs 5e-5 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 5e-5 vs 1e-4 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 1e-4 vs 5e-4 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 5e-4 vs 1e-3 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 1e-5 vs 1e-3 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |

**Learning Rate Scaling Analysis**:
- Optimal learning rate: `<TO BE FILLED IN>`
- Convergence trend: `<TO BE FILLED IN>` (faster/slower as LR increases)
- Training stability: `<TO BE FILLED IN>` (affected by learning rate changes)

---

## Part 7: Discount Factor (Gamma) Exploration

**Objective**: Test models with different discount factors (0.9, 0.95, 0.99, 0.999) to understand impact on long-term value estimation and performance.

### 7.1 Discount Factor Series Training

**Steps**:
1. Train CNN with discount factor=0.9 for 1000 episodes
2. Train CNN with discount factor=0.95 for 1000 episodes
3. Train CNN with discount factor=0.99 for 1000 episodes (optional; baseline gamma=0.5 is reused from Part 1)
4. Train CNN with discount factor=0.999 for 1000 episodes
5. All use depth=3, channels=64, no batch norm, late injection, learning rate=1e-4

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

# Discount factor=0.9 (short-term oriented)
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "gamma-0.9-d3-ch64-late" --model-type cnn --extra-specs '{"depth": 3, "channels": 64, "batch_norm": false, "action_injection": "late"}' --gamma 0.9

# Discount factor=0.95 (medium-term)
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "gamma-0.95-d3-ch64-late" --model-type cnn --extra-specs '{"depth": 3, "channels": 64, "batch_norm": false, "action_injection": "late"}' --gamma 0.95

# Discount factor=0.99 (optional run; baseline gamma=0.5 is reused from Part 1)
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "gamma-0.99-d3-ch64-late" --model-type cnn --extra-specs '{"depth": 3, "channels": 64, "batch_norm": false, "action_injection": "late"}' --gamma 0.99

# Discount factor=0.999 (very long-term oriented)
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb --wandb_name "gamma-0.999-d3-ch64-late" --model-type cnn --extra-specs '{"depth": 3, "channels": 64, "batch_norm": false, "action_injection": "late"}' --gamma 0.999
```

**Results**:
| Discount Factor | Episodes | Checkpoint | Params | WandB Link | Value Est. Horizon |
|---|----------|-----------|--------|-----------|-------------|
| 0.9 | 1000 | `<FILL>` | `<FILL>` | `<FILL>` | Short-term |
| 0.95 | 1000 | `<FILL>` | `<FILL>` | `<FILL>` | Medium-term |
| 0.99 | 1000 | `<FILL>` | `<FILL>` | `<FILL>` | Long-term (baseline) |
| 0.999 | 1000 | `<FILL>` | `<FILL>` | `<FILL>` | Very long-term |

---

### 7.2 Discount Factor Performance Tournament

**Objective**: Compare models trained with different discount factors to identify optimal discount factor for Gobang.

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

# gamma=0.9 vs gamma=0.95
python -m tests.evaluator --player1_path <GAMMA09_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <GAMMA095_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

# gamma=0.95 vs gamma=0.99
python -m tests.evaluator --player1_path <GAMMA095_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <GAMMA099_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

# gamma=0.99 vs gamma=0.999
python -m tests.evaluator --player1_path <GAMMA099_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <GAMMA0999_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

# gamma=0.9 vs gamma=0.999 (extreme comparison)
python -m tests.evaluator --player1_path <GAMMA09_CHECKPOINT>/model_999.pth --player1_type checkpoint --player2_path <GAMMA0999_CHECKPOINT>/model_999.pth --player2_type checkpoint --episodes 500

python calculate_statistics.py  # Tournament statistics
```

**Results**:
| Matchup | Player 1 Win Rate | Player 2 Win Rate | p-value | Winner |
|---------|-------------------|-------------------|---------|--------|
| γ=0.9 vs γ=0.95 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| γ=0.95 vs γ=0.99 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| γ=0.99 vs γ=0.999 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| γ=0.9 vs γ=0.999 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |

**Discount Factor Analysis**:
- Optimal discount factor: `<TO BE FILLED IN>`
- Performance trend: `<TO BE FILLED IN>` (improving/degrading with increasing gamma)
- Planning horizon effect: `<TO BE FILLED IN>` (impact of longer-term value estimation on Gobang strategy)

---

## Part 8: Final Tournament

### 8.1 Grand Tournament

**Objective**: Select top-performing models from each category and run final tournament.

**Steps**:
1. Select best model from each experiment category
2. Run round-robin tournament with all selected models
3. Determine overall champion

**Selected Models**:
- Best Baseline: `<TO BE FILLED IN>`
- Best Depth: `<TO BE FILLED IN>`
- Best Width: `<TO BE FILLED IN>`
- Best BatchNorm: `<TO BE FILLED IN>`
- Best Action Injection: `<TO BE FILLED IN>`
- Best Learning Rate: `<TO BE FILLED IN>`
- Best Discount Factor: `<TO BE FILLED IN>`

**Executed script**:
```bash
cd /home/xinchengo/repo/USTC-AI3002-25fall/LABs/Final-Project/gobang
source /home/xinchengo/repo/USTC-AI3002-25fall/.venv/bin/activate

# Run all pairwise comparisons
# <SCRIPTS TO BE FILLED IN BASED ON SELECTED MODELS>

python calculate_statistics.py  # Final tournament rankings
```

**Results**:
| Rank | Model | Wins | Losses | Win Rate | Notes |
|------|-------|------|--------|----------|-------|
| 1 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 2 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 3 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 4 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 5 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |

---

## Summary and Conclusions

### Key Findings

**Depth Effect**: `<TO BE FILLED IN>`

**Width Effect**: `<TO BE FILLED IN>`

**Batch Normalization**: `<TO BE FILLED IN>`

**Action Injection Methods**: `<TO BE FILLED IN>`

**Learning Rate Impact**: `<TO BE FILLED IN>`

**Discount Factor Effect**: `<TO BE FILLED IN>`

### Recommendations

`<TO BE FILLED IN>`

### Future Work

`<TO BE FILLED IN>`

---

## Validation of Experimental Plan

### Small-Scale Validation Test

As part of validating the experimental plan, we conducted a small-scale test to ensure all components work correctly:

**Validation Steps**:
1. Trained a small CNN model (depth=3, channels=64, late injection) for 100 episodes
2. Evaluated the trained model against a random player for 50 episodes
3. Calculated statistics to verify the methodology

**Validation Results**:
- Training completed successfully with 2,804,064 parameters
- Model achieved 100% win rate against random player (50/50 games won)
- Factory system correctly loaded hyperparameters from training run
- Evaluation system worked as expected
- Statistical calculations validated

**Validation Conclusion**: The experimental plan is correctly structured and all components work as expected. The training, evaluation, and statistical analysis pipelines are functional and ready for full-scale experiments.