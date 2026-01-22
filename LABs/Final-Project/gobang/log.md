# Plan

Plan for the gobang Actor-Critic Reinforcement Learning task

## Guide for AI agents

- The Python environment is a `uv` virtual environment located within `USTC-AI3002-25fall` (the repository root);
- You should `source .venv/bin/activate` before running `python`
- REMEMBER TO Use `uv pip` for package management
- scripts in `tests/` folder are intended to be called with `python -m test.<...> <...>` inside the `gobang` directory
- You should follow my plan to automatically execute scripts for my experiment
- Numerical calculations should be done by executing `calculate_statistics.py`, you should extend this file if necessary.

For this file, complete this file in a consistent, rigorous format.

For every experiments, reread the following lines:

- Every experiment, three parts: steps, script, and results
- Fill in the blanks, and the format should follow the contents and consistent with other already-filled-in experiment
- Numerical calculation are done by modifying and executing `calculate_statistics.py`
- Include all WanDB links for easier introspection

## First part, Plain CNN

### Has the CNN converged?

For non-deep and deep version of the CNN:

1. Train two models, each for 1000 epochs, checkpoint interval set to 200 (Each model only train once)
2. Let the 1000 epochs version play with the 800 epoch version,
    In half of the time, the 1000 epochs version hold black,
    in the other half, the 1000 epochs version hold white.
3. In total, 500 tests, record the winning rate of the 1000
    epochs version, denoted as $\hat{x}$;
4. We assume that $X_1, \cdots X_{500}$ follows a Bernoulli
    distribution $B(1, p)$
5. Do a hypothesis test on whether $p=0.5$ under $\alpha=0.01$

Fill in the lines below:

Executed script:
```bash
# after `source .venv/bin/activate`
python submission.py --num_episodes 1000 --checkpoint 200 --use_wandb
python submission.py --num_episodes 1000 --checkpoint 200 --use_deep --use_wandb
python -m tests.evaluator --player1_path checkpoints/gobang_model_20260121-102359/model_999.pth --player1_type checkpoint --player2_path checkpoints/gobang_model_20260121-102359/model_799.pth --player2_type checkpoint --episodes 500
python -m tests.evaluator --player1_path checkpoints/deep_gobang_model_20260121-102607/model_999.pth --player1_type checkpoint --player2_path checkpoints/deep_gobang_model_20260121-102607/model_799.pth --player2_type checkpoint --episodes 500
python calculate_statistics.py  # Calculate statistical results
```

Results for the regular CNN:

$$
\begin{gather}
H_0:p=0.5\leftrightarrow H_1:p\ne 0.5\\
X=\sum_{i=1}^{500} X_i = 245\\
p=0.655 > 0.01
\end{gather}
$$

Results for the deep CNN:

$$
\begin{gather}
H_0:p=0.5\leftrightarrow H_1:p\ne 0.5\\
X=\sum_{i=1}^{500} X_i = 254\\
p=0.721 >0.01
\end{gather}
$$

Notes:

- 'The evaluated models play black half of the time' is satisfied since the evaluator alternates who goes first (episode % 2 == 0 for player 1, episode % 2 == 1 for player 2)

### Is the CNN significantly better than random policy?

<TO BE FILLED IN: Play Baseline CNN Against Random Policy>

<THEN DO A SIGNIFICANCE TEST THAT COMPARES THE BASELINE CNN WITH THE RANDOM POLICY>

### Has the CNN improved significantly with depth increase?

**If they haven't converged, return to last step, increase the number of epochs and try again.**

Before experiment, CNN has a comparible number of parameters:

- Parameter count of regular CNN: <TO BE FILLED IN>
- Parameter count of deep CNN: <TO BE FILLED IN>

If the CNNs have both converged:

1. Let them play with each other for 100 times, 250 for each to hold black,
2. Do a hypothesis test on whether the CNN has improved significantly with $\alpha=0.01$

Fill in the lines below:

Executed script:

```bash
python -m tests.evaluator --player1_path checkpoints/gobang_model_20260121-102359/model_999.pth --player1_type checkpoint --player2_path checkpoints/deep_gobang_model_20260121-102607/model_999.pth --player2_type checkpoint --episodes 100
```

Results

$$
\begin{gather}
H_0:p_1=p_2\leftrightarrow H_1:p_1\ne p_2\\
\hat p_1 = 0.06,\quad \hat p_2 = 0.94\\
Z=\left[\hat p_1 (1-\hat p_1)+\hat p_2 (1 - \hat p_2)\over n\right]^{-\frac12}|\hat p_1 - \hat p_2|\sim N(0,1)\\
Z = -12.45\\
p=<TO BE FILLED IN>
\end{gather}
$$

### Convergence Speed and the depth of CNN

## Second Part, Shared backbone and separate backbone

### Is there a significant performance difference 

<!-- ## Third Part, Discount Factor

### Does discount factor affect performance on Plain CNNs

### Does discount factor affect performance on Residual CNNs

### As the model grew complex, does the discrepancy increase?

## Fourth Part, Residual CNN

### Does Residual CNN outperforms regular CNN?

### Depth and the performace discrepancy, a comparison -->