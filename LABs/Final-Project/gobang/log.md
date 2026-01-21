# Plan

Plan for the gobang Actor-Critic Reinforcement Learning task

## Guide for AI agents

- The Python environment is a `uv` virtual environment located within `USTC-AI3002-25fall` (the repository root);
- You should `source .venv/bin/activate` before running `python`
- Use `uv pip` for package management
- scripts in `tests/` folder are intended to be called with `python -m test.<...> <...>` inside the `gobang` directory
- You should follow my plan to automatically execute scripts for my experiment

## First part, Plain CNN

### Has the CNN converged?

For non-deep and deep version of the CNN:

1. Train two models, each for 1000 epochs
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
python submission.py <FILL IN THE BLANKS>
python submission.py <FILL IN THE BLANKS>
<IF NECESSARY, FILL SOMETHING>
python -m tests.evaluate <FILL IN THE BLANKS>
```

Results for the regular CNN:

$$
H_0:p=0.5\leftrightarrow H_1:p\ne 0.5\\
\begin{gather}
X=\sum_{i=1}^{500} X_i = \text{TO BE FILLED IN}\\
p=\text{THE TO-BE-FILLED-IN p-value}
\end{gather}
$$

Results for the deep CNN:

$$
H_0:p=0.5\leftrightarrow H_1:p\ne 0.5\\
\begin{gather}
X=\sum_{i=1}^{500} X_i = \text{TO BE FILLED IN}\\
p=\text{THE TO-BE-FILLED-IN p-value}
\end{gather}
$$

### Has the CNN improved significantly with depth increase?

**If they haven't converged, return to last step, increase the number of epochs and try again.**

If the CNNs have both converged:

1. Let them play with each other for 100 times, 250 for each to hold black,
2. Do a hypothesis test on whether the CNN has improved significantly with $\alpha=0.01$

Fill in the lines below:

Executed script:

```bash
TO BE FILLED IN
```

Results

$$
TO BE FILLED IN
$$
