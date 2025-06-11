# GAdaOFUL: Robust Online Bandit Learning
## Overview 
This project implements **GAdaOFUL**, a robust online bandit learning algorithm designed to handle **nonlinear rewards** and **corrupted feedback**. The algorithm is tested against state-of-the-art baselines in various environments, demonstrating superior performance in terms of regret minimization and robustness.
##  Key Features
- 🛡️ **Robustness**: Handles heavy-tailed noise (t-distributed) and adversarial corruption.
- 📈 **Nonlinear Rewards**: Supports exponential reward mappings and general monotonic functions.
- 🔍 **Dynamic Environments**: Works with time-varying decision sets.
- 📊 **Benchmarking**: Includes implementations of 5 baseline algorithms (Greedy, OFUL, CW-OFUL, AdaOFUL).

## Experimental Setup

We perform numerical experiments in a 10-dimensional space $d = 10$, using a unit sphere for the target vector $\theta^*$, and conduct trials to compare different algorithms under various conditions. The setup includes:

- **Decision Set**: At each step, a decision set $D_t$ of 20 random unit vectors is generated.
  
- **Noise Distribution**: Noise is drawn from a heavy-tailed $t$-distribution with 3 degrees of freedom $t_3$.

- **Nonlinear Reward Function**: Rewards are mapped using the exponential function $y = \exp(x)$, where $x \in [-1,1]$.

- **Corruption**: During the first $n = 50$ steps, we simulate reward corruption using the flipping technique, where the reward is intentionally flipped to mislead the bandit into making opposite decisions.

## Running the Algorithm

- Modify parameters in `gada1.ipynb` as needed:

| Parameter    | Description                             | Default         |
|--------------|---------------------------------------|-----------------|
| `func`       | Reward function  | `lambda x: x`   |
| `corruption` | Number of adversarial corruption rounds (0 means no corruption) | 0               |
| `T`          | Number of iterations                   | 1000            |
| `dim`        | Dimension of decision variables       | 10              |
| `actions`    | Number of available actions            | 20              |

## Acknowledgments
This code is based on the implementation of [CW-OFUL](https://github.com/uclaml/CW-OFUL/tree/main). Thanks for their excellent works!
