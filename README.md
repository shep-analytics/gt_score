# GT-Score: A Composite Loss Function for Trading Strategy Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the reproducible code for the GT-Score paper, providing a comprehensive framework for evaluating and optimizing trading strategies with built-in overfitting prevention.

## Overview

GT-Score is a composite loss function that evaluates trading strategies based on:
- **Excess Returns (μ)**: Mean return above buy-and-hold
- **Statistical Significance (ln(z))**: Z-score of excess returns
- **Consistency (R²)**: Coefficient of determination of the equity curve
- **Downside Risk (σd)**: Standard deviation of negative returns

The formula:
```
GT-Score = (μ × ln(z) × R²) / σd
```

## Installation

```bash
# From the extracted supplementary materials
cd reproducible_code

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
import pandas as pd
from src import gt_score, run_backtest, optimize
from strategies import strategies
from data import fetch_test_assets

# 1. Fetch sample data
data = fetch_test_assets()
aapl_data = data['AAPL']

# 2. Prepare data for strategy
df = {'ohlc': aapl_data}

# 3. Optimize a strategy using GT-Score
result = optimize(
    strategies=strategies[:1],  # Use first strategy
    data_frames=[df],
    loss_function=gt_score,
    optimization_method='random',
    max_evals=50
)

print(f"Best GT-Score: {result['best_loss']:.4f}")
print(f"Best Params: {result['best_params']}")
```

## Full Reproduction Steps

To reproduce all results from the paper:

### 1. Fetch Data
```bash
python3 -c "from data import fetch_all; fetch_all()"
```

### 2. Run Walk-Forward Validation
```bash
python3 experiments/run_walkforward.py
```

### 3. Run Monte Carlo Study
```bash
# Quick test first
python3 experiments/run_monte_carlo.py --assets 10 --seeds 5

# Full run (may take hours/days)
python3 experiments/run_monte_carlo.py
```

### 4. Run Statistical Analysis
```bash
python3 analysis/run_statistical_analysis.py
```

### 5. Generate Figures and Tables
```bash
python3 analysis/generate_figures.py
python3 analysis/generate_tables.py
```

## Directory Structure

```
reproducible_code/
├── src/                    # Core source code
│   ├── gt_score.py         # GT-Score implementation
│   ├── loss_functions.py   # All loss functions
│   ├── optimizers.py       # Optimization methods
│   ├── backtester.py       # Backtesting engine
│   ├── walkforward.py      # Walk-forward validation
│   └── statistics.py       # Statistical tests
├── strategies/             # Trading strategies
├── data/                   # Data fetching and caching
├── experiments/            # Experiment runners
├── analysis/               # Analysis scripts
├── tests/                  # Unit tests
└── output/                 # Generated outputs
    ├── figures/
    ├── tables/
    └── results/
```

## Testing

Run the unit tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=src --cov-report=html
```

## Key Features

### Edge Case Handling

GT-Score handles edge cases through a piecewise definition:

| Condition | Formula | Interpretation |
|-----------|---------|----------------|
| z ≤ 0 | 100 + 100×(1-e^(-\|z-1\|)) | Strategy underperforms buy-and-hold |
| 0 < z ≤ 1 | 100×(1-e^(-\|z-1\|)) | Marginal outperformance |
| z > 1 | -(μ×ln(z)×R²)/σd | Statistically significant outperformance |

### Smoothing Parameters

- **ε = 1e-6**: Added to σd when no negative returns exist (prevents division by zero)
- **N_periods = 50**: Default number of periods (≈6 trades/year over 8 years)

### Optimizers Supported

1. **Random Search**: Simple sampling within bounds
2. **Hyperopt (TPE)**: Tree-structured Parzen Estimator
3. **Genetic Algorithm**: DEAP-based evolutionary optimization

## Citation

If you use this code in your research, please cite:

```bibtex
@article{gt_score_2024,
  title={GT-Score: A Composite Loss Function for Trading Strategy Optimization},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kestner (1996) for R² as equity curve smoothness measure
- Sortino & van der Meer (1991) for downside deviation
- Bailey & López de Prado (2014) for probability of backtest overfitting
