# SWWP-EEDPSO: Hybrid Temporal Recommender System

A hybrid temporal recommendation framework integrating **Sliding-Window Weighted Popularity (SWWP)** with **Elite Evolutionary Discrete Particle Swarm Optimization (EEDPSO)** for non-personalized temporal recommendation under extreme data sparsity.

> **Paper:** *A Hybrid Temporal Recommender System Based on Sliding-Window Weighted Popularity and Elite-Evolutionary Discrete Particle Swarm Optimization*  
> Submitted to *Electronics* (MDPI), 2025.

---

## Overview

This framework addresses temporal recommendation in environments with extreme sparsity (density < 0.0005%), where personalized methods suffer from cold-start degradation. The key idea is a **deep integration** of temporal modeling with evolutionary optimization through three mechanisms:

1. **SWWP-guided initialization** — Purchase heat indicator H(τ) biases initial particles toward temporally relevant items
2. **SWWP-guided position updates** — Mutations preferentially select from SWWP candidate pool during evolutionary search
3. **Temporal fitness evaluation** — Unified fitness f_hybrid = f_total + γ·f_pred rewards alignment with future user behavior

## Repository Structure
```
SWWP-EEDPSO/
├── README.md              # This file
├── config.py              # Hyperparameter configuration
├── run_experiment.py      # Experiment runner
├── LICENSE                # MIT License
└── core/                  # Core algorithm modules
    ├── __init__.py
    ├── fitness.py         # Unified fitness function (shared by ALL algorithms)
    ├── swwp.py            # Sliding-Window Weighted Popularity model
    ├── eedpso.py          # Vanilla EEDPSO (baseline)
    ├── swwp_eedpso.py     # SWWP-EEDPSO hybrid framework (proposed method)
    ├── de.py              # Differential Evolution (baseline)
    └── ga.py              # Genetic Algorithm (baseline)
```

## Requirements

- Python >= 3.8
- NumPy >= 1.21
- Pandas >= 1.3
- DEAP >= 1.3 (for DE and GA baselines)

Install dependencies:
```bash
pip install numpy pandas deap
```

## Dataset

Experiments use the **Amazon Reviews Data (2018)** by Ni et al.

### How to obtain the dataset

1. Visit: https://nijianmo.github.io/amazon/index.html
2. Download the following **5-core** review files:
   - `AMAZON_FASHION_5.json.gz`
   - `Appliances_5.json.gz`
   - `Prime_Pantry_5.json.gz`
   - `Software_5.json.gz`
   - `All_Beauty_5.json.gz`
   - `Magazine_Subscriptions_5.json.gz`
3. Also download the corresponding **metadata** files for category information.

### Preprocessing

After downloading, preprocess into two CSV files:

**`data/interactions.csv`** — columns: `user_id`, `item_id`, `rating`, `timestamp`
```
user_id,item_id,rating,timestamp
0,1523,5.0,2018-03-15 14:30:00
1,892,4.0,2018-03-16 09:15:00
...
```

**`data/items.csv`** — columns: `item_id`, `avg_rating`, `category`
```
item_id,avg_rating,category
0,4.2,Books|Fiction
1,3.8,Electronics|Accessories
...
```

- `item_id`: integer index (0-based, contiguous)
- `timestamp`: parseable datetime string
- `category`: pipe-separated category labels

The preprocessing steps are:
1. Merge all category JSON files into a single interaction table
2. Re-index users and items to contiguous integers
3. Compute per-item average rating from the rating column
4. Extract category labels from metadata and join on item ID

### Dataset statistics (as used in the paper)

| Metric | Value |
|--------|-------|
| Users | 2,093,706 |
| Items | 283,932 |
| Interactions | 2,879,497 |
| Density | 0.000484% |
| Avg interactions/user | 1.38 |

## Hyperparameter Configuration

All hyperparameters are centralized in `config.py`. The PSO family parameters were identified via Optuna-based Bayesian optimization (100 trials) as described in the EEDPSO paper.

| Algorithm | Parameter | Value |
|-----------|-----------|-------|
| EEDPSO / SWWP-EEDPSO | Particles | 30 |
| | Max iterations | 500 |
| | Cognitive coefficient (c₁) | 1.26 |
| | Social coefficient (c₂) | 0.74 |
| | Inertia weight (ω) | 0.55 |
| DE | Population | 50 |
| | Generations | 500 |
| | Differential weight (F) | 0.37 |
| | Crossover rate (CR) | 0.71 |
| GA | Population | 50 |
| | Generations | 500 |
| | Crossover probability | 0.94 |
| | Mutation probability | 0.2 |

### Fitness function weights

```
f_hybrid = f_total + γ × f_pred × 100
f_total  = 1·f_pop + 1·f_tag + 3·f_div + 100·f_cov
γ = 10.0
```

## Reproduction Guide

### Temporal train-test split protocol

The experiments use a **sliding window evaluation** with strict temporal separation:

- **Training window**: 7 days of historical interactions `[t-7, t-1]`
- **Test window**: Next 6 hours of interactions `[t, t+6h]`
- **Advancement**: 1 day per evaluation point
- **Test points**: 21 time points sampled from the last 20% of the data timeline
- **Leakage prevention**: SWWP candidate pool and purchase heat are computed using *only* training window data. The temporal prediction component `f_pred` uses future interactions *only* for fitness evaluation (same role as ground-truth labels), not for model construction.

### Running the benchmark

```bash
# Ensure data files are in place
ls data/items.csv data/interactions.csv

# Run the full comparison (EEDPSO, SWWP-EEDPSO, DE, GA)
python run_experiment.py
```

### Expected output

The script prints per-time-point fitness values and a summary table:
```
Summary (mean ± std):
  eedpso         : 3194.14 ± 85.83
  swwp_eedpso    : 3384.26 ± 55.68
  de             :  3020.40 ± 27.77
  ga             :  3024.00 ± 43.13
```

## Key Design: Fairness Guarantee

All optimization algorithms share the **identical** unified fitness function defined in `fitness.py`. Performance differences arise **solely from search strategies**, not from different objectives. This design validates the paper's central hypothesis: temporally-aware search strategies are necessary for discovering temporally relevant recommendations.

## Citation

TODO

## License

MIT License — see [LICENSE](LICENSE) for details.