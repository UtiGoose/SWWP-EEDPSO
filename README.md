# SWWP-EEDPSO: Hybrid Temporal Recommender System

A hybrid temporal recommendation framework integrating **Sliding-Window Weighted Popularity (SWWP)** with **Elite Evolutionary Discrete Particle Swarm Optimization (EEDPSO)**.

## Overview

This repository contains the core implementation for the paper:

> *A Hybrid Temporal Recommender System Based on Sliding-Window Weighted Popularity and Elite-Evolutionary Discrete Particle Swarm Optimization*

## Repository Structure

```
core/
├── fitness.py              # Unified fitness function shared by ALL algorithms
├── swwp.py                 # Sliding-Window Weighted Popularity model
├── eedpso.py               # Vanilla EEDPSO (baseline)
├── swwp_eedpso.py          # SWWP-EEDPSO hybrid framework
├── de.py                   # Differential Evolution (baseline)
├── ga.py                   # Genetic Algorithm (baseline)
└── run_experiment.py        # Experiment runner with ground-truth evaluation
```

## Key Design: Fairness Guarantee

All optimization algorithms (EEDPSO, SWWP-EEDPSO, DE, GA) share the **identical** unified fitness function defined in `fitness.py`:

```
f_hybrid = f_total + γ · f_pred
```

where:
- `f_total = f_pop + f_tag + 3·f_div + 100·f_cov` (base recommendation quality)
- `f_pred = Mass@K` (temporal prediction accuracy based on ground-truth future interactions)
- `γ = 10.0`, with `f_pred` scaled by 100

This ensures performance differences arise **solely from search strategies**, not from different objectives.

## Requirements

- Python >= 3.8
- NumPy
- Pandas
- DEAP (for DE and GA baselines)

## Dataset

Experiments use the [Amazon Reviews Data (2018)](https://nijianmo.github.io/amazon/index.html).

## License

MIT License
