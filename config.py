"""
Hyperparameter configuration for the SWWP-EEDPSO framework.

PSO family parameters were identified via Optuna-based Bayesian optimization
(100 trials with TPE sampler and median pruning) as reported in the EEDPSO paper.
DE and GA parameters follow standard discrete-optimization practice.
"""

# ==============================================================================
# Recommendation settings
# ==============================================================================
RECOMMEND_SIZE = 200          # K: recommendation list size
CANDIDATE_POOL_SIZE = 2000    # N_cand: SWWP candidate pool size
WINDOW_HOURS = 6              # Future interaction window for f_pred (hours)

# ==============================================================================
# Fitness function weights (Equation 2 in the paper)
# ==============================================================================
W_POP = 1.0                  # w1: Bayesian-adjusted popularity
W_TAG = 1.0                  # w2: tag heat
W_DIV = 3.0                  # w3: tag diversity
W_COV = 100.0                # w4: strategic item coverage
GAMMA = 10.0                 # gamma: temporal prediction weight
PRED_SCALE = 100.0            # f_pred scaling factor

# ==============================================================================
# EEDPSO / SWWP-EEDPSO parameters (shared to isolate SWWP contribution)
# ==============================================================================
PSO_PARAMS = {
    'num_particles': 30,
    'max_iter': 500,
    'c1': 1.255435187620126,   # cognitive coefficient
    'c2': 0.7448105445139115,  # social coefficient
    'w':  0.5512370844756477,  # inertia weight
}

# ==============================================================================
# Differential Evolution parameters
# ==============================================================================
DE_PARAMS = {
    'pop_size': 50,
    'ngen': 500,
    'F': 0.37,                 # differential weight
    'CR': 0.71,                # crossover rate
}

# ==============================================================================
# Genetic Algorithm parameters
# ==============================================================================
GA_PARAMS = {
    'pop_size': 50,
    'ngen': 500,
    'cxpb': 0.94,              # crossover probability
    'mutpb': 0.2,              # mutation probability
}

# ==============================================================================
# SWWP model parameters (Section 4.2 in the paper)
# ==============================================================================
SWWP_PARAMS = {
    'short_window': 14,        # Delta_S: short-term window (days)
    'alpha': 0.7,              # trend-periodicity fusion weight
    'trend_decay': 0.9,        # lambda_fast: exponential decay for trend
}

# ==============================================================================
# Purchase Heat factors (Equations 14-16 in the paper)
# Derived from multiplicative regression on 6-month hourly transaction volumes.
# ==============================================================================
PURCHASE_HEAT = {
    'base_heat': {             # B_s(tau): intra-day time segment
        0: 0.5,                # 00:00-04:00 (Q1: 25th percentile)
        1: 0.6,                # 04:00-08:00 (Q2: median)
        2: 0.6,                # 08:00-12:00 (Q2: median)
        3: 0.7,                # 12:00-16:00 (Q3: 75th percentile)
        4: 0.7,                # 16:00-20:00 (Q3: 75th percentile)
        5: 0.8,                # 20:00-24:00 (90th percentile)
    },
    'weekday_factor': {        # F_d(tau): day-of-week
        4: 1.1,                # Friday
        5: 1.2,                # Saturday
        6: 1.2,                # Sunday
        'default': 1.0,        # Mon-Thu
    },
    'month_factor': {          # F_m(tau): seasonal
        6: 1.1, 7: 1.1,       # mid-year promotions
        11: 1.15, 12: 1.15,   # holiday season
        'default': 1.0,
    },
    'max_heat': 0.9,           # H_max: upper bound to prevent over-biasing
}

# ==============================================================================
# Experimental protocol
# ==============================================================================
EXPERIMENT = {
    'train_window_days': 7,    # training window size
    'advance_days': 1,         # sliding step
    'n_test_points': 21,       # number of evaluation time points
    'test_fraction': 0.8,      # test period starts at 80% of data timeline
    'random_seed': 42,
}