"""
Unified fitness function shared by ALL optimization algorithms.

Implements: f_hybrid = f_total + gamma * f_pred
- f_total: base recommendation quality (popularity, tag heat, diversity, coverage)
- f_pred:  Mass@K temporal prediction accuracy (ground-truth future interactions)

This module guarantees that performance differences between algorithms arise
solely from their search strategies, not from different optimization objectives.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Global state (set once via set_data / set_strategic_items / set_ground_truth)
# ---------------------------------------------------------------------------
NUM_ITEMS = None
RECOMMEND_SIZE = None

item_ratings = None          # np.ndarray, shape (NUM_ITEMS,)
item_tags_list = None        # list of list[str]
tag_popularity = None        # dict: tag -> float

tag_bits = None              # np.ndarray, shape (NUM_ITEMS,), dtype int32
tag_popularity_sum = None    # np.ndarray, shape (NUM_ITEMS,), dtype float32
tag_to_bit = None            # dict: tag -> int

STRATEGIC_ITEMS = set()

# Ground-truth future interactions keyed by (hour, month, day, weekday, year)
_gt_items = {}               # time_key -> list[int]
_gt_counts = {}              # time_key -> list[float]

# Fitness weights (from Optuna-based tuning reported in the EEDPSO paper)
W_POP = 1.0
W_TAG = 1.0
W_DIV = 3.0
W_COV = 100.0
GAMMA = 10.0                 # temporal prediction weight
PRED_SCALE = 100.0           # scaling factor for f_pred


def set_data(ratings, tags_list, tag_pop, n_items, rec_size):
    """Initialize item metadata used by the fitness function."""
    global item_ratings, item_tags_list, tag_popularity
    global NUM_ITEMS, RECOMMEND_SIZE
    global tag_bits, tag_popularity_sum, tag_to_bit

    item_ratings = np.asarray(ratings, dtype=np.float32)
    item_tags_list = tags_list
    tag_popularity = tag_pop
    NUM_ITEMS = n_items
    RECOMMEND_SIZE = rec_size

    unique_tags = sorted(tag_pop.keys())
    tag_to_bit = {tag: 1 << i for i, tag in enumerate(unique_tags)}

    tag_bits = np.zeros(NUM_ITEMS, dtype=np.int64)
    tag_popularity_sum = np.zeros(NUM_ITEMS, dtype=np.float32)
    for i in range(NUM_ITEMS):
        for tag in item_tags_list[i]:
            if tag in tag_to_bit:
                tag_bits[i] |= tag_to_bit[tag]
                tag_popularity_sum[i] += tag_pop[tag]


def set_strategic_items(items):
    """Set the strategic item set used by the coverage objective."""
    global STRATEGIC_ITEMS
    STRATEGIC_ITEMS = set(items)


def set_ground_truth(time_key, true_items, true_counts):
    """Register ground-truth future interactions for a given time context."""
    _gt_items[time_key] = true_items
    _gt_counts[time_key] = true_counts


def clear_ground_truth():
    """Clear cached ground-truth data between evaluation rounds."""
    _gt_items.clear()
    _gt_counts.clear()


def _count_bits(x):
    """Count the number of set bits in integer x."""
    return bin(x).count("1")


def _make_time_key(dt):
    """Convert a datetime object to the canonical time-key tuple."""
    return (dt.hour, dt.month, dt.day, dt.weekday(), dt.year)


def evaluate(solution, current_time=None):
    """
    Evaluate a recommendation list under the unified hybrid fitness.

    f_hybrid = f_total + GAMMA * f_pred * PRED_SCALE

    where
        f_total = W_POP*f_pop + W_TAG*f_tag + W_DIV*f_div + W_COV*f_cov
        f_pred  = Mass@K (proportion of future interaction volume captured)

    Parameters
    ----------
    solution : list[int]
        Ordered recommendation list of item indices.
    current_time : datetime or None
        If provided, the temporal prediction component f_pred is computed
        against the ground-truth interactions registered for this time.

    Returns
    -------
    float
        The scalar fitness value.
    """
    total_rating = 0.0
    total_pop = 0.0
    tags_union = 0
    overlap_count = 0
    valid_items = []

    for item in solution:
        if item < 0 or item >= NUM_ITEMS:
            continue
        valid_items.append(item)
        total_rating += item_ratings[item]
        total_pop += tag_popularity_sum[item]
        tags_union |= tag_bits[item]
        if item in STRATEGIC_ITEMS:
            overlap_count += 1

    # f_div: tag diversity
    diversity_reward = _count_bits(tags_union) * W_DIV

    # f_cov: strategic item coverage
    coverage_bonus = (overlap_count / RECOMMEND_SIZE) * W_COV

    base_fitness = (W_POP * total_rating
                    + W_TAG * total_pop
                    + diversity_reward
                    + coverage_bonus)

    # f_pred: temporal prediction accuracy (Mass@K)
    prediction_score = 0.0
    if current_time is not None:
        time_key = _make_time_key(current_time)
        if time_key in _gt_items:
            true_items = _gt_items[time_key]
            true_counts = _gt_counts[time_key]
            count_map = dict(zip(true_items, true_counts))

            k = len(valid_items)
            covered = sum(count_map.get(it, 0) for it in valid_items)
            total = (sum(true_counts[:k])
                     if len(true_counts) >= k else sum(true_counts))
            if total > 0:
                prediction_score = covered / total

    return base_fitness + GAMMA * prediction_score * PRED_SCALE
