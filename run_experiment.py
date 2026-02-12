"""
Experiment runner for the SWWP-EEDPSO hybrid framework evaluation.

Orchestrates fair comparison by:
  1. Loading data and distributing identical ground-truth to all algorithms.
  2. Running all algorithms with the shared fitness function (fitness.py).
  3. Collecting fitness, Mass@K, and runtime statistics.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from core import fitness
from core.swwp import SlidingWindowWeightedPopularity
from core.eedpso import run_eedpso
from core.swwp_eedpso import run_swwp_eedpso
from core.de import run_de
from core.ga import run_ga

# Hyperparameters (from Optuna-based tuning in the EEDPSO paper)
PSO_PARAMS = dict(num_particles=30, max_iter=500, c1=1.13, c2=0.91, w=0.52)
DE_PARAMS = dict(pop_size=50, ngen=500, F=0.5, CR=0.7)
GA_PARAMS = dict(pop_size=50, ngen=500, cxpb=0.7, mutpb=0.2)

RECOMMEND_SIZE = 200
WINDOW_HOURS = 6


def load_and_prepare(items_path, interactions_path):
    """Load CSV files and return (item_ratings, item_tags_list, tag_popularity, interactions_df)."""
    items_df = pd.read_csv(items_path)
    interactions_df = pd.read_csv(interactions_path)
    interactions_df["datetime"] = pd.to_datetime(
        interactions_df["timestamp"], errors="coerce")

    ratings = items_df["avg_rating"].astype(float).tolist()
    tags_list = [
        row["category"].split("|") if pd.notna(row["category"]) else []
        for _, row in items_df.iterrows()
    ]

    # Build tag popularity (placeholder; replace with actual popularity_list)
    all_tags = set(t for ts in tags_list for t in ts)
    tag_pop = {t: 1.0 for t in all_tags}

    return ratings, tags_list, tag_pop, interactions_df


def get_ground_truth(interactions_df, current_time, window_hours=WINDOW_HOURS):
    """Extract future interactions within [current_time, current_time + window)."""
    end_time = current_time + timedelta(hours=window_hours)
    window = interactions_df[
        (interactions_df["datetime"] >= current_time)
        & (interactions_df["datetime"] < end_time)
    ]
    if window.empty:
        return [], []
    pop = window["item_id"].value_counts()
    return pop.index.tolist(), pop.values.tolist()


def run_experiment(items_path="data/items.csv",
                   interactions_path="data/interactions.csv",
                   n_test_points=21, seed=42):
    """Run the full benchmark across all algorithms."""
    random.seed(seed)
    np.random.seed(seed)

    ratings, tags_list, tag_pop, interactions_df = load_and_prepare(
        items_path, interactions_path)
    n_items = len(ratings)
    strategic = set(random.sample(range(n_items), min(500, n_items)))

    # Initialize shared fitness module
    fitness.set_data(ratings, tags_list, tag_pop, n_items, RECOMMEND_SIZE)
    fitness.set_strategic_items(strategic)

    # Initialize SWWP model
    swwp = SlidingWindowWeightedPopularity(interactions_df)
    swwp.precompute()

    # Generate test time points from data
    max_dt = interactions_df["datetime"].max()
    min_dt = interactions_df["datetime"].min()
    test_start = min_dt + (max_dt - min_dt) * 0.8
    test_end = max_dt - timedelta(hours=WINDOW_HOURS)

    test_times = pd.date_range(test_start, test_end,
                               periods=min(n_test_points, 50)).tolist()

    results = []
    for dt in test_times:
        true_items, true_counts = get_ground_truth(interactions_df, dt)
        if not true_items:
            continue

        # Distribute ground truth to shared fitness module
        fitness.set_ground_truth(fitness._make_time_key(dt), true_items, true_counts)

        row = {"datetime": dt, "gt_size": len(true_items)}

        # --- EEDPSO ---
        h, sol, f, t = run_eedpso(**PSO_PARAMS, current_time=dt)
        row.update({"eedpso_fit": f, "eedpso_time": t})

        # --- SWWP-EEDPSO ---
        h, sol, f, t = run_swwp_eedpso(swwp, dt, **PSO_PARAMS)
        row.update({"swwp_eedpso_fit": f, "swwp_eedpso_time": t})

        # --- DE ---
        h, sol, f, t = run_de(**DE_PARAMS, current_time=dt)
        row.update({"de_fit": f, "de_time": t})

        # --- GA ---
        h, sol, f, t = run_ga(**GA_PARAMS, current_time=dt)
        row.update({"ga_fit": f, "ga_time": t})

        results.append(row)
        fitness.clear_ground_truth()

        print(f"[{dt:%Y-%m-%d %H:%M}] "
              f"EEDPSO={row['eedpso_fit']:.1f}  "
              f"SWWP-EEDPSO={row['swwp_eedpso_fit']:.1f}  "
              f"DE={row['de_fit']:.1f}  GA={row['ga_fit']:.1f}")

    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("Summary (mean ± std):")
    for alg in ["eedpso", "swwp_eedpso", "de", "ga"]:
        col = f"{alg}_fit"
        print(f"  {alg:15s}: {df[col].mean():.2f} ± {df[col].std():.2f}")
    return df


if __name__ == "__main__":
    run_experiment()
