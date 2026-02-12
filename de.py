"""
Differential Evolution (DE) baseline for discrete recommendation.

Uses the unified fitness function from fitness.py.
"""

import random
import time as _time
import numpy as np
from core.fitness import evaluate, NUM_ITEMS, RECOMMEND_SIZE


def _init_individual():
    """Create a random recommendation list with no duplicates."""
    return np.random.choice(NUM_ITEMS, RECOMMEND_SIZE, replace=False).tolist()


def _ensure_unique(solution):
    """Remove duplicates and backfill to RECOMMEND_SIZE."""
    seen = set()
    unique = []
    for item in solution:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    while len(unique) < RECOMMEND_SIZE:
        item = random.randint(0, NUM_ITEMS - 1)
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique[:RECOMMEND_SIZE]


def run_de(pop_size=50, ngen=500, F=0.5, CR=0.7, current_time=None):
    """
    Run Differential Evolution.

    Returns (fitness_history, best_solution, best_fitness, elapsed_seconds).
    """
    pop = [_init_individual() for _ in range(pop_size)]
    fit = [evaluate(ind, current_time) for ind in pop]

    t0 = _time.perf_counter()
    history = []

    for _ in range(ngen):
        for i in range(pop_size):
            r1, r2, r3 = random.sample([j for j in range(pop_size) if j != i], 3)

            # Mutation: symmetric set difference
            s2, s3 = set(pop[r2]), set(pop[r3])
            diff = list(s2.symmetric_difference(s3))
            vi = pop[r1][:]
            if diff:
                random.shuffle(diff)
                n_replace = max(1, int(len(diff) * F))
                indices = random.sample(range(RECOMMEND_SIZE),
                                        min(n_replace, RECOMMEND_SIZE))
                for k, idx in enumerate(indices):
                    if k < len(diff):
                        vi[idx] = diff[k]

            # Binomial crossover
            ui = pop[i][:]
            j_rand = random.randrange(RECOMMEND_SIZE)
            for j in range(RECOMMEND_SIZE):
                if random.random() < CR or j == j_rand:
                    ui[j] = vi[j]
            ui = _ensure_unique(ui)

            # Selection
            ui_fit = evaluate(ui, current_time)
            if ui_fit > fit[i]:
                pop[i] = ui
                fit[i] = ui_fit

        history.append(max(fit))

    elapsed = _time.perf_counter() - t0
    best_idx = int(np.argmax(fit))
    return history, pop[best_idx], fit[best_idx], elapsed
