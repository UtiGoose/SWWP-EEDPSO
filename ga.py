"""
Genetic Algorithm (GA) baseline for discrete recommendation.

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


def _tournament_select(pop, fit, k=3):
    """Tournament selection with tournament size k."""
    indices = random.sample(range(len(pop)), k)
    best = max(indices, key=lambda i: fit[i])
    return pop[best][:]


def _crossover_two_point(p1, p2):
    """Two-point crossover with duplicate removal."""
    size = len(p1)
    cx1 = random.randint(1, size - 2)
    cx2 = random.randint(cx1 + 1, size - 1)
    c1 = p1[:cx1] + p2[cx1:cx2] + p1[cx2:]
    c2 = p2[:cx1] + p1[cx1:cx2] + p2[cx2:]
    return _ensure_unique(c1), _ensure_unique(c2)


def _mutate(individual, indpb=0.05):
    """Random replacement mutation."""
    used = set(individual)
    for i in range(len(individual)):
        if random.random() < indpb:
            new_item = random.randint(0, NUM_ITEMS - 1)
            attempts = 0
            while new_item in used and attempts < 100:
                new_item = random.randint(0, NUM_ITEMS - 1)
                attempts += 1
            if new_item not in used:
                used.discard(individual[i])
                used.add(new_item)
                individual[i] = new_item
    return individual


def run_ga(pop_size=50, ngen=500, cxpb=0.7, mutpb=0.2, current_time=None):
    """
    Run Genetic Algorithm.

    Returns (fitness_history, best_solution, best_fitness, elapsed_seconds).
    """
    pop = [_init_individual() for _ in range(pop_size)]
    fit = [evaluate(ind, current_time) for ind in pop]

    t0 = _time.perf_counter()
    history = []

    for _ in range(ngen):
        # Selection
        offspring = [_tournament_select(pop, fit) for _ in range(pop_size)]
        off_fit = [None] * pop_size

        # Crossover
        for i in range(0, pop_size - 1, 2):
            if random.random() < cxpb:
                offspring[i], offspring[i + 1] = _crossover_two_point(
                    offspring[i], offspring[i + 1])
                off_fit[i] = None
                off_fit[i + 1] = None

        # Mutation
        for i in range(pop_size):
            if random.random() < mutpb:
                offspring[i] = _mutate(offspring[i])
                off_fit[i] = None

        # Evaluate
        for i in range(pop_size):
            if off_fit[i] is None:
                off_fit[i] = evaluate(offspring[i], current_time)

        pop = offspring
        fit = off_fit
        history.append(max(fit))

    elapsed = _time.perf_counter() - t0
    best_idx = int(np.argmax(fit))
    return history, pop[best_idx], fit[best_idx], elapsed
