"""
SWWP-EEDPSO: Hybrid Temporal Recommendation Framework.

Deeply integrates Sliding-Window Weighted Popularity (SWWP) with EEDPSO
through three mechanisms:
  1. Temporal-aware particle initialization (Algorithm 2)
  2. Temporal-guided position updates
  3. Unified fitness evaluation (shared with all baselines)

The SWWP candidate pool guides the search strategy (initialization and
mutation), while the fitness function remains identical to vanilla EEDPSO
to ensure fair comparison.
"""

import random
import numpy as np
from core.fitness import evaluate, NUM_ITEMS, RECOMMEND_SIZE
from core.swwp import SlidingWindowWeightedPopularity


def _jaccard_distance(list_a, list_b):
    """Perturbed Jaccard distance with constant offset c=2."""
    sa, sb = set(list_a), set(list_b)
    union = len(sa | sb)
    sim = len(sa & sb) / union if union > 0 else 0
    return 2 + (1 - sim)


def _roulette_partition(num, w, c1, c2):
    """Partition replacement count into random / pbest / gbest via roulette."""
    n_w, n_c1, n_c2 = 0, 0, 0
    total = c1 + c2
    for _ in range(num):
        if random.random() < w:
            n_w += 1
        elif random.uniform(0, total) < c1:
            n_c1 += 1
        else:
            n_c2 += 1
    return n_w, n_c1, n_c2


# ======================================================================
# Temporal-aware initialization (Algorithm 2 in the paper)
# ======================================================================

def _init_particle_temporal(swwp_items, purchase_heat, swwp_set):
    """
    Generate an initial particle biased toward the SWWP candidate pool.

    The proportion of SWWP-sourced items is controlled by purchase_heat H(tau),
    capped at 0.8 to preserve exploration diversity.
    """
    n_temporal = int(RECOMMEND_SIZE * min(purchase_heat, 0.8))
    n_random = RECOMMEND_SIZE - n_temporal

    used = set()
    solution = []

    # Sample from SWWP candidates (top-1000)
    pool = swwp_items[:min(len(swwp_items), 1000)]
    indices = random.sample(range(len(pool)), min(n_temporal, len(pool)))
    for idx in indices:
        item = pool[idx]
        if item not in used and item < NUM_ITEMS:
            solution.append(item)
            used.add(item)

    # Fill remaining slots with random items
    while len(solution) < RECOMMEND_SIZE:
        item = random.randint(0, NUM_ITEMS - 1)
        if item not in used:
            solution.append(item)
            used.add(item)

    return solution[:RECOMMEND_SIZE]


# ======================================================================
# Temporal-guided neighbor generation
# ======================================================================

def _swwp_neighbor(solution, num, swwp_items, swwp_set):
    """
    Generate a neighbor that preferentially replaces non-SWWP positions
    with items from the SWWP candidate pool.
    """
    neighbor = solution.copy()
    num = int(num)

    non_swwp_idx = [i for i, it in enumerate(neighbor) if it not in swwp_set]
    swwp_idx = [i for i, it in enumerate(neighbor) if it in swwp_set]

    # Prefer replacing non-SWWP positions
    if len(non_swwp_idx) >= num:
        replace_idx = random.sample(non_swwp_idx, num)
    else:
        replace_idx = non_swwp_idx[:]
        extra = min(num - len(replace_idx), len(swwp_idx))
        if extra > 0:
            replace_idx += random.sample(swwp_idx, extra)

    used = set(neighbor)
    candidates = [it for it in swwp_items[:500]
                  if it not in used and it < NUM_ITEMS]

    for idx in replace_idx:
        if candidates:
            new_item = random.choice(candidates)
            candidates.remove(new_item)
        else:
            new_item = random.randint(0, NUM_ITEMS - 1)
            while new_item in used:
                new_item = random.randint(0, NUM_ITEMS - 1)
        used.discard(neighbor[idx])
        used.add(new_item)
        neighbor[idx] = new_item

    return neighbor


def _guided_neighbor_temporal(solution, pbest, gbest, num, w, c1, c2,
                              swwp_items, swwp_set):
    """Guided neighbor that injects SWWP items in the random partition."""
    neighbor = solution.copy()
    num = int(num)
    n_rand, n_pb, n_gb = _roulette_partition(num, w, c1, c2)
    indices = random.sample(range(len(neighbor)), num)

    used = set(neighbor)

    # Random partition: 70% chance to draw from SWWP candidates
    for idx in indices[:n_rand]:
        if swwp_items and random.random() < 0.7:
            cands = [it for it in swwp_items[:300]
                     if it not in used and it < NUM_ITEMS]
            if cands:
                new_item = random.choice(cands)
                used.discard(neighbor[idx])
                used.add(new_item)
                neighbor[idx] = new_item
                continue
        new_item = random.randint(0, NUM_ITEMS - 1)
        while new_item in used:
            new_item = random.randint(0, NUM_ITEMS - 1)
        used.discard(neighbor[idx])
        used.add(new_item)
        neighbor[idx] = new_item

    # pbest-guided replacement
    pb_diff = list(np.setdiff1d(pbest, neighbor))
    random.shuffle(pb_diff)
    for i, idx in enumerate(indices[n_rand:n_rand + min(n_pb, len(pb_diff))]):
        neighbor[idx] = pb_diff[i]

    # gbest-guided replacement
    gb_diff = list(np.setdiff1d(gbest, neighbor))
    random.shuffle(gb_diff)
    offset = n_rand + min(n_pb, len(np.setdiff1d(pbest, neighbor)))
    for i, idx in enumerate(indices[offset:offset + min(n_gb, len(gb_diff))]):
        neighbor[idx] = gb_diff[i]

    return neighbor


# ======================================================================
# Main algorithm (Algorithm 3 in the paper)
# ======================================================================

def run_swwp_eedpso(swwp_model, current_time,
                    num_particles=30, max_iter=500,
                    c1=1.13, c2=0.91, w=0.52):
    """
    Run the SWWP-EEDPSO hybrid framework.

    Parameters
    ----------
    swwp_model : SlidingWindowWeightedPopularity
        Pre-fitted SWWP model with cached scores.
    current_time : datetime
        Evaluation timestamp for temporal context.
    num_particles, max_iter, c1, c2, w : EEDPSO hyperparameters.
        Identical to vanilla EEDPSO to isolate the effect of SWWP integration.

    Returns
    -------
    history : list[float]
        Best fitness at each iteration.
    best_solution : list[int]
        Global-best recommendation list.
    best_fitness : float
        Fitness of the best solution.
    elapsed : float
        Wall-clock seconds.
    """
    import time as _time

    # Retrieve SWWP candidates and purchase heat
    swwp_items, _ = swwp_model.recommend(current_time)
    swwp_set = set(swwp_items)
    heat = SlidingWindowWeightedPopularity.purchase_heat(current_time)

    # --- Temporal-aware initialization (Algorithm 2) ---
    swarm_pos = []
    swarm_fit = []
    swarm_pb = []
    swarm_pb_fit = []

    for _ in range(num_particles):
        sol = _init_particle_temporal(swwp_items, heat, swwp_set)
        fit = evaluate(sol, current_time)
        swarm_pos.append(sol)
        swarm_fit.append(fit)
        swarm_pb.append(sol.copy())
        swarm_pb_fit.append(fit)

    gb_idx = int(np.argmax(swarm_pb_fit))
    gbest = swarm_pb[gb_idx].copy()
    gbest_fit = swarm_pb_fit[gb_idx]

    v = RECOMMEND_SIZE / 10
    history = []
    t0 = _time.perf_counter()

    for _ in range(max_iter):
        for i in range(num_particles):
            # --- Main update with temporal guidance ---
            nb = _guided_neighbor_temporal(
                swarm_pos[i], swarm_pb[i], gbest,
                v, w, c1, c2, swwp_items, swwp_set)
            nb_fit = evaluate(nb, current_time)
            if nb_fit > swarm_fit[i]:
                swarm_pos[i] = nb
                swarm_fit[i] = nb_fit
                if nb_fit > swarm_pb_fit[i]:
                    swarm_pb[i] = nb.copy()
                    swarm_pb_fit[i] = nb_fit
                    if nb_fit > gbest_fit:
                        gbest = nb.copy()
                        gbest_fit = nb_fit

            # --- Cognitive perturbation (SWWP-guided) ---
            nb = _swwp_neighbor(swarm_pb[i], 1, swwp_items, swwp_set)
            nb_fit = evaluate(nb, current_time)
            if nb_fit > swarm_fit[i]:
                swarm_pos[i] = nb
                swarm_fit[i] = nb_fit
                if nb_fit > swarm_pb_fit[i]:
                    swarm_pb[i] = nb.copy()
                    swarm_pb_fit[i] = nb_fit
                    if nb_fit > gbest_fit:
                        gbest = nb.copy()
                        gbest_fit = nb_fit

            # --- Social perturbation (SWWP-guided) ---
            nb = _swwp_neighbor(gbest, 1, swwp_items, swwp_set)
            nb_fit = evaluate(nb, current_time)
            if nb_fit > swarm_fit[i]:
                swarm_pos[i] = nb
                swarm_fit[i] = nb_fit
                if nb_fit > swarm_pb_fit[i]:
                    swarm_pb[i] = nb.copy()
                    swarm_pb_fit[i] = nb_fit
                    if nb_fit > gbest_fit:
                        gbest = nb.copy()
                        gbest_fit = nb_fit

            # Velocity update (Eq. 1)
            v = (v * w
                 + random.random() * c1
                   * _jaccard_distance(swarm_pos[i], swarm_pb[i])
                 + random.random() * c2
                   * _jaccard_distance(swarm_pos[i], gbest))

        history.append(gbest_fit)

    elapsed = _time.perf_counter() - t0
    return history, gbest, gbest_fit, elapsed
