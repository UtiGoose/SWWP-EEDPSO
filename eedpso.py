"""
Elite Evolutionary Discrete Particle Swarm Optimization (EEDPSO).

Vanilla baseline that operates on discrete recommendation lists.
Uses the unified fitness function from fitness.py to ensure fair
comparison with SWWP-EEDPSO and other baselines.
"""

import random
import numpy as np
from core.fitness import evaluate, NUM_ITEMS, RECOMMEND_SIZE


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


def _random_neighbor(solution, num):
    """Replace `num` random positions with random items."""
    neighbor = solution.copy()
    indices = random.sample(range(len(neighbor)), int(num))
    used = set(neighbor)
    for idx in indices:
        new_item = random.randint(0, NUM_ITEMS - 1)
        while new_item in used:
            new_item = random.randint(0, NUM_ITEMS - 1)
        used.discard(neighbor[idx])
        used.add(new_item)
        neighbor[idx] = new_item
    return neighbor


def _guided_neighbor(solution, pbest, gbest, num, w, c1, c2):
    """Generate a neighbor guided by pbest and gbest."""
    neighbor = solution.copy()
    num = int(num)
    n_rand, n_pb, n_gb = _roulette_partition(num, w, c1, c2)
    indices = random.sample(range(len(neighbor)), num)

    # Random replacement
    used = set(neighbor)
    for idx in indices[:n_rand]:
        new_item = random.randint(0, NUM_ITEMS - 1)
        while new_item in used:
            new_item = random.randint(0, NUM_ITEMS - 1)
        used.discard(neighbor[idx])
        used.add(new_item)
        neighbor[idx] = new_item

    # pbest-guided replacement
    pb_diff = list(np.setdiff1d(pbest, neighbor))
    random.shuffle(pb_diff)
    n_pb_actual = min(n_pb, len(pb_diff))
    for i, idx in enumerate(indices[n_rand:n_rand + n_pb_actual]):
        neighbor[idx] = pb_diff[i]

    # gbest-guided replacement
    gb_diff = list(np.setdiff1d(gbest, neighbor))
    random.shuffle(gb_diff)
    n_gb_actual = min(n_gb, len(gb_diff))
    offset = n_rand + n_pb_actual
    for i, idx in enumerate(indices[offset:offset + n_gb_actual]):
        neighbor[idx] = gb_diff[i]

    return neighbor


def run_eedpso(num_particles=30, max_iter=500, c1=1.13, c2=0.91, w=0.52,
               current_time=None):
    """
    Run vanilla EEDPSO.

    Returns (fitness_history, best_solution, best_fitness, elapsed_seconds).
    """
    import time as _time

    # Initialize swarm with random solutions
    swarm_pos = []
    swarm_fit = []
    swarm_pb = []
    swarm_pb_fit = []

    for _ in range(num_particles):
        sol = np.random.choice(NUM_ITEMS, RECOMMEND_SIZE, replace=False).tolist()
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
            # Main position update
            nb = _guided_neighbor(swarm_pos[i], swarm_pb[i], gbest,
                                  v, w, c1, c2)
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

            # Cognitive perturbation (pbest neighborhood)
            nb = _random_neighbor(swarm_pb[i], 1)
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

            # Social perturbation (gbest neighborhood)
            nb = _random_neighbor(gbest, 1)
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

            v = (v * w
                 + random.random() * c1 * _jaccard_distance(swarm_pos[i], swarm_pb[i])
                 + random.random() * c2 * _jaccard_distance(swarm_pos[i], gbest))

        history.append(gbest_fit)

    elapsed = _time.perf_counter() - t0
    return history, gbest, gbest_fit, elapsed
