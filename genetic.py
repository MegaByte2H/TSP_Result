"""
Модуль algorithms.genetic: генетический алгоритм построения маршрута.
Публичный вход — run_algorithm(...), формат результата совпадает с другими алгоритмами.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import (
    TOURIST_SATIETY_CFG,
    build_distance_matrix,
    evaluate_tourist_route,
    initial_satiety_after_start,
    is_food_category_value,
    project_satiety_after_stop,
    route_stats,
)

def prepare_ga_arrays(df):
    """
    Что делает: вытаскивает из DataFrame массивы для GA.
        Параметры: df — туристические объекты.
        Возвращает: visit_times, costs, ratings, categories.
    """
    visit_times = df["visit_time_min"].to_numpy(dtype=float)
    costs = df["price_rub"].to_numpy(dtype=float)
    ratings = df["rating"].to_numpy(dtype=float)
    categories = df["category"].to_numpy()
    return visit_times, costs, ratings, categories


def ga_route_score(
    route,
    walk_time_matrix,
    visit_times,
    costs,
    ratings,
    categories,
    max_time_min,
    max_cost,
    min_food_visits=0,
    satiety_cfg=None,
    user_preferences=None
):
    """
    Fitness для GA.
    Теперь GA использует ту же персонализированную оценку, что и остальные алгоритмы.
    """

    score, stats = evaluate_tourist_route(
        route=route,
        walk_time_matrix=walk_time_matrix,
        visit_times=visit_times,
        costs=costs,
        ratings=ratings,
        categories=categories,
        max_time_min=max_time_min,
        max_cost=max_cost,
        min_food_visits=min_food_visits,
        satiety_cfg=satiety_cfg,
        user_preferences=user_preferences
    )

    return float(score), stats


def can_append_stop(route, next_idx, walk_time_matrix, visit_times, costs,
                    max_time_min, max_cost):
    """
    Что делает: проверяет, можно ли добавить точку в маршрут без нарушения ограничений.
        Параметры: route, next_idx, матрицы/массивы и ограничения.
        Возвращает: True/False.
    """
    if next_idx in route:
        return False

    candidate = route + [int(next_idx)]
    stats = route_stats(candidate, walk_time_matrix, visit_times, costs, np.zeros_like(costs))
    return (stats["total_time_min"] <= max_time_min) and (stats["total_cost"] <= max_cost)


def repair_route(route, start_idx, walk_time_matrix, visit_times, costs,
                 max_time_min, max_cost):
    """
    Что делает: чистит маршрут от дублей и точек, нарушающих ограничения.
        Параметры: route, start_idx, матрицы/массивы и ограничения.
        Возвращает: исправленный маршрут.
    """
    if route is None or len(route) == 0:
        return [int(start_idx)]

    cleaned = [int(start_idx)]
    seen = {int(start_idx)}

    for node in route:
        node = int(node)
        if node == start_idx:
            continue
        if node in seen:
            continue

        candidate = cleaned + [node]
        stats = route_stats(candidate, walk_time_matrix, visit_times, costs, np.zeros_like(costs))

        if stats["total_time_min"] <= max_time_min and stats["total_cost"] <= max_cost:
            cleaned.append(node)
            seen.add(node)

    if len(cleaned) == 0:
        cleaned = [int(start_idx)]

    return cleaned


def generate_feasible_route(start_idx, n_points, walk_time_matrix, visit_times, costs,
                            categories, max_time_min, max_cost, rng,
                            min_stops=4, satiety_cfg=None, user_preferences=None):
    """
    Что делает: генерирует допустимый стартовый маршрут.
        Параметры: старт, число точек, матрицы/массивы, ограничения, rng и настройки.
        Возвращает: список индексов маршрута.
    """
    if satiety_cfg is None:
        satiety_cfg = TOURIST_SATIETY_CFG

    route = [int(start_idx)]
    remaining = [i for i in range(n_points) if i != start_idx]
    rng.shuffle(remaining)

    current_satiety = initial_satiety_after_start(
        start_idx=start_idx,
        visit_times=visit_times,
        categories=categories,
        cfg=satiety_cfg
    )

    while remaining:
        feasible_candidates = []
        candidate_states = []

        last_idx = int(route[-1])
        last_is_food = is_food_category_value(categories[last_idx])

        for idx in remaining:
            if not can_append_stop(route, idx, walk_time_matrix, visit_times, costs, max_time_min, max_cost):
                continue

            delta_time = float(walk_time_matrix[last_idx, idx]) + float(visit_times[idx])

            sat_proj = project_satiety_after_stop(
                current_satiety=current_satiety,
                delta_time_min=delta_time,
                next_category=categories[idx],
                cfg=satiety_cfg
            )

            # если совсем критично голоден, не-food не берём
            if sat_proj["critical_without_food"]:
                continue

            # если только что поели и сытость высокая, подряд food стараемся не брать
            if last_is_food and sat_proj["next_is_food"] and current_satiety >= satiety_cfg["high_satiety_threshold"]:
                continue

            feasible_candidates.append(int(idx))
            candidate_states.append(sat_proj)

        if len(feasible_candidates) == 0:
            break

        urgent_food = [
            (idx, st) for idx, st in zip(feasible_candidates, candidate_states)
            if st["needs_food_now"] and st["next_is_food"]
        ]

        if len(urgent_food) > 0:
            chosen_idx, chosen_state = urgent_food[int(rng.integers(len(urgent_food)))]
        else:
            non_food_preferred = [
                (idx, st) for idx, st in zip(feasible_candidates, candidate_states)
                if not st["next_is_food"]
            ]

            if current_satiety >= satiety_cfg["high_satiety_threshold"] and len(non_food_preferred) > 0:
                chosen_idx, chosen_state = non_food_preferred[int(rng.integers(len(non_food_preferred)))]
            else:
                pair_idx = int(rng.integers(len(feasible_candidates)))
                chosen_idx = feasible_candidates[pair_idx]
                chosen_state = candidate_states[pair_idx]

        route.append(int(chosen_idx))
        remaining.remove(int(chosen_idx))
        current_satiety = float(chosen_state["satiety_final"])

        if len(route) >= min_stops and rng.random() < 0.15:
            break

    return repair_route(route, start_idx, walk_time_matrix, visit_times, costs, max_time_min, max_cost)


def tournament_select(population, scores, rng, tournament_size=4):
    """
    Что делает: выбирает родителя турнирным отбором.
        Параметры: population, scores, rng, tournament_size.
        Возвращает: выбранный маршрут.
    """
    idxs = rng.choice(len(population), size=min(tournament_size, len(population)), replace=False)
    best_idx = max(idxs, key=lambda i: scores[i])
    return population[int(best_idx)]


def ordered_crossover(parent1, parent2, start_idx, walk_time_matrix, visit_times, costs,
                      max_time_min, max_cost, rng):
    """
    Что делает: выполняет упорядоченное скрещивание двух маршрутов.
        Параметры: parent1, parent2, старт, матрицы/массивы, ограничения, rng.
        Возвращает: дочерний маршрут.
    """
    if len(parent1) <= 1:
        return parent2.copy()
    if len(parent2) <= 1:
        return parent1.copy()

    core1 = [x for x in parent1 if x != start_idx]
    core2 = [x for x in parent2 if x != start_idx]

    if len(core1) == 0:
        return repair_route(parent2, start_idx, walk_time_matrix, visit_times, costs, max_time_min, max_cost)
    if len(core2) == 0:
        return repair_route(parent1, start_idx, walk_time_matrix, visit_times, costs, max_time_min, max_cost)

    cut1 = int(rng.integers(0, len(core1)))
    cut2 = int(rng.integers(cut1, len(core1) + 1))
    segment = core1[cut1:cut2]

    child_core = segment.copy()
    for gene in core2:
        if gene not in child_core:
            child_core.append(gene)
    for gene in core1:
        if gene not in child_core:
            child_core.append(gene)

    child = [int(start_idx)] + child_core
    return repair_route(child, start_idx, walk_time_matrix, visit_times, costs, max_time_min, max_cost)


def mutate_route(route, start_idx, n_points, walk_time_matrix, visit_times, costs,
                 max_time_min, max_cost, rng,
                 p_swap=0.30, p_delete=0.20, p_insert=0.45):
    """
    Что делает: мутирует маршрут перестановкой/удалением/добавлением точек.
        Параметры: route, старт, число точек, матрицы/массивы, ограничения, rng.
        Возвращает: мутированный маршрут.
    """
    child = route.copy()

    if len(child) > 3 and rng.random() < p_swap:
        i, j = sorted(rng.choice(np.arange(1, len(child)), size=2, replace=False))
        child[i], child[j] = child[j], child[i]

    if len(child) > 2 and rng.random() < p_delete:
        del_idx = int(rng.integers(1, len(child)))
        child.pop(del_idx)

    if rng.random() < p_insert:
        candidates = [i for i in range(n_points) if i not in child]
        if candidates:
            new_node = int(candidates[rng.integers(len(candidates))])
            pos = int(rng.integers(1, len(child) + 1))
            trial = child[:pos] + [new_node] + child[pos:]
            child = repair_route(trial, start_idx, walk_time_matrix, visit_times, costs, max_time_min, max_cost)

    child = repair_route(child, start_idx, walk_time_matrix, visit_times, costs, max_time_min, max_cost)
    return child


def ga_constrained_route(
    tourist_df: pd.DataFrame,
    *,
    start_idx: int = 0,
    max_time_hours: float = 5,
    max_cost: float = 5000,
    population_size: int = 80,
    generations: int = 160,
    elite_size: int = 8,
    crossover_rate: float = 0.90,
    mutation_rate: float = 0.85,
    min_food_visits: int = 1,
    seed: int = 42,
    satiety_cfg=None,
    user_preferences=None
):
    """
    Что делает: запускает генетический алгоритм с ограничениями.
        Параметры: tourist_df и гиперпараметры GA.
        Возвращает: route, stats, history, info, walk_time_matrix.
    """
    if satiety_cfg is None:
        satiety_cfg = TOURIST_SATIETY_CFG

    rng = np.random.default_rng(seed)
    walk_time_matrix = build_distance_matrix(tourist_df, x_col="x", y_col="y") / (5000 / 60.0)

    visit_times, costs, ratings, categories = prepare_ga_arrays(tourist_df)
    n_points = len(tourist_df)
    max_time_min = float(max_time_hours) * 60.0

    population = [
        generate_feasible_route(
            start_idx=start_idx,
            n_points=n_points,
            walk_time_matrix=walk_time_matrix,
            visit_times=visit_times,
            costs=costs,
            categories=categories,
            max_time_min=max_time_min,
            max_cost=max_cost,
            rng=rng,
            satiety_cfg=satiety_cfg,
            user_preferences=user_preferences
        )
        for _ in range(population_size)
    ]

    ga_history_best = []
    ga_history_mean = []
    ga_best_route = None
    ga_best_stats = None
    ga_best_score = -1e18

    for generation in range(generations):
        scores = []
        stats_list = []

        for route in population:
            score, stats = ga_route_score(
                route=route,
                walk_time_matrix=walk_time_matrix,
                visit_times=visit_times,
                costs=costs,
                ratings=ratings,
                categories=categories,
                max_time_min=max_time_min,
                max_cost=max_cost,
                min_food_visits=min_food_visits,
                satiety_cfg=satiety_cfg,
                user_preferences=user_preferences
            )
            scores.append(score)
            stats_list.append(stats)

        scores = np.asarray(scores, dtype=float)
        best_idx = int(np.argmax(scores))

        if scores[best_idx] > ga_best_score:
            ga_best_score = float(scores[best_idx])
            ga_best_route = population[best_idx].copy()
            ga_best_stats = stats_list[best_idx].copy()

        ga_history_best.append(float(np.max(scores)))
        ga_history_mean.append(float(np.mean(scores)))

        elite_indices = list(np.argsort(scores)[-elite_size:][::-1])
        new_population = [population[i].copy() for i in elite_indices]

        while len(new_population) < population_size:
            parent1 = tournament_select(population, scores, rng)
            parent2 = tournament_select(population, scores, rng)

            if rng.random() < crossover_rate:
                child = ordered_crossover(
                    parent1, parent2, start_idx, walk_time_matrix, visit_times, costs,
                    max_time_min, max_cost, rng
                )
            else:
                child = parent1.copy()

            if rng.random() < mutation_rate:
                child = mutate_route(
                    child, start_idx, n_points, walk_time_matrix, visit_times, costs,
                    max_time_min, max_cost, rng
                )

            child = repair_route(child, start_idx, walk_time_matrix, visit_times, costs, max_time_min, max_cost)
            new_population.append(child)

        population = new_population

    info = {
        "algorithm": "genetic_algorithm",
        "population_size": population_size,
        "generations": generations,
        "elite_size": elite_size,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "min_food_visits": min_food_visits,
        "start_idx": start_idx,
        "max_time_hours": max_time_hours,
        "max_cost": max_cost,
        "seed": seed,
        "best_score": ga_best_score,
        "satiety_enabled": True
    }

    history = {
        "best": ga_history_best,
        "mean": ga_history_mean
    }

    return ga_best_route, ga_best_stats, history, info, walk_time_matrix

def run_algorithm(
    tourist_df: pd.DataFrame,
    *,
    start_idx: int = 0,
    max_time_min: float = 300,
    max_cost: float = 5000,
    min_food_visits: int = 0,
    seed: int = 42,
    satiety_cfg=None,
    user_preferences=None,
    **params,
):
    """
    Что делает: единая точка запуска алгоритма из main.py.
        Параметры: tourist_df, ограничения, seed, настройки сытости/пользователя и гиперпараметры.
        Возвращает: словарь с name, route, stats, history, info, walk_time_matrix.
    """
    route, stats, history, info, walk_time_matrix = ga_constrained_route(
        tourist_df=tourist_df,
        start_idx=start_idx,
        max_time_hours=float(max_time_min) / 60.0,
        max_cost=max_cost,
        min_food_visits=min_food_visits,
        seed=seed,
        satiety_cfg=satiety_cfg,
        user_preferences=user_preferences,
        **params,
    )
    return {
        "name": "GA",
        "route": route,
        "stats": stats,
        "history": history,
        "info": info,
        "walk_time_matrix": walk_time_matrix,
    }

