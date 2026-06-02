from __future__ import annotations

import time
from pathlib import Path

from genetic import run_algorithm as run_genetic
from core import (
    TOURIST_SATIETY_CFG,
    build_user_preferences,
    get_effective_satiety_cfg,
)
from data import load_tourist_objects


DB_PATH = Path(r".\tourism_kazan_full.db")

USER_INPUT = {
    # Лимит времени на маршрут.
    #
    # Допустимые значения:
    # "2h"       — до 2 часов (120 мин)
    # "5h"       — до 5 часов (300 мин)
    # "full_day" — полный день (540 мин)
    #
    "time_limit": "5h",

    # Интенсивность маршрута.
    #
    # Допустимые значения:
    # "calm"   — спокойный маршрут (4–6 точек)
    #            доступен только для "2h" и "5h"
    #
    # "medium" — средняя насыщенность (6–9 точек)
    #            доступен для любого лимита времени
    #
    # "intense" — насыщенный маршрут (8–14 точек)
    #             доступен только для "full_day"
    #
    "intensity": "medium",

    # Пространственный охват города.
    #
    # Допустимые значения:
    # "compact"  — маршрут в пределах одного района,
    #              минимальные перемещения
    #
    # "balanced" — сбалансированный охват города
    #
    # "wide"     — широкий охват разных частей города
    #
    "coverage": "balanced",

    # Предпочтительные категории объектов.
    #
    # Допустимые категории:
    # "museum"
    # "historic_site"
    # "park"
    # "attraction"
    # "cafe"
    # "restaurant"
    #
    # Можно указывать несколько категорий:
    # ["museum", "park"]
    # ["historic_site", "museum"]
    # ["cafe", "restaurant"]
    #
    # [] — использовать все категории
    #
    "categories": [],

    # Максимальный бюджет маршрута (руб.)
    "budget": 5000,

    # Индекс стартовой точки в DataFrame туристических объектов.
    #
    # Важно:
    # это индекс строки DataFrame, а не ID объекта в базе данных.
    #
    "start_idx": 0,
}

# Seed для воспроизводимости результатов
SEED = 42

GA_PARAMS = dict(
    min_food_visits=0,
    population_size=80,
    generations=160,
    elite_size=8,
    crossover_rate=0.90,
    mutation_rate=0.85,
)


def main() -> dict:
    # 1. Загрузка данных
    tourist_objects = load_tourist_objects(DB_PATH)

    # 2. Подготовка пользовательских параметров
    user_preferences = build_user_preferences(USER_INPUT)
    satiety_cfg = get_effective_satiety_cfg(
        TOURIST_SATIETY_CFG,
        user_preferences,
    )

    max_time_min = user_preferences["max_time_min"]
    max_cost = user_preferences["max_cost"]

    # 3. Запуск генетического алгоритма
    t0 = time.perf_counter()

    result = run_genetic(
        tourist_df=tourist_objects,
        start_idx=int(USER_INPUT.get("start_idx", 0)),
        max_time_min=max_time_min,
        max_cost=max_cost,
        seed=SEED,
        satiety_cfg=satiety_cfg,
        user_preferences=user_preferences,
        **GA_PARAMS,
    )

    elapsed = time.perf_counter() - t0
    print(f"GA завершён за {elapsed:.1f} сек")

    route = list(result.get("route", []))
    stats = result.get("stats", {}) or {}

    route_points = []
    for idx in route:
        idx = int(idx)
        row = tourist_objects.iloc[idx]
        route_points.append(
            {
                "id": idx,
                "name": row.get("name"),
            }
        )

    return {
        # Список точек маршрута в порядке посещения
        "route": route_points,

        # Количество объектов в маршруте
        "points_count": len(route),

        # Общее время маршрута (минуты)
        "total_time_min": round(float(stats.get("total_time_min", 0.0)), 1),

        # Общая стоимость маршрута (рубли)
        "total_cost": round(float(stats.get("total_cost", 0.0)), 0),

        # Суммарный рейтинг всех посещённых объектов
        "total_rating": round(float(stats.get("total_rating", 0.0)), 2),

        # Средний рейтинг объектов маршрута
        "avg_rating": round(float(stats.get("avg_rating", 0.0)), 2),

        # Итоговая fitness-оценка маршрута после всех бонусов и штрафов
        "score": round(float(stats.get("score", result.get("score", 0.0))),2),

        # Время работы генетического алгоритма (секунды)
        "elapsed_sec": round(elapsed, 2),

        # Полная диагностическая статистика маршрута:
        # сытость, штрафы, категории, диаметр маршрута,
        # количество food-точек и т.д.
        "stats": stats,
    }


if __name__ == "__main__":
    print(main())