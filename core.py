"""
Модуль core.py: общие функции маршрутизации, оценки, сытости и персонализации.
Здесь лежит код, который используют все алгоритмы, чтобы входы/выходы и scoring были едиными.
"""

from __future__ import annotations

import numpy as np


def build_distance_matrix(df, x_col="x", y_col="y"):
    """
    Что делает: строит евклидову матрицу расстояний между объектами.
        Параметры: df — DataFrame; x_col/y_col — колонки координат.
        Возвращает: numpy-матрицу расстояний.
    """
    coords = df[[x_col, y_col]].to_numpy(dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return dist

def has_food_category(route, categories, min_food_visits=2):
    """
    Что делает: проверяет, есть ли в маршруте достаточно кафе/ресторанов.
        Параметры: route — индексы точек; categories — массив категорий; min_food_visits — минимум food-точек.
        Возвращает: True/False.
    """
    route_categories = categories[route]
    food_mask = (route_categories == "cafe") | (route_categories == "restaurant")
    return np.sum(food_mask) >= min_food_visits

def roulette_pick(prob, rng: np.random.Generator):
    """
    Что делает: выбирает индекс по распределению вероятностей методом рулетки.
        Параметры: prob — массив вероятностей; rng — генератор NumPy.
        Возвращает: выбранный индекс.
    """
    cdf = np.cumsum(prob)
    r = rng.random()
    return int(np.searchsorted(cdf, r, side="right"))

def route_stats(route, walk_time_matrix, visit_times, costs, ratings):
    """
    Считает основные метрики маршрута.
    """

    if len(route) == 0:
        return {
            "route_points": 0,
            "total_walk_time_min": 0.0,
            "total_visit_time_min": 0.0,
            "total_time_min": 0.0,
            "total_cost": 0.0,
            "total_rating": 0.0,
            "avg_rating": 0.0,
        }

    total_walk = sum(
        walk_time_matrix[route[i], route[i + 1]]
        for i in range(len(route) - 1)
    )

    total_visit = float(np.sum(visit_times[route]))
    total_cost = float(np.sum(costs[route]))
    total_rating = float(np.sum(ratings[route]))
    avg_rating = float(np.mean(ratings[route]))

    return {
        "route_points": len(route),
        "total_walk_time_min": float(total_walk),
        "total_visit_time_min": float(total_visit),
        "total_time_min": float(total_walk + total_visit),
        "total_cost": total_cost,
        "total_rating": total_rating,
        "avg_rating": avg_rating,
    }

TOURIST_SATIETY_CFG = {
    "satiety_max": 100.0,
    "satiety_decay_per_min": 0.25,
    "low_satiety_threshold": 55.0,
    "critical_satiety_threshold": 30.0,
    "high_satiety_threshold": 75.0,

    # штрафы / бонусы в fitness
    "missed_food_penalty": 40.0,          # штраф: надо было есть, но пошли не в food
    "critical_hunger_penalty": 180.0,     # штраф: допустили очень сильный голод
    "consecutive_food_penalty": 35.0,     # штраф за два food подряд
    "healthy_satiety_bonus_weight": 0.03, # небольшой бонус за хороший минимум сытости

    # коэффициенты для ACO при выборе следующей точки
    "urgent_food_boost": 2.2,             # если надо срочно есть и точка food
    "early_food_penalty_factor": 0.45,    # если есть ещё рано, food хуже
    "consecutive_food_penalty_factor": 0.12  # если подряд food -> режем привлекательность
}

TOURIST_SATIETY_CFG.update({
    # Ограничение на число точек маршрута
    "min_route_points": 8,
    "max_route_points": 12,

    # Штрафы за слишком короткий / слишком длинный маршрут
    "too_few_points_penalty": 80.0,
    "too_many_points_penalty": 80.0,

    # Штраф за близко расположенные точки
    "close_point_threshold_m": 300.0,
    "close_pair_penalty": 6.0,

    # Штраф за слишком маленький пространственный охват
    "min_route_diameter_m": 900.0,
    "small_diameter_penalty": 40.0,

    # Штраф за однотипность категорий
    "max_same_category_ratio": 0.45,
    "same_category_penalty": 50.0,

    # Штраф за ненужную еду
    "satiety_high_threshold": 85.0,
    "unnecessary_food_penalty": 25.0,
})

def is_food_category_value(category_value):
    """
    Что делает: определяет, относится ли категория к еде.
        Параметры: category_value — значение категории.
        Возвращает: True для cafe/restaurant, иначе False.
    """
    return str(category_value) in {"cafe", "restaurant"}


def project_satiety_after_stop(current_satiety, delta_time_min, next_category, cfg=None):
    """
    Смотрит, какой будет сытость после похода в следующую точку.
    delta_time_min = время дороги + время пребывания в самой следующей точке.
    Если следующая точка food, сытость после неё восстанавливается до max.
    Параметры: current_satiety — текущая сытость; delta_time_min — дорога+визит; next_category — категория следующей точки; cfg — настройки сытости.
    Возвращает: словарь с сытостью после убывания, финальной сытостью и food-флагами.
    """
    if cfg is None:
        cfg = TOURIST_SATIETY_CFG

    satiety_after_decay = max(
        0.0,
        float(current_satiety) - float(delta_time_min) * float(cfg["satiety_decay_per_min"])
    )

    next_is_food = is_food_category_value(next_category)
    needs_food_now = satiety_after_decay <= float(cfg["low_satiety_threshold"])
    critical_without_food = (satiety_after_decay <= float(cfg["critical_satiety_threshold"])) and (not next_is_food)

    satiety_final = float(cfg["satiety_max"]) if next_is_food else satiety_after_decay

    return {
        "satiety_after_decay": float(satiety_after_decay),
        "satiety_final": float(satiety_final),
        "next_is_food": bool(next_is_food),
        "needs_food_now": bool(needs_food_now),
        "critical_without_food": bool(critical_without_food)
    }


def initial_satiety_after_start(start_idx, visit_times, categories, cfg=None):
    """
    Турист стартует сытым.
    Сразу учитываем время пребывания в стартовой точке.
    Если стартовая точка food, сытость снова = max.
    Параметры: start_idx — индекс старта; visit_times — массив длительностей; categories — массив категорий; cfg — настройки сытости.
    Возвращает: числовое значение сытости после стартовой точки.
    """
    if cfg is None:
        cfg = TOURIST_SATIETY_CFG

    start_cat = categories[start_idx]
    start_visit_time = float(visit_times[start_idx])

    start_state = project_satiety_after_stop(
        current_satiety=float(cfg["satiety_max"]),
        delta_time_min=start_visit_time,
        next_category=start_cat,
        cfg=cfg
    )
    return float(start_state["satiety_final"])


def compute_satiety_route_stats(route, walk_time_matrix, visit_times, categories, cfg=None):
    """
    Что делает: считает динамику сытости и food-метрики по готовому маршруту.
    Параметры: route — индексы точек; walk_time_matrix — время переходов; visit_times — время визитов; categories — категории; cfg — настройки сытости.
    Возвращает: словарь статистик сытости, включая trace, минимум, финальное значение и нарушения.
    """
    if cfg is None:
        cfg = TOURIST_SATIETY_CFG

    route_len = 0 if route is None else len(route)

    if route is None or len(route) == 0:
        return {
            "route_len": 0,
            "final_satiety": float(cfg["satiety_max"]),
            "min_satiety": float(cfg["satiety_max"]),

            "food_visit_count": 0,
            "urgent_food_steps": 0,
            "missed_food_urgencies": 0,
            "critical_hunger_violations": 0,
            "consecutive_food_pairs": 0,
            "unnecessary_food_visits": 0,

            "close_pair_count": 0,
            "route_diameter_m": 0.0,

            "max_category_ratio": 0.0,
            "dominant_category": None,

            "satiety_trace": []
        }

    route_indices = list(map(int, route))

    # =========================
    # 1. Пространственная статистика
    # =========================
    # walk_time_matrix хранит минуты ходьбы.
    # Восстанавливаем примерное расстояние в метрах:
    # скорость = 5000 м / 60 мин
    walking_speed_m_per_min = 5000.0 / 60.0

    route_distances_m = []

    for i in range(len(route_indices)):
        for j in range(i + 1, len(route_indices)):
            a = route_indices[i]
            b = route_indices[j]
            dist_m = float(walk_time_matrix[a, b]) * walking_speed_m_per_min
            route_distances_m.append(dist_m)

    if len(route_distances_m) > 0:
        close_pair_count = int(
            sum(
                d < float(cfg["close_point_threshold_m"])
                for d in route_distances_m
            )
        )
        route_diameter_m = float(max(route_distances_m))
    else:
        close_pair_count = 0
        route_diameter_m = 0.0

    # =========================
    # 2. Статистика разнообразия категорий
    # =========================

    route_categories = [str(categories[int(i)]) for i in route_indices]
    category_counts = Counter(route_categories)

    dominant_category, dominant_count = category_counts.most_common(1)[0]
    max_category_ratio = dominant_count / route_len

    # =========================
    # 3. Статистика сытости
    # =========================

    satiety_trace = []

    food_visit_count = 0
    urgent_food_steps = 0
    missed_food_urgencies = 0
    critical_hunger_violations = 0
    consecutive_food_pairs = 0
    unnecessary_food_visits = 0

    current_satiety = initial_satiety_after_start(
        start_idx=int(route[0]),
        visit_times=visit_times,
        categories=categories,
        cfg=cfg
    )

    satiety_trace.append(float(current_satiety))

    if is_food_category_value(categories[int(route[0])]):
        food_visit_count += 1

        if current_satiety > float(cfg["satiety_high_threshold"]):
            unnecessary_food_visits += 1

    min_satiety = float(current_satiety)

    for k in range(1, len(route)):
        prev_idx = int(route[k - 1])
        next_idx = int(route[k])

        delta_time = (
            float(walk_time_matrix[prev_idx, next_idx])
            + float(visit_times[next_idx])
        )

        proj = project_satiety_after_stop(
            current_satiety=current_satiety,
            delta_time_min=delta_time,
            next_category=categories[next_idx],
            cfg=cfg
        )

        next_is_food = bool(proj["next_is_food"])

        if proj["needs_food_now"]:
            urgent_food_steps += 1

            if not next_is_food:
                missed_food_urgencies += 1

        if proj["critical_without_food"]:
            critical_hunger_violations += 1

        if is_food_category_value(categories[prev_idx]) and next_is_food:
            consecutive_food_pairs += 1

        # Главное новое:
        # если человек идёт есть при высокой сытости, это считается лишней едой
        if next_is_food and current_satiety > float(cfg["satiety_high_threshold"]):
            unnecessary_food_visits += 1

        if next_is_food:
            food_visit_count += 1

        min_satiety = min(min_satiety, float(proj["satiety_after_decay"]))

        current_satiety = float(proj["satiety_final"])
        satiety_trace.append(float(current_satiety))

    return {
        "route_len": int(route_len),
        "final_satiety": float(current_satiety),
        "min_satiety": float(min_satiety),

        "food_visit_count": int(food_visit_count),
        "urgent_food_steps": int(urgent_food_steps),
        "missed_food_urgencies": int(missed_food_urgencies),
        "critical_hunger_violations": int(critical_hunger_violations),
        "consecutive_food_pairs": int(consecutive_food_pairs),
        "unnecessary_food_visits": int(unnecessary_food_visits),

        "close_pair_count": int(close_pair_count),
        "route_diameter_m": float(route_diameter_m),

        "max_category_ratio": float(max_category_ratio),
        "dominant_category": dominant_category,

        "satiety_trace": satiety_trace
    }


def merge_base_and_satiety_stats(base_stats, satiety_stats):
    """
    Что делает: объединяет базовые метрики маршрута и метрики сытости.
    Параметры: base_stats — словарь времени/стоимости/рейтинга; satiety_stats — словарь сытости.
    Возвращает: один объединённый словарь статистик.
    """
    merged = dict(base_stats)
    merged.update(satiety_stats)
    return merged


def satiety_adjusted_route_score(
    base_stats,
    satiety_stats,
    cfg=None,
    user_preferences=None
):
    """
    Персонализированная fitness-функция.

    Если user_preferences не передан, используется поведение, близкое к старому:
    8-12 точек, сбалансированный охват, стандартные штрафы.
    Параметры: base_stats — базовые метрики; satiety_stats — метрики сытости; cfg — настройки; user_preferences — предпочтения пользователя.
    Возвращает: числовой score маршрута.
    """

    if cfg is None:
        cfg = TOURIST_SATIETY_CFG

    if user_preferences is None:
        user_preferences = {
            "target_route_points_min": int(cfg.get("min_route_points", 8)),
            "target_route_points_max": int(cfg.get("max_route_points", 12)),
            "route_length_penalty_weight": 80,

            "quality_weight": 10,
            "coverage_weight": 4,
            "time_weight": 0.03,
            "satiety_weight": 3,

            "max_category_ratio": 0.45,
            "category_diversity_penalty_weight": 50,

            "min_route_diameter_m": 900,
            "max_route_diameter_m": None,
            "diameter_under_penalty_weight": 40,
            "diameter_over_penalty_weight": 0,

            "close_pair_penalty_weight": 6,

            "missed_food_urgency_penalty_weight": 40,
            "critical_hunger_penalty_weight": 180,
            "consecutive_food_pair_penalty_weight": 35,
            "unnecessary_food_penalty_weight": 25
        }

    route_len = int(base_stats.get("route_len", 0))
    avg_rating = float(base_stats.get("avg_rating", 0.0))
    total_time_min = float(base_stats.get("total_time_min", 0.0))

    min_satiety = float(satiety_stats.get("min_satiety", 100.0))

    quality_score = float(user_preferences["quality_weight"]) * avg_rating
    coverage_score = float(user_preferences["coverage_weight"]) * np.log1p(route_len)
    time_score = float(user_preferences["time_weight"]) * total_time_min
    satiety_score = float(user_preferences["satiety_weight"]) * (min_satiety / 100.0)

    penalty = 0.0

    # =========================
    # 1. Длина маршрута
    # =========================

    target_min = int(user_preferences["target_route_points_min"])
    target_max = int(user_preferences["target_route_points_max"])
    length_weight = float(user_preferences["route_length_penalty_weight"])

    if route_len < target_min:
        penalty += length_weight * (target_min - route_len)

    if route_len > target_max:
        penalty += length_weight * (route_len - target_max)

    # =========================
    # 2. Сытость и еда
    # =========================

    penalty += float(user_preferences["missed_food_urgency_penalty_weight"]) * int(
        satiety_stats.get("missed_food_urgencies", 0)
    )

    penalty += float(user_preferences["critical_hunger_penalty_weight"]) * int(
        satiety_stats.get("critical_hunger_violations", 0)
    )

    penalty += float(user_preferences["consecutive_food_pair_penalty_weight"]) * int(
        satiety_stats.get("consecutive_food_pairs", 0)
    )

    penalty += float(user_preferences["unnecessary_food_penalty_weight"]) * int(
        satiety_stats.get("unnecessary_food_visits", 0)
    )

    # =========================
    # 3. Близкие точки
    # =========================

    close_pair_count = int(base_stats.get("close_pair_count", 0))
    penalty += float(user_preferences["close_pair_penalty_weight"]) * close_pair_count

    # =========================
    # 4. Компактность / охват города
    # =========================

    route_diameter = float(base_stats.get("route_diameter_m", 0.0))

    min_diameter = user_preferences.get("min_route_diameter_m", None)
    max_diameter = user_preferences.get("max_route_diameter_m", None)

    if min_diameter is not None:
        min_diameter = float(min_diameter)

        if min_diameter > 0 and route_diameter < min_diameter:
            penalty += float(user_preferences["diameter_under_penalty_weight"]) * (
                1.0 - route_diameter / min_diameter
            )

    if max_diameter is not None:
        max_diameter = float(max_diameter)

        if max_diameter > 0 and route_diameter > max_diameter:
            penalty += float(user_preferences["diameter_over_penalty_weight"]) * (
                route_diameter / max_diameter - 1.0
            )

    # =========================
    # 5. Однотипность категорий
    # =========================

    max_category_ratio = float(satiety_stats.get("max_category_ratio", 0.0))
    allowed_ratio = float(user_preferences["max_category_ratio"])

    if max_category_ratio > allowed_ratio:
        penalty += float(user_preferences["category_diversity_penalty_weight"]) * (
            max_category_ratio - allowed_ratio
        )

    score = (
        quality_score
        + coverage_score
        + time_score
        + satiety_score
        - penalty
    )

    return float(score)

from collections import Counter
import numpy as np

# =========================
# Персонализация маршрута
# =========================

SUPPORTED_MODEL_CATEGORIES = [
    "museum",
    "historic_site",
    "park",
    "attraction",
    "cafe",
    "restaurant"
]

CATEGORY_ALIASES = {
    # русские варианты
    "музей": "museum",
    "музеи": "museum",
    "исторические места": "historic_site",
    "история": "historic_site",
    "парк": "park",
    "парки": "park",
    "достопримечательности": "attraction",
    "достопримечательность": "attraction",
    "кафе": "cafe",
    "ресторан": "restaurant",
    "рестораны": "restaurant",
    "кафе и рестораны": "food",
    "еда": "food",

    # английские варианты
    "museum": "museum",
    "historic_site": "historic_site",
    "history": "historic_site",
    "park": "park",
    "attraction": "attraction",
    "cafe": "cafe",
    "restaurant": "restaurant",
    "food": "food"
}

TIME_LIMIT_CONFIG = {
    "2h": 120,
    "2_hours": 120,
    "2 часа": 120,
    120: 120,

    "5h": 300,
    "5_hours": 300,
    "5 часов": 300,
    300: 300,

    "full_day": 540,
    "day": 540,
    "8h": 540,
    "9h": 540,
    "целый день": 540,
    540: 540
}

INTENSITY_POINTS_CONFIG = {
    # спокойный маршрут разрешен только для 2-5 часов
    (120, "calm"): (4, 6),
    (300, "calm"): (4, 6),

    # средний маршрут разрешен при любом лимите
    (120, "medium"): (6, 9),
    (300, "medium"): (6, 9),
    (540, "medium"): (6, 9),

    # насыщенный маршрут разрешен только для полного дня
    (540, "intense"): (8, 14)
}

INTENSITY_ALIASES = {
    "calm": "calm",
    "спокойный": "calm",
    "спокойный маршрут": "calm",

    "medium": "medium",
    "средний": "medium",
    "средний маршрут": "medium",

    "intense": "intense",
    "насыщенный": "intense",
    "насыщенный маршрут": "intense"
}

COVERAGE_ALIASES = {
    "compact": "compact",
    "компактный": "compact",
    "хочу компактный маршрут в одном районе": "compact",

    "balanced": "balanced",
    "сбалансированный": "balanced",
    "хочу сбалансированный маршрут": "balanced",

    "wide": "wide",
    "широкий": "wide",
    "хочу посмотреть разные части города": "wide"
}


def normalize_category(category):
    """
    Что делает: нормализует название категории через словарь алиасов.
    Параметры: category — строка или None с категорией пользователя/объекта.
    Возвращает: нормализованную категорию или None.
    """
    if category is None:
        return None

    cat = str(category).strip().lower()
    return CATEGORY_ALIASES.get(cat, cat)


def normalize_categories(categories):
    """
    Что делает: нормализует список категорий и раскрывает food в cafe/restaurant.
    Параметры: categories — список пользовательских категорий или None.
    Возвращает: список поддерживаемых категорий без дублей.
    """
    if categories is None:
        categories = []

    normalized = []

    for cat in categories:
        mapped = normalize_category(cat)

        if mapped == "food":
            normalized.extend(["cafe", "restaurant"])
        elif mapped in SUPPORTED_MODEL_CATEGORIES:
            normalized.append(mapped)

    # Убираем повторы, но сохраняем порядок
    result = []
    seen = set()

    for cat in normalized:
        if cat not in seen:
            result.append(cat)
            seen.add(cat)

    return result


def normalize_time_limit(value):
    """
    Что делает: переводит пользовательский лимит времени в минуты.
    Параметры: value — строковый или числовой вариант лимита.
    Возвращает: число минут; при некорректном значении выбрасывает ValueError.
    """
    if value is None:
        return 300

    if value in TIME_LIMIT_CONFIG:
        return TIME_LIMIT_CONFIG[value]

    value_str = str(value).strip().lower()

    if value_str in TIME_LIMIT_CONFIG:
        return TIME_LIMIT_CONFIG[value_str]

    raise ValueError(
        f"Некорректный лимит времени: {value}. "
        f"Допустимые варианты: '2h', '5h', 'full_day'."
    )


def normalize_intensity(value):
    """
    Что делает: нормализует пользовательскую интенсивность маршрута.
    Параметры: value — строковый вариант интенсивности или None.
    Возвращает: calm, medium или intense; при некорректном значении выбрасывает ValueError.
    """
    if value is None:
        return "medium"

    value_str = str(value).strip().lower()

    if value_str in INTENSITY_ALIASES:
        return INTENSITY_ALIASES[value_str]

    raise ValueError(
        f"Некорректная интенсивность: {value}. "
        f"Допустимые варианты: 'calm', 'medium', 'intense'."
    )


def normalize_coverage(value):
    """
    Что делает: нормализует режим пространственного охвата маршрута.
    Параметры: value — строковый вариант охвата или None.
    Возвращает: compact, balanced или wide; при некорректном значении выбрасывает ValueError.
    """
    if value is None:
        return "balanced"

    value_str = str(value).strip().lower()

    if value_str in COVERAGE_ALIASES:
        return COVERAGE_ALIASES[value_str]

    raise ValueError(
        f"Некорректный тип охвата: {value}. "
        f"Допустимые варианты: 'compact', 'balanced', 'wide'."
    )


def build_user_preferences(user_input=None):
    """
    Переводит пользовательский ввод с сайта в численные параметры fitness-функции.

    Ожидаемый user_input:

    {
        "time_limit": "5h",
        "intensity": "medium",
        "coverage": "balanced",
        "categories": ["museum", "historic_site"],
        "budget": 5000
    }
    Параметры: user_input — словарь пользовательских настроек или None.
    Возвращает: нормализованный словарь параметров fitness-функции.
    """

    if user_input is None:
        user_input = {}

    max_time_min = normalize_time_limit(user_input.get("time_limit", "5h"))
    intensity = normalize_intensity(user_input.get("intensity", "medium"))
    coverage = normalize_coverage(user_input.get("coverage", "balanced"))

    selected_categories = normalize_categories(user_input.get("categories", []))

    max_cost = float(user_input.get("budget", user_input.get("max_cost", 5000)))

    # Проверяем допустимость сочетания лимита времени и интенсивности
    key = (max_time_min, intensity)

    if key not in INTENSITY_POINTS_CONFIG:
        raise ValueError(
            f"Недопустимое сочетание времени и интенсивности: "
            f"time_limit={max_time_min}, intensity={intensity}. "
            f"Спокойный маршрут доступен только для 2-5 часов, "
            f"насыщенный маршрут доступен только для полного дня."
        )

    target_min_points, target_max_points = INTENSITY_POINTS_CONFIG[key]

    # Чем больше времени, тем естественно шире может быть маршрут
    if max_time_min <= 120:
        time_scale = 0.75
    elif max_time_min <= 300:
        time_scale = 1.0
    else:
        time_scale = 1.45

    # Настройки компактности / охвата
    if coverage == "compact":
        min_route_diameter_m = 0
        max_route_diameter_m = 900 * time_scale
        diameter_under_penalty_weight = 0
        diameter_over_penalty_weight = 70
        close_pair_penalty_weight = 0.5
        max_jump_time_min = 22

    elif coverage == "balanced":
        min_route_diameter_m = 900 * time_scale
        max_route_diameter_m = 3500 * time_scale
        diameter_under_penalty_weight = 40
        diameter_over_penalty_weight = 10
        close_pair_penalty_weight = 6
        max_jump_time_min = 40

    else:
        min_route_diameter_m = 1800 * time_scale
        max_route_diameter_m = None
        diameter_under_penalty_weight = 75
        diameter_over_penalty_weight = 0
        close_pair_penalty_weight = 12
        max_jump_time_min = 70

    # Вес категорий
    category_weights = {cat: 1.0 for cat in SUPPORTED_MODEL_CATEGORIES}

    if selected_categories:
        # Не выбранные категории не запрещаем, но делаем менее привлекательными
        for cat in category_weights:
            category_weights[cat] = 0.88

        # Выбранные категории усиливаем
        for cat in selected_categories:
            category_weights[cat] = 1.28

        # Если выбрана еда, усиливаем обе food-категории
        if "cafe" in selected_categories or "restaurant" in selected_categories:
            category_weights["cafe"] = 1.22
            category_weights["restaurant"] = 1.22

        # Если пользователь выбрал 1-2 категории, маршрут может быть более тематическим
        if len(selected_categories) <= 2:
            max_category_ratio = 0.70
            category_diversity_penalty_weight = 20
        else:
            max_category_ratio = 0.60
            category_diversity_penalty_weight = 35

    else:
        max_category_ratio = 0.45
        category_diversity_penalty_weight = 50

    food_selected = (
        "cafe" in selected_categories
        or "restaurant" in selected_categories
    )

    if food_selected:
        unnecessary_food_penalty_weight = 8
        consecutive_food_pair_penalty_weight = 12
        missed_food_urgency_penalty_weight = 25
    else:
        unnecessary_food_penalty_weight = 25
        consecutive_food_pair_penalty_weight = 35
        missed_food_urgency_penalty_weight = 40

    user_preferences = {
        "max_time_min": float(max_time_min),
        "max_cost": float(max_cost),

        "intensity": intensity,
        "coverage": coverage,
        "selected_categories": selected_categories,

        "target_route_points_min": int(target_min_points),
        "target_route_points_max": int(target_max_points),
        "route_length_penalty_weight": 80,

        "quality_weight": 10,
        "coverage_weight": 4,
        "time_weight": 0.03,
        "satiety_weight": 3,

        "category_weights": category_weights,
        "max_category_ratio": float(max_category_ratio),
        "category_diversity_penalty_weight": float(category_diversity_penalty_weight),

        "min_route_diameter_m": None if min_route_diameter_m is None else float(min_route_diameter_m),
        "max_route_diameter_m": None if max_route_diameter_m is None else float(max_route_diameter_m),
        "diameter_under_penalty_weight": float(diameter_under_penalty_weight),
        "diameter_over_penalty_weight": float(diameter_over_penalty_weight),
        "close_pair_penalty_weight": float(close_pair_penalty_weight),

        "max_jump_time_min": float(max_jump_time_min),

        "missed_food_urgency_penalty_weight": float(missed_food_urgency_penalty_weight),
        "critical_hunger_penalty_weight": 180,
        "consecutive_food_pair_penalty_weight": float(consecutive_food_pair_penalty_weight),
        "unnecessary_food_penalty_weight": float(unnecessary_food_penalty_weight)
    }

    return user_preferences


def get_effective_satiety_cfg(base_cfg=None, user_preferences=None):
    """
    Создает копию TOURIST_SATIETY_CFG, но с учетом пользовательской интенсивности.
    Это нужно, потому что часть алгоритмов использует min_route_points/max_route_points из cfg.
    Параметры: base_cfg — базовый словарь сытости; user_preferences — нормализованные предпочтения.
    Возвращает: новый словарь настроек сытости.
    """

    if base_cfg is None:
        base_cfg = TOURIST_SATIETY_CFG

    cfg = dict(base_cfg)

    if user_preferences is not None:
        cfg["min_route_points"] = int(user_preferences["target_route_points_min"])
        cfg["max_route_points"] = int(user_preferences["target_route_points_max"])

    return cfg


def make_weighted_ratings_array(ratings, categories, user_preferences=None):
    """
    Возвращает массив рейтингов с учетом пользовательских предпочтений по категориям.
    Исходные рейтинги не меняет.
    Параметры: ratings — исходные рейтинги; categories — категории объектов; user_preferences — предпочтения.
    Возвращает: новый numpy-массив взвешенных рейтингов.
    """

    ratings = np.asarray(ratings, dtype=float)

    if user_preferences is None:
        return ratings.copy()

    category_weights = user_preferences.get("category_weights", {})

    weighted = ratings.copy()

    for i, cat in enumerate(categories):
        cat = str(cat)
        weight = float(category_weights.get(cat, 1.0))
        weighted[i] = weighted[i] * weight

    return weighted


def get_category_weight(category, user_preferences=None):
    """
    Что делает: возвращает вес категории из пользовательских предпочтений.
        Параметры: category — категория; user_preferences — словарь предпочтений.
        Возвращает: числовой вес категории.
    """
    if user_preferences is None:
        return 1.0

    return float(
        user_preferences
        .get("category_weights", {})
        .get(str(category), 1.0)
    )

def evaluate_tourist_route(
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
    Единая оценка маршрута.

    Здесь:
    1. проверяются жесткие ограничения по времени и бюджету;
    2. считаются сытость и food-статистика;
    3. добавляются статистики категорий;
    4. rating для score пересчитывается с учетом user_preferences.
    Параметры: route, walk_time_matrix, visit_times, costs, ratings, categories, max_time_min, max_cost и настройки.
    Возвращает: пару score и stats.
    """

    if satiety_cfg is None:
        satiety_cfg = TOURIST_SATIETY_CFG

    route = list(map(int, route))

    if len(route) == 0:
        return -1e9, {}

    # Сырые статистики для вывода
    raw_base_stats = route_stats(
        route=route,
        walk_time_matrix=walk_time_matrix,
        visit_times=visit_times,
        costs=costs,
        ratings=ratings
    )

    # Если route_stats не содержит spatial-метрики, добавляем их здесь
    if "route_diameter_m" not in raw_base_stats:
        if len(route) >= 2:
            sub = walk_time_matrix[np.ix_(route, route)]
            raw_base_stats["route_diameter_m"] = float(np.max(sub) * (5000 / 60.0))
        else:
            raw_base_stats["route_diameter_m"] = 0.0

    if "close_pair_count" not in raw_base_stats:
        close_pair_count = 0

        if len(route) >= 2:
            # 150 метров при скорости 5000 м/час = 1.8 минуты
            close_time_threshold = 150 / (5000 / 60.0)

            for a in range(len(route)):
                for b in range(a + 1, len(route)):
                    if float(walk_time_matrix[route[a], route[b]]) <= close_time_threshold:
                        close_pair_count += 1

        raw_base_stats["close_pair_count"] = int(close_pair_count)

    total_time_min = float(raw_base_stats.get("total_time_min", 0.0))
    total_cost = float(raw_base_stats.get("total_cost", raw_base_stats.get("total_cost_rub", 0.0)))

    # Жесткие ограничения
    if total_time_min > float(max_time_min):
        return -1e9, raw_base_stats

    if total_cost > float(max_cost):
        return -1e9, raw_base_stats

    satiety_stats = compute_satiety_route_stats(
        route=route,
        walk_time_matrix=walk_time_matrix,
        visit_times=visit_times,
        categories=categories,
        cfg=satiety_cfg
    )

    # Категорийная статистика
    route_categories = [str(categories[i]) for i in route]
    category_counts = dict(Counter(route_categories))

    if len(route_categories) > 0:
        max_category_ratio = max(category_counts.values()) / len(route_categories)
    else:
        max_category_ratio = 0.0

    satiety_stats["category_counts"] = category_counts
    satiety_stats["max_category_ratio"] = float(max_category_ratio)

    # Проверка минимального числа food-точек
    food_visits = int(
        satiety_stats.get("food_visit_count", 0)
    )

    if food_visits < int(min_food_visits):
        merged_stats = merge_base_and_satiety_stats(raw_base_stats, satiety_stats)
        return -1e9, merged_stats

    # Для score используем рейтинги с учетом категорийных предпочтений
    weighted_ratings = make_weighted_ratings_array(
        ratings=ratings,
        categories=categories,
        user_preferences=user_preferences
    )

    score_base_stats = route_stats(
        route=route,
        walk_time_matrix=walk_time_matrix,
        visit_times=visit_times,
        costs=costs,
        ratings=weighted_ratings
    )

    # Переносим spatial-метрики из raw_base_stats
    score_base_stats["route_diameter_m"] = raw_base_stats.get("route_diameter_m", 0.0)
    score_base_stats["close_pair_count"] = raw_base_stats.get("close_pair_count", 0)

    score = satiety_adjusted_route_score(
        base_stats=score_base_stats,
        satiety_stats=satiety_stats,
        cfg=satiety_cfg,
        user_preferences=user_preferences
    )

    merged_stats = merge_base_and_satiety_stats(raw_base_stats, satiety_stats)

    merged_stats["score"] = float(score)
    merged_stats["weighted_avg_rating"] = float(score_base_stats.get("avg_rating", 0.0))
    merged_stats["user_preferences"] = user_preferences
    merged_stats["route_order"] = route
    merged_stats["points_count"] = len(route)
    return float(score), merged_stats
