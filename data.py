"""
Модуль data.py: чтение SQLite-базы туристических объектов.
Возвращает DataFrame, готовый для работы алгоритма.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd


def preprocess_tourist_objects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Подготавливает таблицу объектов для алгоритма маршрутизации.
    """

    df = df.copy()

    # Приведение строковых полей
    df["category"] = df["category"].str.strip()
    df["name"] = df["name"].str.strip()

    # Приведение числовых полей
    numeric_cols = [
        "rating",
        "reviews_count",
        "price_rub",
        "visit_duration_min",
        "x",
        "y",
        "lat",
        "lon",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    # Совместимость с существующим кодом алгоритма
    df["visit_time_min"] = df["visit_duration_min"]

    return df


def load_tourist_objects(
    db_path: str | Path,
    table_name: str = "tourist_objects",
) -> pd.DataFrame:
    """
    Загружает туристические объекты из SQLite.
    """

    with sqlite3.connect(Path(db_path)) as conn:
        tourist_objects = pd.read_sql_query(
            f"SELECT * FROM {table_name}",
            conn,
        )

    return preprocess_tourist_objects(tourist_objects)