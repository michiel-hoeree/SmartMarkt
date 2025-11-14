"""Data loading helpers for the supermarket app."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from sklearn_recommend import SklearnRecommender

CSV_FILENAME = "Supermart-Grocery-Sales-Retail-Analytics-Dataset-Euro.csv"


_recommender: SklearnRecommender | None = None

# Public module-level variables expected by the rest of the app
klanten: List[dict] = []
producten: List[dict] = []
aankoopgeschiedenis: Dict[str, List[List[str]]] = {}


def _get_csv_path() -> Path:
    return Path(__file__).resolve().with_name(CSV_FILENAME)


def _initialise_recommender() -> SklearnRecommender:
    global _recommender

    if _recommender is not None:
        return _recommender

    csv_path = _get_csv_path()
    recommender = SklearnRecommender(str(csv_path))

    # Try loading a previously trained model first to avoid recomputing.
    if not recommender.load_model():
        # Fall back to reading the CSV directly.
        recommender.load_csv_data()

        # Persist a model so subsequent boots use the cached data.
        try:
            recommender.train_model()
        except Exception as exc:  # pragma: no cover - logging only
            # Training is not critical for the dashboard to render; log and continue.
            print(f"Warning: kon sklearn-model niet trainen: {exc}")

    _recommender = recommender
    return _recommender


def _ensure_data_loaded() -> None:
    global klanten, producten, aankoopgeschiedenis

    if klanten and producten and aankoopgeschiedenis:
        return

    recommender = _initialise_recommender()

    # When we load from a pickled model the attributes are already filled.
    if not recommender.klanten or not recommender.producten:
        recommender.load_csv_data()

    klanten = recommender.klanten
    producten = recommender.producten
    aankoopgeschiedenis = recommender.aankoopgeschiedenis


def get_recommender() -> SklearnRecommender:
    """Expose the shared recommender instance used across the app."""

    _ensure_data_loaded()
    # _initialise_recommender() already returns a SklearnRecommender
    return _initialise_recommender()


def reload_data() -> None:
    """Force a reload of data from the CSV and retrain the model."""

    global _recommender, klanten, producten, aankoopgeschiedenis

    csv_path = _get_csv_path()
    _recommender = SklearnRecommender(str(csv_path))
    _recommender.load_csv_data()
    _recommender.train_model()

    klanten = _recommender.klanten
    producten = _recommender.producten
    aankoopgeschiedenis = _recommender.aankoopgeschiedenis


# Ensure data is ready as soon as the module is imported.
_ensure_data_loaded()


