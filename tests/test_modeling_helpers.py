"""Тесты для helpers моделирования в notebooks/_common.py.

Проверяют чистоту реализации (без leakage) и корректность сборки артефактов.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Добавляем корень проекта в sys.path чтобы импорт notebooks._common работал
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class TestChronologicalSplit:
    def test_splits_80_20_by_default(self):
        from notebooks._common import chronological_split
        df = pd.DataFrame({"x": range(100)})
        tr, te = chronological_split(df)
        assert len(tr) == 80
        assert len(te) == 20

    def test_split_preserves_order(self):
        from notebooks._common import chronological_split
        df = pd.DataFrame({"x": range(50)})
        tr, te = chronological_split(df, test_frac=0.2)
        assert tr["x"].iloc[0] == 0
        assert tr["x"].iloc[-1] == 39
        assert te["x"].iloc[0] == 40
        assert te["x"].iloc[-1] == 49

    def test_rejects_invalid_frac(self):
        from notebooks._common import chronological_split
        df = pd.DataFrame({"x": range(50)})
        with pytest.raises(ValueError):
            chronological_split(df, test_frac=0.0)
        with pytest.raises(ValueError):
            chronological_split(df, test_frac=1.0)

    def test_rejects_too_small_df(self):
        from notebooks._common import chronological_split
        df = pd.DataFrame({"x": range(3)})
        with pytest.raises(ValueError):
            chronological_split(df)

    def test_rejects_unsorted_date_column(self):
        from notebooks._common import chronological_split
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02",
                                    "2020-01-05", "2020-01-04"] * 4),
            "x": range(20),
        })
        with pytest.raises(ValueError, match="отсортирован"):
            chronological_split(df)


class TestSelectFeaturesClean:
    def test_returns_list_of_strings(self, sample_df):
        from notebooks._common import select_features_clean
        df_train = sample_df.head(20).copy()
        # operational_whitelist=[] для legacy-поведения: только top-N, без расширения
        names = select_features_clean(df_train, target="stickiness", max_features=5,
                                      operational_whitelist=[])
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert len(names) <= 5

    def test_excludes_meta_and_targets(self, sample_df):
        from notebooks._common import select_features_clean
        df_train = sample_df.head(20).copy()
        names = select_features_clean(df_train, target="stickiness", max_features=15)
        forbidden = {"date", "app_id", "category", "stickiness", "rating", "retention_d7"}
        assert not (set(names) & forbidden)

    def test_drops_all_nan_columns(self, sample_df):
        """Колонки полностью-NaN (как store у f2) должны быть отброшены."""
        from notebooks._common import select_features_clean
        df_f2 = sample_df[sample_df["app_id"] == "com.google.flood2"].copy()
        # Зануляем store-колонки (как в реальных данных f2)
        for col in ["rating", "reviews_count", "positive", "negative",
                    "positive_ratio", "sentiment_score", "reviews_30d"]:
            df_f2[col] = np.nan
        names = select_features_clean(df_f2, target="stickiness", max_features=15)
        assert "rating" not in names
        assert "reviews_count" not in names

    def test_raises_on_too_few_rows(self, sample_df):
        from notebooks._common import select_features_clean
        df_tiny = sample_df.head(5).copy()
        with pytest.raises(ValueError):
            select_features_clean(df_tiny, target="stickiness", max_features=5)


class TestBuildXY:
    def test_returns_X_y_medians(self, sample_df):
        from notebooks._common import build_xy
        df = sample_df.head(15).copy()
        feats = ["dau", "mau", "stickiness"]
        X, y, medians = build_xy(df, target="retention_d7",
                                 feature_names=feats)
        assert X.shape == (15, 3)
        assert y.shape == (15,)
        assert set(medians.keys()) == set(feats)

    def test_test_uses_train_medians(self, sample_df):
        from notebooks._common import build_xy
        df_tr = sample_df.head(15).copy()
        df_te = sample_df.tail(10).copy()
        feats = ["dau", "stickiness"]
        _, _, medians_tr = build_xy(df_tr, target="retention_d7",
                                    feature_names=feats)
        # Поломаем тестовые значения NaN, проверим что fillna взял медианы train
        df_te.loc[df_te.index[0], "dau"] = np.nan
        X_te, _, medians_used = build_xy(df_te, target="retention_d7",
                                         feature_names=feats,
                                         train_medians=medians_tr)
        # Первая строка теста должна быть заполнена медианой train
        assert X_te[0, 0] == pytest.approx(medians_tr["dau"])
        assert medians_used == medians_tr

    def test_drops_nan_target_rows(self, sample_df):
        from notebooks._common import build_xy
        df = sample_df.head(15).copy()
        df.loc[df.index[:3], "stickiness"] = np.nan
        X, y, _ = build_xy(df, target="stickiness", feature_names=["dau"])
        assert X.shape == (12, 1)
        assert y.shape == (12,)


class TestFitPredictEvaluate:
    def test_returns_expected_keys(self, sample_X, sample_y):
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold
        from notebooks._common import fit_predict_evaluate
        sp = 18
        X_tr, X_te = sample_X[:sp], sample_X[sp:]
        y_tr, y_te = sample_y[:sp], sample_y[sp:]
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        result = fit_predict_evaluate(
            lambda: Ridge(alpha=1.0),
            X_tr, y_tr, X_te, y_te,
            target_name="synthetic", cv=cv,
        )
        for k in ["model", "cv_rmse_scores", "cv_rmse_mean", "cv_rmse_std",
                  "test_metrics", "y_pred", "residuals"]:
            assert k in result
        assert len(result["cv_rmse_scores"]) == 3
        assert result["test_metrics"]["rmse"] >= 0
        assert len(result["y_pred"]) == len(y_te)


class TestUpdateFinalConfig:
    def test_creates_file_if_missing(self, tmp_path):
        from notebooks._common import update_final_config
        cfg_path = tmp_path / "final_config.json"
        update_final_config("f1", "stickiness",
                            {"model": "NW", "test_metrics": {"rmse": 0.01}},
                            path=cfg_path)
        assert cfg_path.exists()
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert data["schema_version"] == "1.0"
        assert data["f1"]["stickiness"]["model"] == "NW"

    def test_appends_without_overwriting(self, tmp_path):
        from notebooks._common import update_final_config
        cfg_path = tmp_path / "final_config.json"
        update_final_config("f1", "stickiness", {"model": "NW"}, path=cfg_path)
        update_final_config("f1", "dau", {"model": "Ridge"}, path=cfg_path)
        update_final_config("f2", "stickiness", {"model": "XGB"}, path=cfg_path)
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert data["f1"]["stickiness"]["model"] == "NW"
        assert data["f1"]["dau"]["model"] == "Ridge"
        assert data["f2"]["stickiness"]["model"] == "XGB"

    def test_updates_existing_target(self, tmp_path):
        from notebooks._common import update_final_config
        cfg_path = tmp_path / "final_config.json"
        update_final_config("f1", "stickiness", {"model": "OLD"}, path=cfg_path)
        update_final_config("f1", "stickiness", {"model": "NEW"}, path=cfg_path)
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert data["f1"]["stickiness"]["model"] == "NEW"


class TestSelectFeaturesCleanWhitelist:
    """Тесты для whitelist-расширения select_features_clean (Variant Z)."""

    def test_default_whitelist_constant_exists(self):
        """DEFAULT_OPERATIONAL_WHITELIST имеет ожидаемое содержимое."""
        from notebooks._common import DEFAULT_OPERATIONAL_WHITELIST
        assert isinstance(DEFAULT_OPERATIONAL_WHITELIST, list)
        assert len(DEFAULT_OPERATIONAL_WHITELIST) >= 5
        # Ключевые операционные метрики должны быть в whitelist
        expected_subset = {"crash_rate", "onboarding_completion_rate",
                           "avg_session_duration"}
        assert expected_subset.issubset(set(DEFAULT_OPERATIONAL_WHITELIST))

    def test_whitelist_features_force_included(self, sample_df):
        """Whitelist признаки добавляются если прошли VIF, даже при малом max_features."""
        from notebooks._common import (
            select_features_clean, DEFAULT_OPERATIONAL_WHITELIST,
        )
        df = sample_df.head(20).copy()
        names = select_features_clean(df, target="stickiness", max_features=2)
        # Должны быть и top-2 по importance, и whitelist-признаки сверху
        whitelist_in_data = set(DEFAULT_OPERATIONAL_WHITELIST) & set(df.columns)
        # Минимум один whitelist-признак должен попасть в результат
        assert set(names) & whitelist_in_data, \
            f"Ни один whitelist-признак не попал в {names}"
        # Финальный набор крупнее max_features
        assert len(names) > 2, f"Whitelist не расширил набор: {names}"

    def test_empty_whitelist_disables_extension(self, sample_df):
        """operational_whitelist=[] возвращает legacy-поведение (только top-N)."""
        from notebooks._common import select_features_clean
        df = sample_df.head(20).copy()
        names = select_features_clean(
            df, target="stickiness", max_features=3, operational_whitelist=[]
        )
        assert len(names) <= 3, f"С пустым whitelist должно быть ≤3 признаков, получено {names}"
