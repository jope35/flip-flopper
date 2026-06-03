"""Tests for shared classifier training helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from flip_flopper.classifier_training import (
    TARGET_COLUMN,
    feature_columns_from_frame,
    normalize_model_name,
    qualified_table_name,
    resolve_effective_cv_folds,
    resolve_experiment_path,
    uc_registered_model_name,
)


def test_qualified_table_name_quotes_backticks() -> None:
    assert qualified_table_name("cat", "sch`ema", "tbl") == "`cat`.`sch``ema`.`tbl`"


def test_normalize_model_name_rejects_empty() -> None:
    with pytest.raises(ValueError, match="registered_model_suffix"):
        normalize_model_name("!!!")


def test_resolve_experiment_path() -> None:
    assert resolve_experiment_path("/Users/exp") == "/Users/exp"
    assert resolve_experiment_path("my_exp") == "/Shared/my_exp"


def test_uc_registered_model_name() -> None:
    name = uc_registered_model_name("c", "s", "Flip-Flopper", "a")
    assert name == "c.s.flip_flopper_a"


def test_feature_columns_from_frame_sorted_excludes_label() -> None:
    frame = pd.DataFrame({"b": [1], "a": [2], TARGET_COLUMN: [0]})
    assert feature_columns_from_frame(frame) == ["a", "b"]


def test_resolve_effective_cv_folds_caps_by_minority_class() -> None:
    labels = pd.Series([0] * 3 + [1] * 2)
    assert resolve_effective_cv_folds(5, labels) == 2


def test_resolve_effective_cv_folds_rejects_too_few_minority_samples() -> None:
    labels = pd.Series([0, 1, 1, 1])
    with pytest.raises(ValueError, match="at least 2 samples"):
        resolve_effective_cv_folds(5, labels)
