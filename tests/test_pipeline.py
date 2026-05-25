"""
Unit tests for preprocessing and model building.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from preprocessing import engineer_features, build_preprocessor
from model import build_mlp, build_wide_deep


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "PassengerId":   ["0001_01", "0001_02", "0002_01"],
        "HomePlanet":    ["Earth", "Mars", "Europa"],
        "CryoSleep":     [True, False, True],
        "Cabin":         ["B/0/P", "F/1/S", "A/2/P"],
        "Destination":   ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"],
        "Age":           [25.0, 31.0, 8.0],
        "VIP":           [False, True, False],
        "RoomService":   [0.0, 100.0, 0.0],
        "FoodCourt":     [0.0, 50.0, 0.0],
        "ShoppingMall":  [0.0, 200.0, 0.0],
        "Spa":           [0.0, 0.0, 0.0],
        "VRDeck":        [0.0, 75.0, 0.0],
        "Name":          ["Alice A", "Bob B", "Charlie C"],
        "Transported":   [True, False, True],
    })

@pytest.fixture
def default_params():
    return {
        "n_layers": 2, "units": 64, "dropout": 0.2,
        "lr": 1e-3, "batch_size": 32,
        "activation": "relu", "l2_reg": 1e-4,
    }


# ─── Feature Engineering Tests ─────────────────────────────────────────────────

def test_engineer_features_shape(sample_df):
    out = engineer_features(sample_df)
    assert "Deck" in out.columns
    assert "Side" in out.columns
    assert "Cabin" not in out.columns
    assert "Total_spend" in out.columns
    assert "Group_size" in out.columns
    assert "Is_alone" in out.columns

def test_engineer_features_no_cabin_col(sample_df):
    out = engineer_features(sample_df)
    assert "Cabin" not in out.columns

def test_total_spend_calculation(sample_df):
    out = engineer_features(sample_df)
    # Row 1: 100+50+200+0+75 = 425
    assert out.loc[1, "Total_spend"] == pytest.approx(425.0)

def test_is_alone_flag(sample_df):
    out = engineer_features(sample_df)
    # PassengerId 0002_01 has only 1 member → is_alone=1
    assert out.loc[2, "Is_alone"] == 1
    # PassengerId 0001_xx has 2 members → is_alone=0
    assert out.loc[0, "Is_alone"] == 0


# ─── Preprocessor Tests ────────────────────────────────────────────────────────

def test_preprocessor_fit_transform(sample_df):
    df = engineer_features(sample_df)
    drop_cols = ["Transported", "PassengerId", "Name"]
    X = df.drop(columns=drop_cols, errors="ignore")

    prep = build_preprocessor()
    X_t = prep.fit_transform(X)
    assert X_t.shape[0] == 3
    assert X_t.shape[1] > 0

def test_preprocessor_no_nan_after_transform(sample_df):
    df = engineer_features(sample_df)
    X = df.drop(columns=["Transported", "PassengerId", "Name"], errors="ignore")
    prep = build_preprocessor()
    X_t = prep.fit_transform(X)
    assert not np.isnan(X_t).any()


# ─── Model Tests ───────────────────────────────────────────────────────────────

def test_mlp_output_shape(default_params):
    model = build_mlp(input_dim=20, params=default_params)
    x = np.random.randn(10, 20).astype("float32")
    preds = model.predict(x, verbose=0)
    assert preds.shape == (10, 1)
    assert (preds >= 0).all() and (preds <= 1).all()

def test_wide_deep_output_shape(default_params):
    model = build_wide_deep(input_dim=20, params=default_params)
    x = np.random.randn(10, 20).astype("float32")
    preds = model.predict(x, verbose=0)
    assert preds.shape == (10, 1)
    assert (preds >= 0).all() and (preds <= 1).all()

def test_mlp_training_step(default_params):
    model = build_mlp(input_dim=10, params=default_params)
    X = np.random.randn(50, 10).astype("float32")
    y = np.random.randint(0, 2, 50).astype("float32")
    history = model.fit(X, y, epochs=2, batch_size=16, verbose=0)
    assert "loss" in history.history
    assert len(history.history["loss"]) == 2
