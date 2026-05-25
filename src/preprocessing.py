"""
Feature engineering & sklearn preprocessing pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


# ─── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Cabin → Deck / Cabin_num / Side
    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"]       = cabin[0]
    df["Cabin_num"]  = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"]       = cabin[2]
    df.drop(columns=["Cabin"], inplace=True)

    # Group from PassengerId
    df["Group"]      = df["PassengerId"].str.split("_").str[0]
    df["Group_size"] = df.groupby("Group")["Group"].transform("count")
    df["Is_alone"]   = (df["Group_size"] == 1).astype(int)
    df.drop(columns=["Group"], inplace=True)

    # Spending features
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["Total_spend"]     = df[spend_cols].sum(axis=1)
    df["Spend_per_room"]  = df["RoomService"] / (df["Total_spend"] + 1)
    df["Any_spend"]       = (df["Total_spend"] > 0).astype(int)
    df["Log_total_spend"] = np.log1p(df["Total_spend"])

    # Age bins
    df["Age_bin"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 200],
        labels=[0, 1, 2, 3, 4],
    ).astype("float32")

    # CryoSleep & VIP interactions
    df["CryoSleep"] = df["CryoSleep"].map({True: 1, False: 0, "True": 1, "False": 0})
    df["VIP"]       = df["VIP"].map({True: 1, False: 0, "True": 1, "False": 0})
    df["CryoSleep_spend"] = df["CryoSleep"] * df["Total_spend"]

    return df


# ─── Preprocessing Pipeline ────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
        "Total_spend", "Spend_per_room", "Log_total_spend", "Any_spend",
        "Cabin_num", "Group_size", "Is_alone", "CryoSleep_spend", "Age_bin",
        "CryoSleep", "VIP",
    ]
    categorical_features = ["HomePlanet", "Destination", "Deck", "Side"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor
