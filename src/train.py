"""
Spaceship Titanic — Advanced Neural Network Training Pipeline
TensorFlow/Keras Ensemble + Optuna HPO + MLflow Tracking
"""

import os
import warnings
import logging
import argparse

import numpy as np
import pandas as pd
import mlflow
import mlflow.tensorflow
import optuna
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from preprocessing import build_preprocessor, engineer_features
from model import build_mlp, build_wide_deep

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ─── Data Loading ──────────────────────────────────────────────────────────────

def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    log.info(f"Train: {train.shape} | Test: {test.shape}")
    return train, test


# ─── Optuna Objective ──────────────────────────────────────────────────────────

def make_objective(X, y, preprocessor):
    def objective(trial):
        params = {
            "n_layers":    trial.suggest_int("n_layers", 2, 6),
            "units":       trial.suggest_categorical("units", [64, 128, 256, 512]),
            "dropout":     trial.suggest_float("dropout", 0.1, 0.5),
            "lr":          trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size":  trial.suggest_categorical("batch_size", [128, 256, 512]),
            "activation":  trial.suggest_categorical("activation", ["relu", "swish", "gelu"]),
            "l2_reg":      trial.suggest_float("l2_reg", 1e-5, 1e-2, log=True),
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            X_tr_t  = preprocessor.fit_transform(X_tr)
            X_val_t = preprocessor.transform(X_val)

            model = build_mlp(X_tr_t.shape[1], params)
            model.fit(
                X_tr_t, y_tr,
                validation_data=(X_val_t, y_val),
                epochs=30,
                batch_size=params["batch_size"],
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                verbose=0,
            )
            preds = model.predict(X_val_t, verbose=0).ravel()
            scores.append(roc_auc_score(y_val, preds))

        return np.mean(scores)
    return objective


# ─── Ensemble Training ─────────────────────────────────────────────────────────

def train_ensemble(X, y, X_test, best_params, preprocessor, n_folds=5, n_models=2):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds   = np.zeros(len(y))
    test_preds  = np.zeros((len(X_test), n_folds * n_models))
    col = 0

    model_builders = [build_mlp, build_wide_deep]

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        prep = build_preprocessor()
        X_tr_t  = prep.fit_transform(X_tr)
        X_val_t = prep.transform(X_val)
        X_tst_t = prep.transform(X_test)

        fold_oof = np.zeros(len(val_idx))

        for builder in model_builders:
            model = builder(X_tr_t.shape[1], best_params)
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0),
            ]
            model.fit(
                X_tr_t, y_tr,
                validation_data=(X_val_t, y_val),
                epochs=100,
                batch_size=best_params["batch_size"],
                callbacks=callbacks,
                verbose=0,
            )
            val_p  = model.predict(X_val_t, verbose=0).ravel()
            test_p = model.predict(X_tst_t, verbose=0).ravel()
            fold_oof += val_p / n_models
            test_preds[:, col] = test_p
            col += 1

        oof_preds[val_idx] = fold_oof
        auc = roc_auc_score(y_val, fold_oof)
        acc = accuracy_score(y_val, (fold_oof > 0.5).astype(int))
        log.info(f"  Fold {fold+1}/{n_folds} → AUC={auc:.4f}  ACC={acc:.4f}")

    final_test = test_preds[:, :col].mean(axis=1)
    oof_auc = roc_auc_score(y, oof_preds)
    oof_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
    log.info(f"\n✅ OOF AUC={oof_auc:.4f}  OOF ACC={oof_acc:.4f}")
    return oof_preds, final_test, oof_auc, oof_acc


# ─── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    mlflow.set_experiment("spaceship-titanic")

    with mlflow.start_run(run_name="ensemble_keras"):
        train_df, test_df = load_data()

        # Feature engineering
        train_df = engineer_features(train_df)
        test_df  = engineer_features(test_df)

        target     = "Transported"
        drop_cols  = [target, "PassengerId", "Name"]
        X          = train_df.drop(columns=drop_cols, errors="ignore")
        y          = train_df[target].astype(int).values
        X_test     = test_df.drop(columns=["PassengerId", "Name"], errors="ignore")

        preprocessor = build_preprocessor()
        X_np      = X.values
        X_test_np = X_test.values

        # ── HPO ────────────────────────────────────────────────────────
        log.info("🔍 Running Optuna HPO …")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(
            make_objective(X_np, y, build_preprocessor()),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )
        best_params = study.best_params
        log.info(f"Best params: {best_params}")
        mlflow.log_params(best_params)
        mlflow.log_metric("hpo_best_auc", study.best_value)

        # ── Ensemble ───────────────────────────────────────────────────
        log.info("🏋️  Training ensemble …")
        oof_preds, test_preds, oof_auc, oof_acc = train_ensemble(
            X_np, y, X_test_np, best_params, preprocessor,
            n_folds=args.n_folds,
        )

        mlflow.log_metric("oof_auc", oof_auc)
        mlflow.log_metric("oof_acc", oof_acc)

        # ── Submission ─────────────────────────────────────────────────
        sub = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Transported": (test_preds > 0.5),
        })
        sub_path = os.path.join(MODEL_DIR, "submission.csv")
        sub.to_csv(sub_path, index=False)
        mlflow.log_artifact(sub_path)
        log.info(f"📄 Submission saved → {sub_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=30, help="Optuna trials")
    parser.add_argument("--n_folds",  type=int, default=5,  help="CV folds")
    main(parser.parse_args())
