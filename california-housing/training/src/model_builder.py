"""Builds every experiment: the deployed models (XGBoost + the MLP from ../model/model.py) and the
baselines that are only evaluated (LinearRegression, RandomForest)."""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import model as M

EXPORTED = ("xgboost", "mlp")                       # written to model/ (the Space can serve both)
BASELINES = ("linear_regression", "random_forest")  # evaluated for the comparison only


def build_scaler() -> StandardScaler:
    """Standardization for the MLP inputs (fit on the train split only)."""
    return StandardScaler()


def build_linear_regression() -> LinearRegression:
    return LinearRegression()


def build_random_forest(cfg) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=cfg.rf_n_estimators, n_jobs=cfg.n_jobs, random_state=cfg.seed)


def build_xgboost(cfg) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=cfg.xgb_n_estimators,
        learning_rate=cfg.xgb_lr,
        max_depth=cfg.xgb_max_depth,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample,
        early_stopping_rounds=cfg.xgb_early_stopping_rounds,
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=cfg.n_jobs,
        random_state=cfg.seed,
    )


def build_mlp(cfg) -> "M.HousingMLP":
    return M.HousingMLP(in_features=len(M.FEATURES), hidden_units=list(cfg.hidden_units), dropout=cfg.dropout)


def build_all(cfg) -> dict:
    """{name: untrained model} for every experiment, in training order."""
    return {
        "linear_regression": build_linear_regression(),
        "random_forest": build_random_forest(cfg),
        "xgboost": build_xgboost(cfg),
        "mlp": build_mlp(cfg),
    }


def describe(name: str, cfg) -> str:
    return {
        "linear_regression": "ordinary least squares on the 11 features",
        "random_forest": f"RandomForestRegressor, {cfg.rf_n_estimators} trees",
        "xgboost": (f"XGBRegressor hist, depth {cfg.xgb_max_depth}, lr {cfg.xgb_lr}, subsample {cfg.xgb_subsample}, "
                    f"colsample {cfg.xgb_colsample}, <= {cfg.xgb_n_estimators} rounds, early stopping "
                    f"{cfg.xgb_early_stopping_rounds}"),
        "mlp": (f"PyTorch MLP {len(M.FEATURES)}-{'-'.join(map(str, cfg.hidden_units))}-1, ReLU, dropout "
                f"{cfg.dropout}, AdamW lr {cfg.lr}, wd {cfg.weight_decay}, batch {cfg.batch_size}"),
    }[name]
