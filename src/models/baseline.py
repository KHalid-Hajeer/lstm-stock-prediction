from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from utils import clean_series, align_X_y, as_series


# Helpers

def pearson_corr(y_true: pd.Series, y_pred: pd.Series, raise_on_degenerate: bool = False) -> float:
    """Safe Pearson correlation (returns NaN or optional error if degenerate).
    
    Degenerate cases:
        - Fewer than 2 aligned points after cleaning/alignment.
        - Zero variance in either series.

    Args:
        y_true (pd.Series): True numeric series.
        y_pred (pd.Series): Predicted numeric series.
        raise_on_degenerate (bool): Whether to raise ValueError if degenerate.

    Returns:
        float: Pearson correlation coefficient or NaN.

    Raises:
        ValueError: If raise_on_degenerate is True and degenerate.
    
    """
    clean_true = clean_series(pd.Series(y_true))
    clean_pred = clean_series(pd.Series(y_pred))
    clean_true, clean_pred = clean_true.align(clean_pred, join = "inner")

    # Degeneracy checks
    if len(clean_true) < 2:
        if raise_on_degenerate:
            raise ValueError("Fewer than 2 aligned points after cleaning/alignment.")
        return np.nan
    if float(clean_true.var(ddof=0)) == 0.0 or float(clean_pred.var(ddof=0)) == 0.0:
        if raise_on_degenerate:
            raise ValueError("Zero variance in either series.")
        return np.nan
    return float(clean_true.corr(clean_pred))

def _ridge_pipeline(alpha: float) -> Pipeline:
    """Standardise features (fit on training only) + Ridge regression.
    
    Args:
        alpha: Ridge penalty value.

    Returns:
        Pipeline: StandardScaler + Ridge

    Raises:
        ValueError: If alpha is not a positive finite float.
    """
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"alpha must be a positive finite float, got {alpha!r}.")
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=float(alpha), fit_intercept=True))
    ])



# Persistence baseline
def predict_persistence(y_raw_next: pd.Series) -> pd.Series:
    """Persistence baseline: use r_t to predict r_{t+1}.
    
    The target 'y_raw_next' should be the next day return aligned at time t (i.e., index t holds r_{t+1}).
    The persistence predictor at time t is r_t, which equals 'y_raw_next.shift(1)'.

    Args:
        y_raw_next (pd.Series): Next day return series aligned at time t.

    Returns:
        pd.Series: Persistence predictions aligned to the same index.
    """
    series_float64 = pd.Series(y_raw_next).astype("float64")
    return series_float64.shift(1).rename("pred_persistence")


# Ridge Baseline (validation selection by Pearson correlation)
def train_ridge_with_val(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        alphas: Iterable[float] = tuple(np.logspace(-4, 4, 25)),
        refit_on_train_val: bool = True
) -> Tuple[Pipeline, Dict[str, float], pd.Series]:
    """Train Ridge on the train window and pick alpha using validation correlation.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target (next day returns aligned to X_train).
        X_val (pd.DataFrame): Validation feature matrix.
        y_val (pd.Series): Validation target (next day returns aligned to X_val).
        alphas (Iterable[float], optional): Sequence of positive Ridge penalty values to try (recommended in log-space). Defaults to tuple(np.logspace(-4, 4, 25)).
        refit_on_train_val (bool, optional): Whether to refit the best pipeline on train+val before returning. Defaults to True.

    Returns:
        Tuple[Pipeline, Dict[str, float], pd.Series]:
            - fitted_model: Pipeline (scaler + Ridge).
            - diagnostics: dict with {'best_alpha', 'val_corr', 'val_rmse'}.
            - val_pred: Validation predictions (as Series) using the best alpha.

    Raise:
        ValueError: If alignment/cleaning empties the dataset or alpha grid invalid.
        RuntimeError: If no valid alpha yields a finite correlation.

    Notes:
        - Scaling is fit only on the data used to train the model (no leakage).
        - If 'refit_on_train_val=True', we rebuild the pipeline with the chosen alpha and fit it on the concatenated train+val slice to use later on the test set.
    """
    X_train, y_train = align_X_y(X_train, y_train, dropna_rows=True)
    X_val, y_val = align_X_y(X_val, y_val, dropna_rows=True)

    alpha_grid = [float(alpha) for alpha in alphas if np.isfinite(alpha) and alpha >0]
    if not alpha_grid:
        raise ValueError("alphas must contain at least one positive finite value.")
    
    best_alpha: Optional[float] = None
    best_val_corr: float = -np.inf
    best_val_pred: Optional[pd.Series] = None

    for alpha in alpha_grid:
        pipeline = _ridge_pipeline(alpha=alpha).fit(X_train, y_train)
        val_pred = as_series(pipeline.predict(X_val), index=X_val.index, name="pred_ridge_val")
        corr = pearson_corr(y_val, val_pred)

        if np.isnan(corr):
            continue
        if corr > best_val_corr:
            best_val_corr = corr
            best_alpha = alpha
            best_val_pred = val_pred

    if best_alpha is None or best_val_pred is None:
        raise RuntimeError("Ridge selection failed: all validation correlations were NaN or no valid alphas.")
    
    val_rmse = float(np.sqrt(mean_squared_error(y_val, best_val_pred))) 

    if refit_on_train_val:
        X_train_val = pd.concat([X_train, X_val], axis=0)
        y_train_val = pd.concat([y_train, y_val], axis=0)
        fitted_model = _ridge_pipeline(alpha=best_alpha).fit(X_train_val, y_train_val)
    else:
        fitted_model = _ridge_pipeline(alpha=best_alpha).fit(X_train, y_train)

    diagnostics = {"best_alpha": float(best_alpha), "val_corr": float(best_val_corr), "val_rmse": val_rmse}
    return fitted_model, diagnostics, best_val_pred.rename("pred_ridge_val")


# XGBoost baseline (early stopping on validation)

def train_xgb_with_val(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_estimators: int = 2000,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    early_stopping_rounds: int = 50,
    n_jobs: int = -1,
    random_state: int = 42,
    refit_on_trainval: bool = True,
) -> Tuple["XGBRegressor", Dict[str, float], pd.Series]:
    """Train XGBoost regressor with a validation set and early stopping.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        n_estimators: Maximum boosting rounds (actual rounds may be less if early stopping triggers).
        learning_rate: Learning rate (eta) in (0,1].
        max_depth: Tree depth (>=1).
        subsample: Row subsampling rate in (0,1].
        colsample_bytree: Column subsampling per tree in (0,1].
        reg_lambda: L2 regularisation >=0.
        min_child_weight: Minimum sum of instance weight (Hessian) needed in a child (>=0).
        early_stopping_rounds: Stop if no validation RMSE improvement after this many rounds (>=1).
        n_jobs: Threads (-1 uses all cores).
        random_state: Seed.
        refit_on_trainval: Whether to refit with best_n_estimators on train+val before returning.

    Returns:
        Tuple[XGBRegressor, Dict[str, float], pd.Series]:
            - fitted_model: XGBRegressor fitted either on train or train+val.
            - diagnostics: dict with {'best_n_estimators','val_rmse','val_corr'}.
            - val_pred: Validation predictions from the early-stopped model.

    Raises:
        ValueError: If parameters are out of valid ranges, columns mismatch, or data is empty.
    """

    # Hyperparameter validation
    if not isinstance(n_estimators, int) or n_estimators < 1:
        raise ValueError(f"`n_estimators` must be an integer >= 1, got {n_estimators!r}.")
    if not (0 < learning_rate <= 1):
        raise ValueError(f"`learning_rate` must be in (0, 1], got {learning_rate!r}.")
    if not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError(f"`max_depth` must be an integer >= 1, got {max_depth!r}.")
    if not (0 < subsample <= 1):
        raise ValueError(f"`subsample` must be in (0, 1], got {subsample!r}.")
    if not (0 < colsample_bytree <= 1):
        raise ValueError(f"`colsample_bytree` must be in (0, 1], got {colsample_bytree!r}.")
    if not (np.isfinite(reg_lambda) and reg_lambda >= 0):
        raise ValueError(f"`reg_lambda` must be a finite value >= 0, got {reg_lambda!r}.")
    if not (np.isfinite(min_child_weight) and min_child_weight >= 0):
        raise ValueError(f"`min_child_weight` must be a finite value >= 0, got {min_child_weight!r}.")
    if not isinstance(early_stopping_rounds, int) or early_stopping_rounds < 1:
        raise ValueError(f"`early_stopping_rounds` must be an integer >= 1, got {early_stopping_rounds!r}.")

    # Align and validate data
    X_train, y_train = align_X_y(X_train, y_train, dropna_rows=True)
    X_val, y_val = align_X_y(X_val, y_val, dropna_rows=True)

    # Require identical feature columns and order (avoid silent train/val mismatch)
    if not X_val.columns.equals(X_train.columns):
        raise ValueError(
            "Mismatch between training and validation feature columns. \n"
            f"Train columns: {list(X_train.columns)}\n"
            f"Val columns: {list(X_val.columns)}"
        )

    # Fit with early stopping
    base_model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        objective="reg:squarederror",
        n_jobs=n_jobs,
        random_state=random_state,
        verbosity=0,
    )

    base_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        verbose=False,
        early_stopping_rounds=early_stopping_rounds,
    )

    # If early stopping didn't trigger, best_iteration is None; fall back to n_estimators
    best_n = int(base_model.best_iteration + 1) if getattr(base_model, "best_iteration", None) is not None else int(n_estimators)
    if best_n < 1:
        raise RuntimeError("XGBoost reported an invalid 'best_iteration'. This should not happen; check your data/parameters.")

    # Use the early-stopped iteration range for validation predictions
    val_pred = as_series(
        base_model.predict(X_val, iteration_range=(0, best_n)),
        index=X_val.index,
        name="pred_xgb_val"
    )
    if len(val_pred) != len(X_val):
        raise RuntimeError("Prediction length mismatch on validation set.")
    
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    val_corr = pearson_corr(y_val, val_pred) # may be NaN if degenerate; allowed in diagnostics

    # Optional refit on train+val with best_n
    if refit_on_trainval:
        X_train_val = pd.concat([X_train, X_val], axis=0)
        y_train_val = pd.concat([y_train, y_val], axis=0)
        final_model = XGBRegressor(
            n_estimators=best_n,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            objective="reg:squarederror",
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=0,
        ).fit(X_train_val, y_train_val)
    else:
        final_model = base_model  # already fitted on train with early stopping guidance

    diagnostics = {
        "best_n_estimators": int(best_n), 
        "val_rmse": val_rmse, 
        "val_corr": float(val_corr) if np.isfinite(val_corr) else np.nan
    }
    return final_model, diagnostics, val_pred

# Prediction helpers
def predict_series(model, X: pd.DataFrame, name: str = "prediction") -> pd.Series:
    """Predict with a fitted model and return a Series aligned to X.index.

    Args:
        model: Fitted scikit-learn or XGBoost regressor/pipeline.
        X: Feature matrix.
        name: Name for the returned series.

    Returns:
        pd.Series: Float64 predictions indexed like X.

    Raises:
        AttributeError: If model has no 'predict' method.
        ValueError: If model predictions length does not match X length.
    """
    if not hasattr(model, "predict"):
        raise AttributeError(f"Model {model!r} has no 'predict' method.")
    yhat = model.predict(X)
    return as_series(yhat, index=X.index, name=name)


__all__ = [
    "pearson_corr",
    "predict_persistence",
    "train_ridge_with_val",
    "train_xgb_with_val",
    "predict_series",
]