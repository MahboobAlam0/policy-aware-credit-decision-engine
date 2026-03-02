from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

from training.config import PipelineConfig
from monitoring.evaluation import classification_metrics
from training.progress import tqdm

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import optuna
except ImportError:  # pragma: no cover - optional dependency
    optuna = None


@dataclass
class ModelResult:
    key: str
    model_name: str
    pipeline: Pipeline
    validation_probabilities: np.ndarray
    metrics: dict[str, float]
    threshold: float
    business_cost: float
    ensemble_models: list[Pipeline] | None = None
    tuned_params: dict[str, Any] | None = None
    tuning_score: float | None = None
    tuning_trials: pd.DataFrame | None = None


@dataclass
class TrainingBundle:
    baseline: ModelResult
    champion: ModelResult
    winner_key: str
    winner_name: str
    comparison: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    id_valid: pd.Series
    X_oot: pd.DataFrame | None
    y_oot: pd.Series | None
    id_oot: pd.Series | None
    split_info: dict[str, float | int | str | bool]


def _make_one_hot_encoder() -> OneHotEncoder:
    base_kwargs: dict[str, Any] = {"handle_unknown": "ignore", "dtype": np.float32}
    try:
        return OneHotEncoder(sparse_output=True, **base_kwargs)
    except TypeError:
        return OneHotEncoder(sparse=True, **base_kwargs)


def split_features_target(
    df: pd.DataFrame, config: PipelineConfig
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    y = df[config.target_col].astype(int)
    ids = df[config.id_col]
    drop_cols = [config.target_col, config.id_col] + list(config.protected_attributes)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y, ids


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in features.columns if c not in categorical_cols]

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    return preprocessor


def build_baseline_estimator(config: PipelineConfig) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        random_state=config.random_state,
        max_iter=1,
        warm_start=True,
        tol=None,
        shuffle=True,
    )


def build_champion_estimator(
    config: PipelineConfig,
    params_override: dict[str, Any] | None = None,
    seed_override: int | None = None,
) -> tuple[Any, str]:
    try:
        from lightgbm import LGBMClassifier

        params = dict(config.lightgbm_params)
        if params_override:
            params.update(params_override)
        params["random_state"] = seed_override if seed_override is not None else config.random_state
        params["n_jobs"] = config.n_jobs
        return LGBMClassifier(**params), "LightGBM"
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier

        fallback_model = RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed_override if seed_override is not None else config.random_state,
            n_jobs=config.n_jobs,
        )
        return fallback_model, "RandomForestFallback"


def build_pipeline(
    model_key: str,
    features: pd.DataFrame,
    config: PipelineConfig,
    params_override: dict[str, Any] | None = None,
    seed_override: int | None = None,
) -> tuple[Pipeline, str]:
    preprocessor = build_preprocessor(features)

    if model_key == "baseline":
        model = build_baseline_estimator(config)
        model_name = "SGDLogisticBaseline"
    elif model_key == "champion":
        model, backend = build_champion_estimator(
            config=config,
            params_override=params_override,
            seed_override=seed_override,
        )
        model_name = backend
    else:
        raise ValueError(f"Unsupported model_key: {model_key}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return pipeline, model_name


def _predict_probability(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]

    decision = pipeline.decision_function(X)
    return 1.0 / (1.0 + np.exp(-decision))


def _make_lgbm_tqdm_callback(total_iterations: int, desc: str):
    progress_bar = tqdm(total=total_iterations, desc=desc, unit="iter", leave=False)
    state = {"last_iteration": 0}

    def _callback(env):
        current = int(env.iteration) + 1
        delta = current - state["last_iteration"]
        if delta > 0:
            progress_bar.update(delta)
            state["last_iteration"] = current
        if current >= int(env.end_iteration):
            progress_bar.close()

    _callback.order = 0
    return _callback, progress_bar


def predict_proba_for_result(result: ModelResult, X: pd.DataFrame) -> np.ndarray:
    models = result.ensemble_models if result.ensemble_models else [result.pipeline]
    probabilities = [_predict_probability(model, X) for model in models]
    return np.mean(np.column_stack(probabilities), axis=1)


def optimize_threshold_by_cost(
    y_true: pd.Series,
    y_proba: np.ndarray,
    false_positive_cost: float,
    false_negative_cost: float,
    progress_desc: str = "Threshold search",
) -> tuple[float, float]:
    best_threshold = 0.5
    best_cost = float("inf")

    thresholds = np.linspace(0.01, 0.99, 99)
    for threshold in tqdm(
        thresholds,
        desc=progress_desc,
        unit="thr",
        leave=False,
    ):
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        _ = tn + tp
        cost = fp * false_positive_cost + fn * false_negative_cost
        if cost < best_cost:
            best_cost = float(cost)
            best_threshold = float(threshold)

    return best_threshold, best_cost


def _sample_for_tuning(
    X_train: pd.DataFrame, y_train: pd.Series, config: PipelineConfig
) -> tuple[pd.DataFrame, pd.Series]:
    if config.tuning_max_rows <= 0 or len(X_train) <= config.tuning_max_rows:
        return X_train, y_train

    X_sample, _, y_sample, _ = train_test_split(
        X_train,
        y_train,
        train_size=config.tuning_max_rows,
        random_state=config.random_state,
        stratify=y_train,
    )
    LOGGER.info(
        "Optuna tuning row cap enabled: using %s rows (from %s total).",
        len(X_sample),
        len(X_train),
    )
    return X_sample, y_sample


def _lightgbm_optuna_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: PipelineConfig,
) -> tuple[dict[str, Any], float | None, pd.DataFrame | None]:
    if optuna is None:
        LOGGER.info("Optuna is not installed; skipping hyperparameter tuning.")
        return {}, None, None
    if not config.optuna_enabled or config.optuna_n_trials <= 0:
        LOGGER.info("Optuna tuning disabled by configuration.")
        return {}, None, None

    probe_model, model_name = build_champion_estimator(config)
    if "lgbm" not in probe_model.__class__.__name__.lower():
        LOGGER.info("Optuna tuning skipped because LightGBM is not available.")
        return {}, None, None

    X_tune, y_tune = _sample_for_tuning(X_train, y_train, config)
    n_splits = max(3, int(config.cv_n_splits))
    minority_count = int(y_tune.value_counts().min())
    n_splits = min(n_splits, max(2, minority_count))
    if n_splits < 2:
        LOGGER.info("Insufficient minority samples for CV tuning; skipping Optuna.")
        return {}, None, None
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)

    LOGGER.info(
        "Starting Optuna tuning for %s with %s trials and %s-fold CV.",
        model_name,
        config.optuna_n_trials,
        n_splits,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        trial_params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1400, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        }

        ap_scores: list[float] = []
        auc_scores: list[float] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_tune, y_tune)):
            X_tr = X_tune.iloc[tr_idx]
            y_tr = y_tune.iloc[tr_idx]
            X_va = X_tune.iloc[va_idx]
            y_va = y_tune.iloc[va_idx]

            model_seed = config.random_state + fold_idx
            pipeline, _ = build_pipeline(
                model_key="champion",
                features=X_tr,
                config=config,
                params_override=trial_params,
                seed_override=model_seed,
            )
            pipeline.fit(X_tr, y_tr)
            p_va = _predict_probability(pipeline, X_va)
            ap_scores.append(float(average_precision_score(y_va, p_va)))
            auc_scores.append(float(roc_auc_score(y_va, p_va)))

        ap_mean = float(np.mean(ap_scores))
        auc_mean = float(np.mean(auc_scores))
        return 0.7 * ap_mean + 0.3 * auc_mean

    sampler = optuna.samplers.TPESampler(seed=config.random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=config.optuna_n_trials,
        timeout=config.optuna_timeout_seconds,
        show_progress_bar=False,
    )

    best_params = dict(study.best_trial.params) if study.best_trial else {}
    best_score = float(study.best_value) if study.best_trial else None
    trials_df = study.trials_dataframe() if hasattr(study, "trials_dataframe") else None
    LOGGER.info("Optuna tuning complete. Best objective score=%.6f", best_score or float("nan"))
    return best_params, best_score, trials_df


def _train_single_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    config: PipelineConfig,
) -> ModelResult:
    pipeline, model_name = build_pipeline(model_key="baseline", features=X_train, config=config)

    X_fit = X_train
    y_fit = y_train
    if config.baseline_max_train_rows > 0 and len(X_train) > config.baseline_max_train_rows:
        X_fit, _, y_fit, _ = train_test_split(
            X_train,
            y_train,
            train_size=config.baseline_max_train_rows,
            random_state=config.random_state,
            stratify=y_train,
        )
        LOGGER.info(
            "Baseline row cap enabled: fitting on %s rows (from %s total).",
            len(X_fit),
            len(X_train),
        )
    else:
        LOGGER.info("Baseline uses full development training rows: %s.", len(X_fit))

    LOGGER.info("Training model=%s ...", model_name)
    start_time = time.perf_counter()
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    X_fit_transformed = preprocessor.fit_transform(X_fit)
    classes = np.array([0, 1], dtype=int)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.asarray(y_fit))
    weight_lookup = {int(cls): float(w) for cls, w in zip(classes, class_weights)}
    sample_weight = y_fit.map(weight_lookup).to_numpy(dtype=float)

    epochs = max(1, int(config.baseline_epochs))
    for epoch in tqdm(range(epochs), desc="Baseline epochs", unit="epoch", leave=False):
        if epoch == 0:
            classifier.partial_fit(X_fit_transformed, y_fit, classes=classes, sample_weight=sample_weight)
        else:
            classifier.partial_fit(X_fit_transformed, y_fit, sample_weight=sample_weight)
    elapsed = time.perf_counter() - start_time
    LOGGER.info("Finished training model=%s in %.1f seconds.", model_name, elapsed)

    y_proba = _predict_probability(pipeline, X_valid)
    threshold, business_cost = optimize_threshold_by_cost(
        y_valid,
        y_proba,
        config.false_positive_cost,
        config.false_negative_cost,
        progress_desc=f"Threshold search ({model_name})",
    )
    metrics = classification_metrics(
        y_valid,
        y_proba,
        threshold,
        calibration_bins=config.calibration_bins,
    )
    metrics["optimal_threshold"] = threshold
    metrics["business_cost"] = business_cost
    metrics["business_cost_per_1k"] = (business_cost / len(y_valid)) * 1000.0

    return ModelResult(
        key="baseline",
        model_name=model_name,
        pipeline=pipeline,
        validation_probabilities=y_proba,
        metrics=metrics,
        threshold=threshold,
        business_cost=business_cost,
        ensemble_models=[pipeline],
    )


def _train_champion_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    config: PipelineConfig,
) -> ModelResult:
    tuned_params, tuning_score, tuning_trials = _lightgbm_optuna_search(X_train, y_train, config)
    seeds = list(config.ensemble_seeds) if config.ensemble_seeds else [config.random_state]

    models: list[Pipeline] = []
    backend_name = "ChampionModel"
    for seed in tqdm(seeds, desc="Champion seed ensemble", unit="seed", leave=False):
        pipeline, model_name = build_pipeline(
            model_key="champion",
            features=X_train,
            config=config,
            params_override=tuned_params if tuned_params else None,
            seed_override=int(seed),
        )
        backend_name = model_name
        LOGGER.info("Training model=%s with seed=%s ...", model_name, seed)
        start_time = time.perf_counter()
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        X_train_transformed = preprocessor.fit_transform(X_train)
        classifier_name = classifier.__class__.__name__.lower()
        if "lgbmclassifier" in classifier_name:
            n_estimators = int(getattr(classifier, "n_estimators", 100))
            callback, progress_bar = _make_lgbm_tqdm_callback(
                total_iterations=n_estimators,
                desc=f"LGBM iter (seed={seed})",
            )
            try:
                classifier.fit(X_train_transformed, y_train, callbacks=[callback])
            finally:
                progress_bar.close()
        else:
            classifier.fit(X_train_transformed, y_train)
        elapsed = time.perf_counter() - start_time
        LOGGER.info("Finished model=%s seed=%s in %.1f seconds.", model_name, seed, elapsed)
        models.append(pipeline)

    ensemble_name = f"{backend_name}Ensemble({len(models)})" if len(models) > 1 else backend_name
    y_proba = np.mean(np.column_stack([_predict_probability(model, X_valid) for model in models]), axis=1)

    threshold, business_cost = optimize_threshold_by_cost(
        y_valid,
        y_proba,
        config.false_positive_cost,
        config.false_negative_cost,
        progress_desc=f"Threshold search ({ensemble_name})",
    )
    metrics = classification_metrics(
        y_valid,
        y_proba,
        threshold,
        calibration_bins=config.calibration_bins,
    )
    metrics["optimal_threshold"] = threshold
    metrics["business_cost"] = business_cost
    metrics["business_cost_per_1k"] = (business_cost / len(y_valid)) * 1000.0

    return ModelResult(
        key="champion",
        model_name=ensemble_name,
        pipeline=models[0],
        validation_probabilities=y_proba,
        metrics=metrics,
        threshold=threshold,
        business_cost=business_cost,
        ensemble_models=models,
        tuned_params=tuned_params if tuned_params else None,
        tuning_score=tuning_score,
        tuning_trials=tuning_trials,
    )


def _time_based_oot_mask(train_df: pd.DataFrame, config: PipelineConfig) -> np.ndarray | None:
    column = config.oot_time_column
    if config.oot_fraction <= 0.0 or config.oot_fraction >= 0.5:
        return None
    if column not in train_df.columns:
        return None

    series = pd.to_numeric(train_df[column], errors="coerce")
    valid = series.notna()
    if valid.sum() < 1000:
        return None

    threshold = float(series[valid].quantile(1.0 - config.oot_fraction))
    mask = (series >= threshold).fillna(False).to_numpy()
    return mask


def compare_models(train_df: pd.DataFrame, config: PipelineConfig) -> TrainingBundle:
    X, y, ids = split_features_target(train_df, config)
    split_info: dict[str, float | int | str | bool] = {
        "oot_enabled": False,
        "oot_time_column": config.oot_time_column,
        "oot_fraction_requested": config.oot_fraction,
        "cv_n_splits": config.cv_n_splits,
        "optuna_enabled": config.optuna_enabled,
        "optuna_n_trials": config.optuna_n_trials,
        "ensemble_seeds": ",".join(str(s) for s in config.ensemble_seeds),
    }

    oot_mask = _time_based_oot_mask(train_df, config)
    X_oot = None
    y_oot = None
    id_oot = None

    if oot_mask is not None:
        y_oot_candidate = y[oot_mask]
        y_pool_candidate = y[~oot_mask]
        if len(y_oot_candidate) > 0 and y_oot_candidate.nunique() == 2 and y_pool_candidate.nunique() == 2:
            X_oot = X[oot_mask]
            y_oot = y_oot_candidate
            id_oot = ids[oot_mask]

            X = X[~oot_mask]
            y = y_pool_candidate
            ids = ids[~oot_mask]
            split_info["oot_enabled"] = True
            split_info["oot_rows"] = int(len(X_oot))
            split_info["oot_fraction_effective"] = float(len(X_oot) / len(train_df))
        else:
            split_info["oot_enabled"] = False
            split_info["oot_rows"] = 0

    split_info["development_rows"] = int(len(X))

    X_train, X_valid, y_train, y_valid, _, id_valid = train_test_split(
        X,
        y,
        ids,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=y,
    )

    model_results: dict[str, ModelResult] = {}
    for model_key in tqdm(["baseline", "champion"], desc="Training models", unit="model"):
        if model_key == "baseline":
            model_results[model_key] = _train_single_baseline(X_train, y_train, X_valid, y_valid, config)
        else:
            model_results[model_key] = _train_champion_ensemble(X_train, y_train, X_valid, y_valid, config)

    baseline = model_results["baseline"]
    champion = model_results["champion"]

    if X_oot is not None and y_oot is not None:
        for result in (baseline, champion):
            oot_proba = predict_proba_for_result(result, X_oot)
            oot_metrics = classification_metrics(
                y_oot,
                oot_proba,
                result.threshold,
                calibration_bins=config.calibration_bins,
            )
            for key, value in oot_metrics.items():
                result.metrics[f"oot_{key}"] = value

    winner = max(
        [baseline, champion],
        key=lambda item: (item.metrics["roc_auc"], item.metrics["average_precision"]),
    )

    comparison = pd.DataFrame(
        [
            {"model_key": baseline.key, "model_name": baseline.model_name, **baseline.metrics},
            {"model_key": champion.key, "model_name": champion.model_name, **champion.metrics},
        ]
    ).sort_values(["roc_auc", "average_precision"], ascending=False)

    summary_cols = [
        "model_key",
        "model_name",
        "roc_auc",
        "average_precision",
        "ks_statistic",
        "business_cost",
        "business_cost_per_1k",
        "optimal_threshold",
    ]
    available_cols = [col for col in summary_cols if col in comparison.columns]
    LOGGER.info(
        "Validation comparison summary:\n%s",
        comparison[available_cols].to_string(index=False),
    )
    LOGGER.info(
        "Winner criteria: highest ROC-AUC then PR-AUC. Winner=%s (%s)",
        winner.key,
        winner.model_name,
    )

    return TrainingBundle(
        baseline=baseline,
        champion=champion,
        winner_key=winner.key,
        winner_name=winner.model_name,
        comparison=comparison,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        id_valid=id_valid,
        X_oot=X_oot,
        y_oot=y_oot,
        id_oot=id_oot,
        split_info=split_info,
    )
