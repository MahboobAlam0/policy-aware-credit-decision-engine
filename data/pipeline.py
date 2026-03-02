from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from training.config import PipelineConfig
from training.progress import tqdm

LOGGER = logging.getLogger(__name__)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: np.nan})
    return numerator / denominator


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features aligned with credit-risk underwriting logic."""
    out = df.copy()

    # Home Credit convention: 365243 means "not employed".
    if "DAYS_EMPLOYED" in out.columns:
        out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(365243, np.nan)

    if "DAYS_BIRTH" in out.columns:
        out["AGE_YEARS"] = -out["DAYS_BIRTH"] / 365.0
    if "DAYS_EMPLOYED" in out.columns:
        out["EMPLOYED_YEARS"] = -out["DAYS_EMPLOYED"] / 365.0

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["CREDIT_TO_INCOME_RATIO"] = _safe_divide(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["ANNUITY_TO_INCOME_RATIO"] = _safe_divide(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(out.columns):
        out["CREDIT_TO_GOODS_RATIO"] = _safe_divide(out["AMT_CREDIT"], out["AMT_GOODS_PRICE"])
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(out.columns):
        out["ANNUITY_TO_CREDIT_RATIO"] = _safe_divide(out["AMT_ANNUITY"], out["AMT_CREDIT"])
    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(out.columns):
        out["EMPLOYED_TO_AGE_RATIO"] = _safe_divide(out["DAYS_EMPLOYED"], out["DAYS_BIRTH"])
    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(out.columns):
        out["INCOME_PER_FAMILY_MEMBER"] = _safe_divide(out["AMT_INCOME_TOTAL"], out["CNT_FAM_MEMBERS"])

    return out


def _read_numeric_columns(path: Path, group_key: str, numeric_columns: Sequence[str]) -> pd.DataFrame:
    expected = [group_key, *numeric_columns]
    if not path.exists():
        LOGGER.warning("Skipping %s: file not found.", path)
        return pd.DataFrame(columns=[group_key])

    header = pd.read_csv(path, nrows=0).columns.tolist()
    seen = set()
    usecols = []
    for col in expected:
        if col in header and col not in seen:
            usecols.append(col)
            seen.add(col)
    if group_key not in usecols or len(usecols) <= 1:
        LOGGER.warning("Skipping %s: no required columns available.", path.name)
        return pd.DataFrame(columns=[group_key])

    return pd.read_csv(path, usecols=usecols, low_memory=False)


def _aggregate_numeric_table(
    path: Path,
    group_key: str,
    prefix: str,
    numeric_columns: Sequence[str],
    agg_functions: Iterable[str] = ("mean", "max", "min"),
) -> pd.DataFrame:
    df = _read_numeric_columns(path, group_key=group_key, numeric_columns=numeric_columns)
    if df.empty or group_key not in df.columns:
        return pd.DataFrame(columns=[group_key])

    usable_columns = [c for c in numeric_columns if c in df.columns]
    if not usable_columns:
        return pd.DataFrame(columns=[group_key])

    grouped = df.groupby(group_key)[usable_columns].agg(list(agg_functions))
    grouped.columns = [f"{prefix}_{col}_{agg}" for col, agg in grouped.columns]
    grouped[f"{prefix}_record_count"] = df.groupby(group_key).size()
    grouped = grouped.reset_index()

    return grouped


def _aggregate_installments(path: Path, group_key: str = "SK_ID_CURR") -> pd.DataFrame:
    columns = [
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
        "NUM_INSTALMENT_NUMBER",
    ]
    df = _read_numeric_columns(path, group_key=group_key, numeric_columns=columns)
    if df.empty:
        return pd.DataFrame(columns=[group_key])

    available = set(df.columns)
    if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(available):
        df["INST_DAYS_PAST_DUE"] = (df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]).clip(lower=0)
        df["INST_DAYS_EARLY"] = (df["DAYS_INSTALMENT"] - df["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    if {"AMT_PAYMENT", "AMT_INSTALMENT"}.issubset(available):
        df["INST_PAYMENT_GAP"] = df["AMT_PAYMENT"] - df["AMT_INSTALMENT"]
        df["INST_PAYMENT_RATIO"] = _safe_divide(df["AMT_PAYMENT"], df["AMT_INSTALMENT"])

    numeric_columns = [c for c in df.columns if c != group_key]
    grouped = df.groupby(group_key)[numeric_columns].agg(["mean", "max", "min"])
    grouped.columns = [f"inst_{col}_{agg}" for col, agg in grouped.columns]
    grouped["inst_record_count"] = df.groupby(group_key).size()
    grouped = grouped.reset_index()
    return grouped


def build_auxiliary_features(base_ids: pd.Series, config: PipelineConfig) -> pd.DataFrame:
    """Aggregate secondary tables to one record per applicant."""
    features = pd.DataFrame({config.id_col: base_ids.drop_duplicates()}).reset_index(drop=True)

    bureau_numeric = [
        "DAYS_CREDIT",
        "CREDIT_DAY_OVERDUE",
        "DAYS_CREDIT_ENDDATE",
        "AMT_CREDIT_MAX_OVERDUE",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_SUM_OVERDUE",
        "CNT_CREDIT_PROLONG",
        "AMT_ANNUITY",
    ]
    previous_numeric = [
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "HOUR_APPR_PROCESS_START",
        "RATE_DOWN_PAYMENT",
        "DAYS_DECISION",
        "CNT_PAYMENT",
    ]
    pos_numeric = [
        "MONTHS_BALANCE",
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        "SK_DPD",
        "SK_DPD_DEF",
    ]
    credit_card_numeric = [
        "MONTHS_BALANCE",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_CURRENT",
        "AMT_INST_MIN_REGULARITY",
        "AMT_PAYMENT_CURRENT",
        "AMT_TOTAL_RECEIVABLE",
        "CNT_DRAWINGS_CURRENT",
        "SK_DPD",
        "SK_DPD_DEF",
    ]

    agg_steps = [
        (
            "bureau",
            _aggregate_numeric_table,
            {
                "path": config.bureau_path,
                "group_key": config.id_col,
                "prefix": "bureau",
                "numeric_columns": bureau_numeric,
            },
        ),
        (
            "previous_application",
            _aggregate_numeric_table,
            {
                "path": config.previous_application_path,
                "group_key": config.id_col,
                "prefix": "prev",
                "numeric_columns": previous_numeric,
            },
        ),
        (
            "pos_cash",
            _aggregate_numeric_table,
            {
                "path": config.pos_cash_path,
                "group_key": config.id_col,
                "prefix": "pos",
                "numeric_columns": pos_numeric,
            },
        ),
        (
            "installments",
            _aggregate_installments,
            {"path": config.installments_path, "group_key": config.id_col},
        ),
        (
            "credit_card",
            _aggregate_numeric_table,
            {
                "path": config.credit_card_path,
                "group_key": config.id_col,
                "prefix": "cc",
                "numeric_columns": credit_card_numeric,
            },
        ),
    ]

    for label, builder, kwargs in tqdm(agg_steps, desc="Aggregating auxiliary tables", unit="table"):
        aggregate_df = builder(**kwargs)
        if aggregate_df.empty:
            LOGGER.warning("No auxiliary features merged for %s.", label)
            continue
        features = features.merge(aggregate_df, on=config.id_col, how="left")
        LOGGER.info("Merged %s features. Current shape=%s", label, features.shape)

    return features


def load_modeling_frames(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.train_path.exists():
        raise FileNotFoundError(f"Training file not found: {config.train_path}")
    if not config.test_path.exists():
        raise FileNotFoundError(f"Test file not found: {config.test_path}")

    LOGGER.info("Loading application_train/application_test data...")
    train_df = pd.read_csv(config.train_path, low_memory=False).copy()
    test_df = pd.read_csv(config.test_path, low_memory=False).copy()

    # Build train/test indicator after concat to avoid column insert on fragmented frames.
    full_df = pd.concat([train_df, test_df], ignore_index=True, sort=False, copy=False)
    full_df[config.target_col] = full_df[config.target_col].astype(float)
    full_df["__is_train"] = np.concatenate(
        [
            np.ones(len(train_df), dtype=np.int8),
            np.zeros(len(test_df), dtype=np.int8),
        ]
    )
    full_df = add_business_features(full_df)

    if config.include_auxiliary_tables:
        LOGGER.info("Building auxiliary aggregate features...")
        aux_df = build_auxiliary_features(full_df[config.id_col], config)
        full_df = full_df.merge(aux_df, on=config.id_col, how="left")
        LOGGER.info("Combined frame with auxiliary features shape=%s", full_df.shape)

    train_out = full_df[full_df["__is_train"] == 1].drop(columns=["__is_train"]).copy()
    test_out = full_df[full_df["__is_train"] == 0].drop(columns=["__is_train"]).copy()

    train_out[config.target_col] = train_out[config.target_col].astype(int)
    test_out[config.target_col] = np.nan

    return train_out, test_out


def build_dataset_profile(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: PipelineConfig
) -> dict[str, float | int]:
    target_rate = float(train_df[config.target_col].mean())
    target_counts = train_df[config.target_col].value_counts().to_dict()

    return {
        "train_rows": int(train_df.shape[0]),
        "train_columns": int(train_df.shape[1]),
        "test_rows": int(test_df.shape[0]),
        "test_columns": int(test_df.shape[1]),
        "target_rate": target_rate,
        "target_count_0": int(target_counts.get(0, 0)),
        "target_count_1": int(target_counts.get(1, 0)),
    }

def load_and_preprocess_data(config: PipelineConfig):
    from sklearn.model_selection import train_test_split
    from training.model import split_features_target, _time_based_oot_mask
    
    train_df, test_df = load_modeling_frames(config)
    
    if config.baseline_max_train_rows > 0 and len(train_df) > config.baseline_max_train_rows:
        train_df = train_df.sample(n=config.baseline_max_train_rows, random_state=config.random_state)
        
    X_full, y_full, ids_full = split_features_target(train_df, config)
    
    oot_mask = _time_based_oot_mask(train_df, config)
    
    if oot_mask is not None:
        X_oot = X_full[oot_mask]
        y_oot = y_full[oot_mask]
        ids_oot = ids_full[oot_mask]
        
        X_dev = X_full[~oot_mask]
        y_dev = y_full[~oot_mask]
        ids_dev = ids_full[~oot_mask]
        train_df_dev = train_df[~oot_mask]
    else:
        X_oot, y_oot, ids_oot = pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=int)
        X_dev = X_full
        y_dev = y_full
        ids_dev = ids_full
        train_df_dev = train_df
        
    idx_dev = np.arange(len(X_dev))
    idx_train, idx_valid = train_test_split(
        idx_dev,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=y_dev
    )
    
    idx_train = np.asarray(idx_train)
    idx_valid = np.asarray(idx_valid)
    
    X_train = X_dev.iloc[idx_train].copy().reset_index(drop=True)
    y_train = y_dev.iloc[idx_train].copy().reset_index(drop=True)
    ids_train = ids_dev.iloc[idx_train].copy().reset_index(drop=True)
    
    X_valid = X_dev.iloc[idx_valid].copy().reset_index(drop=True)
    y_valid = y_dev.iloc[idx_valid].copy().reset_index(drop=True)
    ids_valid = ids_dev.iloc[idx_valid].copy().reset_index(drop=True)
    
    protected_cols = [c for c in config.fairness_monitor_columns if c in train_df_dev.columns]
    protected_valid = train_df_dev.iloc[idx_valid][protected_cols].copy().reset_index(drop=True)
    
    return X_train, X_valid, X_oot, y_train, y_valid, y_oot, ids_train, ids_valid, ids_oot, protected_valid
