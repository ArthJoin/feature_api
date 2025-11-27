# fraud_utils_weighted.py
# Версия utils с:
# - feature engineering
# - anomaly features
# - SMOTE
# - вычислением sample_weight по сценариям
# - расчётом FPR
# - SHAP-логированием в MLflow

import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    fbeta_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTENC

from catboost import CatBoostClassifier, Pool

import mlflow
import mlflow.catboost
from mlflow.tracking import MlflowClient
from pandas.api.types import is_numeric_dtype

import shap

# ==========================
# Константы
# ==========================
EXPERIMENT_NAME = "fraud_antifraud_experiment"
REGISTERED_MODEL_NAME = "fraud_detector"


# ============================================================
# 1. Статистика по train для безопасных фичей (БЕЗ direction)
# ============================================================
def compute_feature_stats(df_train_raw: pd.DataFrame) -> dict:
    stats = {}

    # ----- amount -----
    if "amount" in df_train_raw.columns:
        amt = pd.to_numeric(df_train_raw["amount"], errors="coerce").fillna(0)
        stats["amount_q99"] = float(amt.quantile(0.99))
        amt_clipped = amt.clip(lower=0, upper=stats["amount_q99"])

        # биннинги по квантилям (5 бинов)
        quantiles = np.quantile(amt_clipped, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        quantiles = np.unique(quantiles)
        if len(quantiles) < 3:
            quantiles = np.linspace(amt_clipped.min(), amt_clipped.max(), 6)
        stats["amount_bins"] = quantiles.tolist()

        # high amount threshold (верхние 10%)
        stats["amount_high_thresh"] = float(amt_clipped.quantile(0.9))

        # глобальные mean/std для z-score
        stats["amount_mean"] = float(amt_clipped.mean())
        std = amt_clipped.std(ddof=0)
        stats["amount_std"] = float(std if std > 0 else 1.0)
    else:
        stats["amount_q99"] = 0.0
        stats["amount_bins"] = [0.0, 1.0]
        stats["amount_high_thresh"] = 0.0
        stats["amount_mean"] = 0.0
        stats["amount_std"] = 1.0

    # ----- rare categories для девайса / ОС -----
    def get_rare_set(df, col, threshold=30):
        if col not in df.columns:
            return set()
        vc = df[col].astype(str).value_counts()
        rare = vc[vc < threshold].index
        return set(rare.tolist())

    stats["rare_phone_models"] = get_rare_set(
        df_train_raw, "last_phone_model_categorical", threshold=30
    )
    stats["rare_oses"] = get_rare_set(df_train_raw, "last_os_categorical", threshold=30)

    # ----- per-customer агрегаты -----
    if "cst_dim_id" in df_train_raw.columns:
        df_tmp = df_train_raw.copy()
        df_tmp["amount"] = pd.to_numeric(df_tmp["amount"], errors="coerce").fillna(0)

        cust_agg = (
            df_tmp.groupby("cst_dim_id")["amount"]
            .agg(["mean", "std", "max", "count"])
            .rename(
                columns={
                    "mean": "cust_amount_mean",
                    "std": "cust_amount_std",
                    "max": "cust_amount_max",
                    "count": "cust_txn_count",
                }
            )
            .reset_index()
        )
        cust_agg["cust_amount_std"] = cust_agg["cust_amount_std"].fillna(0.0)

        stats["cust_agg"] = cust_agg
        stats["cust_txn_q75"] = float(cust_agg["cust_txn_count"].quantile(0.75))
        stats["cust_txn_q90"] = float(cust_agg["cust_txn_count"].quantile(0.90))
    else:
        stats["cust_agg"] = None
        stats["cust_txn_q75"] = 0.0
        stats["cust_txn_q90"] = 0.0

    # ---- порог по burstiness (для high_burstiness) ----
    if "burstiness_login_interval" in df_train_raw.columns:
        b = pd.to_numeric(
            df_train_raw["burstiness_login_interval"], errors="coerce"
        ).fillna(0)
        stats["burst_q90"] = float(b.abs().quantile(0.9))
    else:
        stats["burst_q90"] = 0.0

    return stats


# ============================================================
# 2. Feature engineering (base + EXT взаимодействия)
# ============================================================
def build_features(df_raw: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df_raw.copy()
    eps = 1e-6

    # ---- 1. Datetime features ----
    if "transdatetime" in df.columns:
        df["transdatetime_parsed"] = pd.to_datetime(
            df["transdatetime"], errors="coerce"
        )
        df["hour"] = df["transdatetime_parsed"].dt.hour.fillna(-1).astype(int)
        df["weekday"] = df["transdatetime_parsed"].dt.weekday.fillna(-1).astype(int)
        df["is_night"] = df["hour"].between(0, 6).astype(int)
        df["is_evening"] = df["hour"].between(18, 23).astype(int)
        df["is_morning"] = df["hour"].between(6, 11).astype(int)
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    else:
        df["hour"] = -1
        df["weekday"] = -1
        df["is_night"] = 0
        df["is_evening"] = 0
        df["is_morning"] = 0
        df["is_weekend"] = 0

    # ---- 2. Amount numeric + clipping + лог + бинны ----
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
        q99 = stats.get("amount_q99", None)
        if q99 is not None:
            df["amount_clipped"] = df["amount"].clip(lower=0, upper=q99)
        else:
            df["amount_clipped"] = df["amount"].clip(lower=0)

        df["amount_log"] = np.log1p(df["amount_clipped"])

        bins = stats.get("amount_bins", None)
        if bins is not None and len(bins) >= 2:
            df["amount_bin"] = (
                pd.cut(df["amount_clipped"], bins=bins, include_lowest=True)
                .astype(str)
                .fillna("ALL")
            )
        else:
            df["amount_bin"] = "ALL"

        high_thr = stats.get("amount_high_thresh", None)
        if high_thr is not None:
            df["is_high_amount"] = (df["amount_clipped"] >= high_thr).astype(int)
        else:
            df["is_high_amount"] = 0
    else:
        df["amount"] = 0.0
        df["amount_clipped"] = 0.0
        df["amount_log"] = 0.0
        df["amount_bin"] = "ALL"
        df["is_high_amount"] = 0

    # глобальный z-score amount
    amt_mean = stats.get("amount_mean", 0.0)
    amt_std = stats.get("amount_std", 1.0)
    df["amount_z_overall"] = (df["amount_clipped"] - amt_mean) / (amt_std + eps)

    # ---- 3. Rare grouping for phone / OS ----
    def apply_rare_grouping(col, rare_set):
        if col not in df.columns:
            return
        df[col] = df[col].astype(str)
        df[col] = df[col].where(~df[col].isin(rare_set), other="OTHER")

    apply_rare_grouping(
        "last_phone_model_categorical", stats.get("rare_phone_models", set())
    )
    apply_rare_grouping("last_os_categorical", stats.get("rare_oses", set()))

    # ---- 4. Поведенческие базовые фичи ----
    for col in [
        "logins_last_7_days",
        "logins_last_30_days",
        "login_frequency_7d",
        "login_frequency_30d",
        "freq_change_7d_vs_mean",
        "avg_login_interval_30d",
        "std_login_interval_30d",
        "var_login_interval_30d",
        "ewm_login_interval_7d",
        "burstiness_login_interval",
        "fano_factor_login_interval",
        "zscore_avg_login_interval_7d",
        "monthly_os_changes",
        "monthly_phone_model_changes",
        "logins_7d_over_30d_ratio",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ---- 5. Поведенческие производные ----
    df["logins_7d_per_day"] = df.get("logins_last_7_days", 0) / 7.0
    df["logins_30d_per_day"] = df.get("logins_last_30_days", 0) / 30.0

    df["amount_per_login_30d"] = df["amount_clipped"] / (
        1.0 + df.get("logins_last_30_days", 0)
    )

    df["freq_change_7d_vs_mean_abs"] = df.get("freq_change_7d_vs_mean", 0).abs()

    df["login_freq_ratio_7_30"] = df.get("login_frequency_7d", 0) / (
        df.get("login_frequency_30d", 0) + eps
    )
    df["logins_ratio_7_30"] = df.get("logins_last_7_days", 0) / (
        df.get("logins_last_30_days", 0) + 1.0
    )

    df["interval_cv_30d"] = df.get("std_login_interval_30d", 0) / (
        df.get("avg_login_interval_30d", 0) + eps
    )
    df["interval_fano_30d"] = df.get("var_login_interval_30d", 0) / (
        df.get("avg_login_interval_30d", 0) + eps
    )

    df["ewm_vs_avg_interval"] = df.get("ewm_login_interval_7d", 0) / (
        df.get("avg_login_interval_30d", 0) + eps
    )

    df["burstiness_abs"] = df.get("burstiness_login_interval", 0).abs()
    df["zscore_login_abs"] = df.get("zscore_avg_login_interval_7d", 0).abs()

    # Interaction: amount_log × freq_change_7d_vs_mean_abs
    df["amountlog_x_freqchange_abs"] = (
        df["amount_log"] * df["freq_change_7d_vs_mean_abs"]
    )

    # ---- 6. Per-customer фичи (из stats["cust_agg"]) ----
    cust_agg = stats.get("cust_agg", None)
    if cust_agg is not None and "cst_dim_id" in df.columns:
        df = df.merge(cust_agg, on="cst_dim_id", how="left")

        df["cust_txn_count"] = df["cust_txn_count"].fillna(0)
        df["cust_amount_mean"] = df["cust_amount_mean"].fillna(0)
        df["cust_amount_std"] = df["cust_amount_std"].fillna(0)
        df["cust_amount_max"] = df["cust_amount_max"].fillna(0)

        df["amount_vs_cust_mean"] = df["amount_clipped"] / (
            df["cust_amount_mean"] + eps
        )
        df["amount_vs_cust_max"] = df["amount_clipped"] / (
            df["cust_amount_max"] + eps
        )

        q75 = stats.get("cust_txn_q75", 0.0)
        q90 = stats.get("cust_txn_q90", 0.0)
        df["cust_is_heavy_q75"] = (df["cust_txn_count"] >= q75).astype(int)
        df["cust_is_heavy_q90"] = (df["cust_txn_count"] >= q90).astype(int)
    else:
        df["cust_txn_count"] = 0
        df["cust_amount_mean"] = 0.0
        df["cust_amount_std"] = 0.0
        df["cust_amount_max"] = 0.0
        df["amount_vs_cust_mean"] = 0.0
        df["amount_vs_cust_max"] = 0.0
        df["cust_is_heavy_q75"] = 0
        df["cust_is_heavy_q90"] = 0

    # ---- 7. Time × behavior interactions ----
    burst_q90 = stats.get("burst_q90", 0.0)
    df["is_high_burst"] = (df["burstiness_abs"] >= burst_q90).astype(int)

    df["night_and_high_amount"] = df["is_night"] * df["is_high_amount"]
    df["weekend_and_high_amount"] = df["is_weekend"] * df["is_high_amount"]

    df["night_and_high_burst"] = df["is_night"] * df["is_high_burst"]
    df["weekend_and_high_burst"] = df["is_weekend"] * df["is_high_burst"]

    # ---- 8. EXT: дополнительные взаимодействия ----
    df["amount_x_burst"] = df["amount_clipped"] * df["burstiness_abs"]
    df["amount_x_fano"] = df["amount_clipped"] * df.get("fano_factor_login_interval", 0)

    df["amount_x_login_freq7"] = df["amount_log"] * df.get("login_frequency_7d", 0)
    df["amount_x_login_freq30"] = df["amount_log"] * df.get("login_frequency_30d", 0)

    df["amount_x_logins_7d"] = df["amount_clipped"] * df.get("logins_last_7_days", 0)
    df["amount_x_logins_30d"] = df["amount_clipped"] * df.get(
        "logins_last_30_days", 0
    )

    df["burst_x_logins_7d"] = df["burstiness_abs"] * df.get("logins_last_7_days", 0)
    df["burst_x_logins_30d"] = df["burstiness_abs"] * df.get("logins_last_30_days", 0)

    df["high_amount_and_no_logins30"] = (
        (df["is_high_amount"] == 1) & (df.get("logins_last_30_days", 0) == 0)
    ).astype(int)

    df["night_x_amount_log"] = df["is_night"] * df["amount_log"]
    df["weekend_x_amount_log"] = df["is_weekend"] * df["amount_log"]
    df["evening_x_amount_log"] = df["is_evening"] * df["amount_log"]

    df["cust_heavy_q75_x_amount"] = df["cust_is_heavy_q75"] * df["amount_log"]
    df["cust_heavy_q75_x_night"] = df["cust_is_heavy_q75"] * df["is_night"]
    df["cust_heavy_q75_x_burst"] = df["cust_is_heavy_q75"] * df["is_high_burst"]

    df["atypical_amount_vs_mean"] = (df["amount_vs_cust_mean"] > 3.0).astype(int)
    df["atypical_amount_vs_max"] = (df["amount_vs_cust_max"] > 1.5).astype(int)

    df["high_amount_z_flag"] = (df["amount_z_overall"] > 2.0).astype(int)
    df["high_burst_and_high_amount"] = (
        df["is_high_burst"] * df["is_high_amount"]
    ).astype(int)

    df["login_spiky_ratio_max"] = df["burstiness_abs"] * df.get(
        "logins_7d_over_30d_ratio", 0
    )

    # ---- 9. Чистка inf/NaN ----
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


# ============================================================
# 3. Anomaly-фичи (IsolationForest / LOF)
# ============================================================
def add_anomaly_features(
    df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str, cat_cols: list
):
    print("\n>>> Строю anomaly-фичи (IsolationForest / LOF) ...")

    drop_cols = [target_col] if target_col in df_train.columns else []

    num_cols = [
        c
        for c in df_train.columns
        if (c not in cat_cols)
        and (c not in drop_cols)
        and is_numeric_dtype(df_train[c])
    ]

    X_train_num = df_train[num_cols].values
    X_test_num = df_test[num_cols].values

    print(f"Числовых фичей для anomaly-модели: {len(num_cols)}")

    # IsolationForest
    iforest = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )
    iforest.fit(X_train_num)
    df_train["iforest_score"] = -iforest.score_samples(X_train_num)
    df_test["iforest_score"] = -iforest.score_samples(X_test_num)

    # LocalOutlierFactor
    lof = LocalOutlierFactor(
        n_neighbors=50,
        contamination=0.01,
        novelty=True,
        n_jobs=-1,
    )
    lof.fit(X_train_num)
    df_train["lof_score"] = -lof.decision_function(X_train_num)
    df_test["lof_score"] = -lof.decision_function(X_test_num)

    return df_train, df_test


# ============================================================
# 4. Подготовка матриц признаков
# ============================================================
def prepare_matrices(df_train, df_test, target_col):
    service_cols = [
        "cst_dim_id",
        "transdate",
        "transdatetime",
        "transdate_clean",
        "transdatetime_parsed",
        "direction",
    ]
    service_cols = [c for c in service_cols if c in df_train.columns]

    feature_cols = [
        c for c in df_train.columns if c not in service_cols + [target_col]
    ]

    X_train = df_train[feature_cols].copy()
    y_train = df_train[target_col].astype(int).copy()

    X_test = df_test[feature_cols].copy()
    y_test = df_test[target_col].astype(int).copy()

    cat_features = [
        c
        for c in X_train.columns
        if X_train[c].dtype == "object" or X_train[c].dtype.name == "category"
    ]
    cat_feature_indices = [X_train.columns.get_loc(c) for c in cat_features]

    return X_train, y_train, X_test, y_test, cat_features, cat_feature_indices


# ============================================================
# 5. SMOTENC-oversampling
# ============================================================
def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, cat_features: list):
    print("\n>>> Строю SMOTE-датасет для A4_anomaly_smote ...")
    if len(cat_features) == 0:
        cat_indices = []
    else:
        cat_indices = [X_train.columns.get_loc(c) for c in cat_features]

    smote = SMOTENC(
        categorical_features=cat_indices,
        random_state=42,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(
        f"Размер train до SMOTE: {X_train.shape}, после SMOTE: {X_res.shape}\n"
        f"Доля класса 1 до SMOTE: {y_train.mean()} после SMOTE: {y_res.mean()}"
    )
    return X_res, y_res


# ============================================================
# 6. Расчёт sample_weight под сценарии
# ============================================================
def compute_sample_weights_for_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> np.ndarray:
    """
    Возвращает вектор весов для каждой строки train.

    Идея:
    - базовый вес = 1
    - усиливаем класс 1 (фрод) в целом
    - отдельно усиливаем "сложные" фроды:
      * большие суммы у клиентов с богатой историей
      * большие суммы + мало истории / высокая burstiness / ночь
    - немного ослабляем "очевидно легитные" негативы
    """
    n = len(X_train)
    w = np.ones(n, dtype=float)

    is_pos = (y_train == 1)
    w[is_pos] *= 3.0  # общий буст для всех фродовых примеров

    # amount_clipped или amount
    if "amount_clipped" in X_train.columns:
        amt = X_train["amount_clipped"]
    elif "amount" in X_train.columns:
        amt = pd.to_numeric(X_train["amount"], errors="coerce").fillna(0)
    else:
        amt = pd.Series(0.0, index=X_train.index)

    cust_txn = X_train.get(
        "cust_txn_count",
        pd.Series(0.0, index=X_train.index),
    )
    logins_30 = X_train.get(
        "logins_last_30_days",
        pd.Series(0.0, index=X_train.index),
    )

    is_night = X_train.get(
        "is_night",
        pd.Series(0, index=X_train.index),
    ).astype(int)
    is_high_burst = X_train.get(
        "is_high_burst",
        pd.Series(0, index=X_train.index),
    ).astype(int)

    amt_q90 = float(np.quantile(amt, 0.90)) if len(amt) > 0 else 0.0
    high_amount = amt >= amt_q90

    cust_q75 = float(np.quantile(cust_txn, 0.75)) if len(cust_txn) > 0 else 0.0
    heavy_cust = cust_txn >= cust_q75

    low_logins = logins_30 <= 2

    # S2-подобные фроды: high_amount + heavy_cust
    mask_hard_fraud_1 = is_pos & high_amount & heavy_cust
    w[mask_hard_fraud_1] *= 2.0

    # S3-подобные фроды: high_amount + мало истории / burst / ночь
    mask_hard_fraud_2 = is_pos & high_amount & (
        low_logins | (is_high_burst == 1) | (is_night == 1)
    )
    w[mask_hard_fraud_2] *= 2.0

    # "легкие" негативы: не фрод, не high_amount, heavy_cust с высокой активностью
    mask_easy_neg = (
        (~is_pos)
        & (~high_amount)
        & (cust_txn > cust_q75)
        & (logins_30 > 20)
    )
    w[mask_easy_neg] *= 0.5

    print(
        f"\n>>> Sample weights stats: min={w.min():.3f}, "
        f"max={w.max():.3f}, mean={w.mean():.3f}"
    )

    return w


# ============================================================
# 7. Обучение + MLflow логирование одной модели
# ============================================================
def train_and_log_catboost_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cat_feature_indices: list,
    extra_tags: dict = None,
):
    mlflow.set_experiment(EXPERIMENT_NAME)

    params = dict(
        iterations=1800,
        depth=6,
        learning_rate=0.035,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        l2_leaf_reg=5.0,
        random_strength=1.5,
        bagging_temperature=0.3,
        subsample=0.9,
        rsm=0.9,
    )

    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id
        print(f"\n>>> Обучаю {name} ... (run_id={run_id})")

        mlflow.log_params(params)
        if extra_tags:
            mlflow.set_tags(extra_tags)

        # sample_weight под сценарии
        sample_weight = compute_sample_weights_for_training(X_train, y_train)

        train_pool = Pool(
            X_train,
            y_train,
            cat_features=cat_feature_indices,
            weight=sample_weight,
        )
        test_pool = Pool(X_test, y_test, cat_features=cat_feature_indices)

        model = CatBoostClassifier(**params, verbose=200)
        model.fit(train_pool, eval_set=test_pool, use_best_model=True)

        # Предсказания
        y_proba = model.predict_proba(test_pool)[:, 1]
        roc = roc_auc_score(y_test, y_proba)

        beta = 2.0
        best_thr = 0.5
        best_fbeta = -1.0
        thresholds = np.linspace(0.05, 0.95, 50)

        for thr in thresholds:
            preds = (y_proba >= thr).astype(int)
            score = fbeta_score(y_test, preds, beta=beta)
            if score > best_fbeta:
                best_fbeta = score
                best_thr = thr

        y_best = (y_proba >= best_thr).astype(int)

        # confusion matrix для FPR
        tn, fp, fn, tp = confusion_matrix(y_test, y_best).ravel()
        fpr = fp / (fp + tn + 1e-9)

        print(f"\n[{name}] ROC-AUC={roc:.4f}, F2={best_fbeta:.4f}, FP Rate={fpr:.4f}")
        print(
            f"[{name}] Лучший порог по F2.0: {best_thr:.3f}, "
            f"TP={tp}, FP={fp}, TN={tn}, FN={fn}"
        )
        print(
            f"\n[{name}] --- Classification report (threshold={best_thr:.3f}) ---"
        )
        print(classification_report(y_test, y_best, digits=4))

        # precision/recall/f1
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_best, average="binary", zero_division=0
        )
        pos_detected = int(y_best.sum())

        # Логируем метрики
        mlflow.log_metric("roc_auc", float(roc))
        mlflow.log_metric("best_thr_F2", float(best_thr))
        mlflow.log_metric("F2", float(best_fbeta))
        mlflow.log_metric("precision", float(prec))
        mlflow.log_metric("recall", float(rec))
        mlflow.log_metric("f1", float(f1))
        mlflow.log_metric("pos_detected", pos_detected)
        mlflow.log_metric("fp_rate", float(fpr))
        mlflow.log_metric("tn", int(tn))
        mlflow.log_metric("fp", int(fp))
        mlflow.log_metric("fn", int(fn))
        mlflow.log_metric("tp", int(tp))

        # Логируем список фичей и категориальных фичей
        feature_info = {
            "feature_cols": list(X_train.columns),
            "cat_feature_names": [
                X_train.columns[i] for i in cat_feature_indices
            ],
        }
        os.makedirs("artifacts_tmp", exist_ok=True)
        cfg_path = os.path.join("artifacts_tmp", f"feature_config_{name}.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(feature_info, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(cfg_path, artifact_path="feature_config")

        # Важности фичей
        feat_imp = pd.Series(
            model.get_feature_importance(train_pool),
            index=X_train.columns,
        ).sort_values(ascending=False)
        fi_path = os.path.join("artifacts_tmp", f"feature_importance_{name}.csv")
        feat_imp.to_csv(fi_path, index=True, encoding="utf-8")
        mlflow.log_artifact(fi_path, artifact_path="feature_importance")

        # SHAP (подвыборка теста)
        try:
            print(f">>> SHAP для {name}")
            sample_size = min(500, len(X_test))
            sample_idx = np.random.choice(len(X_test), size=sample_size, replace=False)
            X_sample = X_test.iloc[sample_idx].copy()

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            shap_path = os.path.join(
                "artifacts_tmp", f"shap_values_{name}.npy"
            )
            np.save(shap_path, shap_values)
            mlflow.log_artifact(shap_path, artifact_path="shap_values")

        except Exception as e:
            print(f"⚠ Ошибка при расчёте SHAP: {repr(e)}")

        # Логируем модель
        mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="model",
            registered_model_name=None,
        )

        return {
            "name": name,
            "run_id": run_id,
            "roc_auc": float(roc),
            "F2": float(best_fbeta),
            "best_thr": float(best_thr),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "fpr": float(fpr),
            "pos_detected": pos_detected,
        }


# ============================================================
# 8. Регистрация лучшей модели в Model Registry
# ============================================================
def register_best_model_in_registry(results: list):
    if not results:
        print("⚠ Нет результатов для регистрации модели.")
        return

    best = max(results, key=lambda r: r["F2"])
    print("\n############################################################")
    print("VALIDATION SUMMARY (по F2)")
    print("############################################################")
    df_res = pd.DataFrame(results)
    print(df_res.sort_values("F2", ascending=False).to_string(index=False))

    print(
        f"\n>>> Лучшая модель по F2: {best['name']} "
        f"(F2={best['F2']:.4f}, ROC-AUC={best['roc_auc']:.4f}), run_id={best['run_id']}"
    )

    model_uri = f"runs:/{best['run_id']}/model"

    try:
        client = MlflowClient()

        mv = mlflow.register_model(
            model_uri=model_uri,
            name=REGISTERED_MODEL_NAME,
        )

        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )

        print(
            f"\n✅ Модель {best['name']} зарегистрирована как "
            f"'{REGISTERED_MODEL_NAME}' v{mv.version} и переведена в Production."
        )
        print(
            "Теперь сервис может забирать её по URI вида:\n"
            f"  models:/{REGISTERED_MODEL_NAME}/Production"
        )

    except Exception as e:
        print(
            "\n⚠ Не удалось зарегистрировать модель в MLflow Model Registry.\n"
            "   Возможно, трекинг-сервер MLflow запущен без поддержки Model Registry.\n"
            f"   Детали ошибки: {repr(e)}"
        )
