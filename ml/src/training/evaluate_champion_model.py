import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    classification_report,
    fbeta_score,
    confusion_matrix,
)

import mlflow
import mlflow.catboost
import shap

from catboost import CatBoostClassifier, Pool

# ====== ВАЖНО: поправь имя модуля, если нужно ======
# from fraud_utils import (
from fraud_utils_weighted import (
    compute_feature_stats,
    build_features,
    add_anomaly_features,
    prepare_matrices,
)

# ==========================
# Константы / пути
# ==========================
DATA_PATH = "merged_dataset_final.csv"
TARGET_COL = "target"

# champion-модель (CatBoost Balanced)
MODEL_NAME = "fraud_detector"
MODEL_STAGE_OR_VERSION = "Production"  # можно заменить на конкретный номер версии
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_STAGE_OR_VERSION}"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
OUTPUT_DIR = "figures_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# beta для F2
BETA = 2.0


# ==========================
# Вспомогательные функции
# ==========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_best_threshold_fbeta(y_true, y_proba, beta=BETA):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_f = -1.0
    for thr in thresholds:
        preds = (y_proba >= thr).astype(int)
        f = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        if f > best_f:
            best_f = f
            best_thr = thr
    return best_thr, best_f


def plot_confusion_matrix(cm, classes, normalize, title, filename):
    """
    cm: np.array 2x2
    classes: ['Non-fraud', 'Fraud']
    """
    if normalize:
        cm_to_show = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_to_show = cm

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_to_show, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm_to_show.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm_to_show[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm_to_show[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(y_true, y_proba, prefix: str):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, f"{prefix}_roc.png")
    fig.savefig(roc_path, bbox_inches="tight")
    plt.close(fig)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    fig.tight_layout()
    pr_path = os.path.join(OUTPUT_DIR, f"{prefix}_pr.png")
    fig.savefig(pr_path, bbox_inches="tight")
    plt.close(fig)

    return roc_path, pr_path


def evaluate_and_plot(
    y_true,
    y_proba,
    model_name: str,
    scenario_name: str,
    prefix: str,
    beta: float = BETA,
):
    """
    Считает метрики, подбирает порог по F-beta, печатает summary
    и сохраняет confusion matrix (обычную и нормированную) + ROC / PR графики.
    """
    best_thr, best_fbeta = find_best_threshold_fbeta(y_true, y_proba, beta=beta)
    y_pred = (y_proba >= best_thr).astype(int)

    roc_auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    fpr = 0.0 if (fp + tn) == 0 else fp / (fp + tn)

    print(f"\n=== {scenario_name} ({model_name}) ===")
    print(f"Best threshold (F{beta:.1f}): {best_thr:.3f}")
    print("Confusion matrix [ [TN FP], [FN TP] ]:")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Accuracy:   {accuracy:7.4f}")
    print(f"Precision:  {precision:7.4f}")
    print(f"Recall:     {recall:7.4f}")
    print(f"FPR:        {fpr:7.4f}")
    print(f"ROC-AUC:    {roc_auc:7.4f}")
    print(f"F{beta:.1f}-score: {best_fbeta:7.4f}\n")

    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4))

    # Сохраняем ROC / PR
    roc_path, pr_path = plot_roc_pr(y_true, y_proba, prefix=prefix)
    print(f"Saved ROC figure to: {roc_path}")
    print(f"Saved PR figure to:  {pr_path}")

    # Сохраняем confusion matrix
    cm_path = os.path.join(OUTPUT_DIR, f"{prefix}_cm.png")
    cm_norm_path = os.path.join(OUTPUT_DIR, f"{prefix}_cm_norm.png")
    plot_confusion_matrix(
        cm,
        classes=["Non-fraud", "Fraud"],
        normalize=False,
        title=f"{scenario_name} – Confusion matrix",
        filename=cm_path,
    )
    plot_confusion_matrix(
        cm,
        classes=["Non-fraud", "Fraud"],
        normalize=True,
        title=f"{scenario_name} – Confusion matrix (normalized)",
        filename=cm_norm_path,
    )
    print(f"Saved confusion matrix to:       {cm_path}")
    print(f"Saved normalized confusion matrix: {cm_norm_path}")

    metrics = {
        "model": model_name,
        "scenario": scenario_name,
        "best_thr": float(best_thr),
        "Fbeta": float(best_fbeta),
        "beta": float(beta),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "accuracy": float(accuracy),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def make_stress_scenario_amount_x3(df_test_raw: pd.DataFrame) -> pd.DataFrame:
    df_mod = df_test_raw.copy()
    df_mod["amount"] = pd.to_numeric(df_mod["amount"], errors="coerce").fillna(0)
    df_mod["amount"] = df_mod["amount"] * 3.0
    return df_mod


def make_stress_scenario_all_night(df_test_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Переводим все транзакции в ночной диапазон (условно ставим час=2).
    """
    df_mod = df_test_raw.copy()
    if "transdatetime" in df_mod.columns:
        dt = pd.to_datetime(df_mod["transdatetime"], errors="coerce")
        # заменим только час, остальные поля не трогаем
        dt = dt.mask(dt.notna(), dt.dt.normalize() + pd.to_timedelta(2, unit="h"))
        df_mod["transdatetime"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    return df_mod


def make_stress_scenario_burst_x3(df_test_raw: pd.DataFrame) -> pd.DataFrame:
    df_mod = df_test_raw.copy()
    if "burstiness_login_interval" in df_mod.columns:
        df_mod["burstiness_login_interval"] = (
            pd.to_numeric(df_mod["burstiness_login_interval"], errors="coerce")
            .fillna(0)
            * 3.0
        )
    return df_mod


def compute_shap_for_catboost(
    model,
    X: pd.DataFrame,
    cat_features_idx: list,
    out_dir: str,
    prefix: str = "cb_balanced",
    max_samples: int = 1000,
):
    """
    Считает SHAP для CatBoost-модели и сохраняет:
      - bar chart (top-k по |SHAP|)
      - beeswarm plot

    model            - CatBoostClassifier (champion)
    X                - фичи (DataFrame) на которых считаем SHAP (например X_test_an)
    cat_features_idx - индексы категориальных признаков (как при обучении)
    out_dir          - папка для сохранения картинок (например 'figures_eval')
    prefix           - префикс в названии файлов
    max_samples      - ограничение по количеству объектов для SHAP
    """
    os.makedirs(out_dir, exist_ok=True)

    # ----- сэмплим, чтобы не упасть по времени / памяти -----
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X.copy()

    feature_names = list(X_sample.columns)

    # ----- строим CatBoost Pool (НОРМАЛЬНЫЙ API, без приватного _build_train_pool) -----
    pool = Pool(
        X_sample,
        cat_features=cat_features_idx if cat_features_idx else None,
    )

    # ----- SHAP через shap.TreeExplainer -----
    # Для CatBoost TreeExplainer умеет работать напрямую с моделью
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pool)

    # Для бинарной классификации shap_values может быть:
    #  - list из [shap_values_class0, shap_values_class1]  ИЛИ
    #  - np.array shape (n_samples, n_features)
    if isinstance(shap_values, list):
        # Берём вклад "положительного" класса (1)
        shap_for_plot = shap_values[1]
    else:
        shap_for_plot = shap_values

    # ----- Bar plot: top-25 признаков по среднему |SHAP| -----
    mean_abs = np.mean(np.abs(shap_for_plot), axis=0)
    idx_sorted = np.argsort(mean_abs)[::-1]  # от большего к меньшему
    top_k = 25
    idx_top = idx_sorted[:top_k]

    plt.figure(figsize=(10, 6))
    plt.barh(
        [feature_names[i] for i in idx_top][::-1],
        mean_abs[idx_top][::-1],
    )
    plt.title("CatBoost Balanced — SHAP feature importance (top 25)")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()

    bar_path = os.path.join(out_dir, f"{prefix}_shap_bar.png")
    plt.savefig(bar_path, dpi=200)
    plt.close()

    # ----- Beeswarm plot -----
    plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_for_plot,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=25,
    )
    beeswarm_path = os.path.join(out_dir, f"{prefix}_shap_beeswarm.png")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=200)
    plt.close()

    print(f"Saved CatBoost SHAP bar to:      {bar_path}")
    print(f"Saved CatBoost SHAP beeswarm to: {beeswarm_path}")



# ==========================
# main()
# ==========================

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f">>> Using MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
    print(f">>> Using MODEL_URI: {MODEL_URI}")

    print(">>> Загружаю датасет...")
    df_raw = pd.read_csv(DATA_PATH, sep=";", encoding="cp1251", low_memory=False)
    df_raw[TARGET_COL] = df_raw[TARGET_COL].astype(int)

    print("Форма:", df_raw.shape)
    print("Колонки:", df_raw.columns.tolist())
    print("\nРаспределение таргета:")
    print(df_raw[TARGET_COL].value_counts(normalize=True))

    # ===== Train / test split (как раньше) =====
    from sklearn.model_selection import train_test_split

    indices = df_raw.index.values
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=df_raw[TARGET_COL],
        random_state=42,
    )

    df_train_raw = df_raw.loc[train_idx].reset_index(drop=True)
    df_test_raw = df_raw.loc[test_idx].reset_index(drop=True)

    print("\nTrain size raw:", df_train_raw.shape, "Test size raw:", df_test_raw.shape)

    # ===== Базовый фичеринг + anomaly =====
    print("\n train-статистики")
    stats = compute_feature_stats(df_train_raw)

    print("фичи для train/test")
    df_train_fe = build_features(df_train_raw, stats)
    df_test_fe = build_features(df_test_raw, stats)

    all_cols = [c for c in df_train_fe.columns if c != TARGET_COL]
    cat_cols = [
        c
        for c in all_cols
        if df_train_fe[c].dtype == "object" or df_train_fe[c].dtype.name == "category"
    ]
    print(
        f"\nВсего фичей: {len(all_cols)} "
        f"(категориальных: {len(cat_cols)})"
    )

    print("\nСтрою anomaly-фичи для A3/A4")
    df_train_an, df_test_an = add_anomaly_features(
        df_train_fe.copy(), df_test_fe.copy(), TARGET_COL, cat_cols
    )

    # матрицы признаков
    X_train_an, y_train_an, X_test_an, y_test_an, cat_features_an, cat_idx_an = (
        prepare_matrices(df_train_an, df_test_an, TARGET_COL)
    )

    print(X_train_an.shape)
    print(X_test_an.shape)

    model = mlflow.catboost.load_model(MODEL_URI)
    if not isinstance(model, CatBoostClassifier):
        raise RuntimeError(f"is not CatBoostClassifier: {type(model)}")

    y_proba_base = model.predict_proba(X_test_an)[:, 1]

    base_metrics = evaluate_and_plot(
        y_true=y_test_an.values,
        y_proba=y_proba_base,
        model_name="CB_BALANCED",
        scenario_name="Base test set",
        prefix="cb_balanced_base",
        beta=BETA,
    )


    all_metrics = [base_metrics]

    df_test_raw_amount = make_stress_scenario_amount_x3(df_test_raw)
    df_test_fe_amount = build_features(df_test_raw_amount, stats)
    _, df_test_an_amount = add_anomaly_features(
        df_train_fe.copy(), df_test_fe_amount.copy(), TARGET_COL, cat_cols
    )
    (
        _,
        _,
        X_test_amount,
        y_test_amount,
        _,
        _,
    ) = prepare_matrices(df_train_an, df_test_an_amount, TARGET_COL)

    y_proba_amount = model.predict_proba(X_test_amount)[:, 1]
    m_amount = evaluate_and_plot(
        y_true=y_test_amount.values,
        y_proba=y_proba_amount,
        model_name="CB_BALANCED",
        scenario_name="Stress: High amount x3",
        prefix="cb_balanced_high_amount_x3",
        beta=BETA,
    )
    all_metrics.append(m_amount)

    df_test_raw_night = make_stress_scenario_all_night(df_test_raw)
    df_test_fe_night = build_features(df_test_raw_night, stats)
    _, df_test_an_night = add_anomaly_features(
        df_train_fe.copy(), df_test_fe_night.copy(), TARGET_COL, cat_cols
    )
    (
        _,
        _,
        X_test_night,
        y_test_night,
        _,
        _,
    ) = prepare_matrices(df_train_an, df_test_an_night, TARGET_COL)

    y_proba_night = model.predict_proba(X_test_night)[:, 1]
    m_night = evaluate_and_plot(
        y_true=y_test_night.values,
        y_proba=y_proba_night,
        model_name="CB_BALANCED",
        scenario_name="Stress: All transactions at night",
        prefix="cb_balanced_all_night",
        beta=BETA,
    )
    all_metrics.append(m_night)

    df_test_raw_burst = make_stress_scenario_burst_x3(df_test_raw)
    df_test_fe_burst = build_features(df_test_raw_burst, stats)
    _, df_test_an_burst = add_anomaly_features(
        df_train_fe.copy(), df_test_fe_burst.copy(), TARGET_COL, cat_cols
    )
    (
        _,
        _,
        X_test_burst,
        y_test_burst,
        _,
        _,
    ) = prepare_matrices(df_train_an, df_test_an_burst, TARGET_COL)

    y_proba_burst = model.predict_proba(X_test_burst)[:, 1]
    m_burst = evaluate_and_plot(
        y_true=y_test_burst.values,
        y_proba=y_proba_burst,
        model_name="CB_BALANCED",
        scenario_name="Stress: High burstiness x3",
        prefix="cb_balanced_high_burst_x3",
        beta=BETA,
    )
    all_metrics.append(m_burst)

    compute_shap_for_catboost(
        model=model,
        X=X_test_an,
        cat_features_idx=cat_idx_an,
        prefix="cb_balanced",
        out_dir="figures_eval",
        max_samples=500,
    )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv_path = os.path.join(OUTPUT_DIR, "champion_model_metrics_and_stress.csv")
    metrics_df.to_csv(metrics_csv_path, index=False, encoding="utf-8")
    print("\nDone.")


if __name__ == "__main__":
    main()
