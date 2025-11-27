import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    fbeta_score,
)

import xgboost as xgb
import shap
import mlflow
import mlflow.catboost
from mlflow.tracking import MlflowClient

from catboost import CatBoostClassifier, Pool

from fraud_utils_weighted import (
    compute_feature_stats,
    build_features,
    add_anomaly_features,
    prepare_matrices,
)

DATA_PATH = "merged_dataset_final.csv"
TARGET_COL = "target"

EXPERIMENT_NAME = "fraud_antifraud_experiment"
REGISTERED_MODEL_NAME = "fraud_detector"

os.makedirs("figures", exist_ok=True)
os.makedirs("artifacts_tmp", exist_ok=True)



def evaluate_binary_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    prefix: str,
    beta: float = 2.0,
):

    # ROC-AUC
    roc = roc_auc_score(y_true, y_proba)

    # Точки для поиска лучшего порога
    thresholds = np.linspace(0.01, 0.99, 200)

    best_thr = 0.5
    best_fbeta = -1.0
    best_cm = None

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_thr = thr
            best_cm = cm

    # Итог по лучшему порогу
    tn, fp, fn, tp = best_cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    acc = (tp + tn) / (tn + fp + fn + tp)

    print(f"\n=== {model_name} ===")
    print(f"Best threshold (F{beta}): {best_thr:.3f}")
    print("Confusion matrix [ [TN FP], [FN TP] ]:")
    print(best_cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Accuracy:  {acc: .4f}")
    print(f"Precision: {precision: .4f}")
    print(f"Recall:    {recall: .4f}")
    print(f"FPR:       {fpr: .4f}")
    print(f"ROC-AUC:   {roc: .4f}")
    print(f"F{beta}-score:  {best_fbeta: .4f}\n")

    print("report:")
    print(
        classification_report(
            y_true,
            (y_proba >= best_thr).astype(int),
            digits=4,
            zero_division=0,
        )
    )

    # ROC-кривая
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr_arr, tpr_arr, label=f"{model_name} (AUC={roc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve - {model_name}")
    plt.legend(loc="lower right")
    roc_path = os.path.join("figures", f"{prefix}_roc.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC figure to: {roc_path}")

    # PR-кривая
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_proba)
    plt.figure()
    plt.plot(recall_arr, precision_arr, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curve - {model_name}")
    plt.legend(loc="lower left")
    pr_path = os.path.join("figures", f"{prefix}_pr.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()
    print(f"Saved PR figure to: {pr_path}")

    return {
        "model": model_name,
        "F2": float(best_fbeta),
        "ROC_AUC": float(roc),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "best_thr": float(best_thr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def plot_global_roc_pr(models_info, y_true, proba_dict):
    plt.figure()
    for name in models_info:
        y_proba = proba_dict[name]
        roc = roc_auc_score(y_true, y_proba)
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_proba)
        plt.plot(fpr_arr, tpr_arr, label=f"{name} (AUC={roc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves - all models")
    plt.legend(loc="lower right")
    out_path = os.path.join("figures", "roc_all_models_balanced_ensemble.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC comparison to: {out_path}")

    # PR
    plt.figure()
    for name in models_info:
        y_proba = proba_dict[name]
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_proba)
        plt.plot(recall_arr, precision_arr, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves - all models")
    plt.legend(loc="lower left")
    out_path = os.path.join("figures", "pr_all_models_balanced_ensemble.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved PR comparison to: {out_path}")


def compute_shap_catboost(model_cb, X_sample: pd.DataFrame, prefix: str):
    print(f"\nSHAP for CatBoost Balanced ({prefix})")
    explainer = shap.TreeExplainer(model_cb)
    shap_values = explainer.shap_values(X_sample)

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    path_bar = os.path.join("figures", f"shap_cb_{prefix}_bar.png")
    plt.savefig(path_bar, bbox_inches="tight")
    plt.close()
    print(path_bar)

    # Beeswarm
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    path_bee = os.path.join("figures", f"shap_cb_{prefix}_beeswarm.png")
    plt.savefig(path_bee, bbox_inches="tight")
    plt.close()
    print(path_bee)


def compute_shap_xgb(model_xgb, X_sample: pd.DataFrame, prefix: str):
    print(f"\nSHAP for XGBoost ({prefix})")
    explainer = shap.TreeExplainer(model_xgb)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    path_bar = os.path.join("figures", f"shap_xgb_{prefix}_bar.png")
    plt.savefig(path_bar, bbox_inches="tight")
    plt.close()
    print(path_bar)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    path_bee = os.path.join("figures", f"shap_xgb_{prefix}_beeswarm.png")
    plt.savefig(path_bee, bbox_inches="tight")
    plt.close()
    print(path_bee)



def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    df_raw = pd.read_csv(DATA_PATH, sep=";", encoding="cp1251", low_memory=False)
    df_raw[TARGET_COL] = df_raw[TARGET_COL].astype(int)

    print("Форма:", df_raw.shape)
    print("Колонки:", df_raw.columns.tolist())
    print("\nРаспределение таргета:")
    print(df_raw[TARGET_COL].value_counts(normalize=True).rename("proportion"))

    # Сплит
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

    # Статистики + фичеринг
    stats = compute_feature_stats(df_train_raw)

    df_train = build_features(df_train_raw, stats)
    df_test = build_features(df_test_raw, stats)

    all_cols = [c for c in df_train.columns if c != TARGET_COL]
    cat_cols = [
        c
        for c in all_cols
        if df_train[c].dtype == "object" or df_train[c].dtype.name == "category"
    ]
    print(
        f"\nВсего фичей: {len(all_cols)} "
        f"(категориальных: {len(cat_cols)})"
    )

    df_train_an, df_test_an = add_anomaly_features(
        df_train.copy(), df_test.copy(), TARGET_COL, cat_cols
    )

    X_train_an, y_train_an, X_test_an, y_test_an, cat_features_an, cat_idx_an = (
        prepare_matrices(df_train_an, df_test_an, TARGET_COL)
    )

    print("X_train_an shape:", X_train_an.shape)
    print("X_test_an shape :", X_test_an.shape)


    neg = (y_train_an == 0).sum()
    pos = (y_train_an == 1).sum()
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight (approx): {scale_pos_weight:.1f}")

    params_cb_balanced = dict(
        iterations=1200,
        depth=6,
        learning_rate=0.035,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights=None,  # выключаем auto, используем scale_pos_weight
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        l2_leaf_reg=5.0,
        random_strength=1.5,
        bagging_temperature=0.3,
        subsample=0.9,
        rsm=0.9,
        verbose=200,
    )

    train_pool = Pool(
        X_train_an,
        y_train_an,
        cat_features=cat_idx_an if len(cat_idx_an) > 0 else None,
    )
    test_pool = Pool(
        X_test_an,
        y_test_an,
        cat_features=cat_idx_an if len(cat_idx_an) > 0 else None,
    )

    model_cb = CatBoostClassifier(**params_cb_balanced)
    model_cb.fit(train_pool, eval_set=test_pool, use_best_model=True)

    y_proba_cb = model_cb.predict_proba(test_pool)[:, 1]

    metrics_cb = evaluate_binary_model(
        y_test_an, y_proba_cb, model_name="CB_BALANCED", prefix="cb_balanced"
    )

 
    # Берём только числовые колонки
    num_cols = [
        c
        for c in X_train_an.columns
        if np.issubdtype(X_train_an[c].dtype, np.number)
    ]
    print(f"Числовых фичей: {len(num_cols)} из {X_train_an.shape[1]}")
    X_train_num = X_train_an[num_cols]
    X_test_num = X_test_an[num_cols]

    dtrain = xgb.DMatrix(X_train_num, label=y_train_an)
    dtest = xgb.DMatrix(X_test_num, label=y_test_an)

    scale_pos_weight_xgb = scale_pos_weight
    print(f"scale_pos_weight (XGB): {scale_pos_weight_xgb:.1f}")

    params_xgb = {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.05,
        "max_depth": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "scale_pos_weight": scale_pos_weight_xgb,
        "tree_method": "hist",
        "random_state": 42,
    }

    evals = [(dtest, "validation_0")]
    model_xgb = xgb.train(
        params_xgb,
        dtrain,
        num_boost_round=600,
        evals=evals,
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    y_proba_xgb = model_xgb.predict(dtest)

    metrics_xgb = evaluate_binary_model(
        y_test_an, y_proba_xgb, model_name="XGB_NUMERIC", prefix="xgb_numeric"
    )


    print("\n [Model 3] Soft-voting ensemble: 0.7 * CB + 0.3 * XGB")
    alpha = 0.7
    y_proba_ens = alpha * y_proba_cb + (1 - alpha) * y_proba_xgb

    metrics_ens = evaluate_binary_model(
        y_test_an, y_proba_ens, model_name="ENSEMBLE_CB_XGB", prefix="ensemble_cb_xgb"
    )

    # Общие ROC/PR графики
    proba_dict = {
        "CB_BALANCED": y_proba_cb,
        "XGB_NUMERIC": y_proba_xgb,
        "ENSEMBLE_CB_XGB": y_proba_ens,
    }
    plot_global_roc_pr(
        ["CB_BALANCED", "XGB_NUMERIC", "ENSEMBLE_CB_XGB"],
        y_test_an,
        proba_dict,
    )

    sample_size = min(1000, X_test_an.shape[0])
    X_sample_cb = X_test_an.iloc[:sample_size].copy()
    X_sample_xgb = X_test_num.iloc[:sample_size].copy()

    # CatBoost SHAP
    compute_shap_catboost(model_cb, X_sample_cb, prefix="balanced")

    # XGBoost SHAP
    compute_shap_xgb(model_xgb, X_sample_xgb, prefix="numeric")

    champion = metrics_cb
    print(champion)

    # Логируем чемпиона в MLflow как catboost модель
    with mlflow.start_run(run_name="CB_BALANCED_final") as run:
        run_id = run.info.run_id
        print(f"\n>>> MLflow run_id={run_id}")

        mlflow.log_params(params_cb_balanced)

        mlflow.log_metric("roc_auc", champion["ROC_AUC"])
        mlflow.log_metric("F2", champion["F2"])
        mlflow.log_metric("precision", champion["precision"])
        mlflow.log_metric("recall", champion["recall"])
        mlflow.log_metric("fpr", champion["fpr"])
        mlflow.log_metric("best_thr_F2", champion["best_thr"])

        feature_info = {
            "feature_cols": list(X_train_an.columns),
            "cat_feature_names": [
                X_train_an.columns[i] for i in cat_idx_an
            ],
        }
        cfg_path = os.path.join("artifacts_tmp", "feature_config_CB_BALANCED.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(feature_info, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(cfg_path, artifact_path="feature_config")

        mlflow.catboost.log_model(
            cb_model=model_cb,
            artifact_path="model",
            registered_model_name=None,  # регистрируем ниже явно
        )

        model_uri = f"runs:/{run_id}/model"
        client = MlflowClient()

        try:
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
                f"'{REGISTERED_MODEL_NAME}' v{mv.version} и переведена в Production."
            )
            print(
                f"  models:/{REGISTERED_MODEL_NAME}/Production"
            )

        except Exception as e:
            print(
                repr(e)
            )


if __name__ == "__main__":
    shap.initjs()
    main()
