from catboost import CatBoostClassifier

ARTIFACT_DIR = "/Users/arturnavruzov/Desktop/forte_fs_service/mlflow/m-a1c1873435df4f3986306afc75f6530f/artifacts"

model = CatBoostClassifier()
model.load_model(ARTIFACT_DIR + "/model.cb")



print(model.feature_names_)
print(model.get_feature_importance(prettified=True))
