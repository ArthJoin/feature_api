from catboost import CatBoostClassifier

ARTIFACT_DIR = "/Users/arturnavruzov/Desktop/forte_fs_service/mlflow/m-9b9299f0108c489897e3835c1e032f01/artifacts"

model = CatBoostClassifier()
model.load_model(ARTIFACT_DIR + "/model.cb")

print(model.feature_names_)
print(model._feature_names)
print(model.get_feature_importance(prettified=True))
