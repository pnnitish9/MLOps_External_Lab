import os, json, joblib, mlflow, mlflow.sklearn, pandas as pd
from feast import FeatureStore
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.build_features import FEATURES

MODEL_NAME = "conveyor-jam-model"
TRAIN_VERSIONS = [int(v) for v in os.getenv("TRAIN_VERSIONS", "1").split(",")]

def load():
    labels = pd.read_parquet("data/labels.parquet")
    store = FeatureStore(repo_path="feature_repo")
    feats = store.get_historical_features(
        entity_df=labels[["shift_id", "event_timestamp"]],
        features=[f"conveyor_features:{f}" for f in FEATURES]).to_df()
    return feats.merge(labels[["shift_id", "next_shift_jam", "dataset_version"]], on="shift_id")

def metrics(model, X, y):
    p = model.predict(X)
    return {"accuracy": accuracy_score(y, p), "recall": recall_score(y, p),
            "f1": f1_score(y, p)}, confusion_matrix(y, p).tolist()

def main():
    df = load()
    train_df = df[df.dataset_version.isin(TRAIN_VERSIONS)]
    X_tr, X_te, y_tr, y_te = train_test_split(
        train_df[FEATURES], train_df.next_shift_jam, test_size=0.2,
        stratify=train_df.next_shift_jam, random_state=42)
    drift = df[~df.dataset_version.isin(TRAIN_VERSIONS)]

    models = {
        "logistic_regression": make_pipeline(StandardScaler(),
                                LogisticRegression(class_weight="balanced", max_iter=1000)),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                class_weight="balanced", random_state=42),
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("conveyor-jam")
    results = []
    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_tr, y_tr)
            m, cm = metrics(model, X_te, y_te)
            mlflow.log_param("model_type", name)
            mlflow.log_param("train_versions", str(TRAIN_VERSIONS))
            mlflow.log_metrics(m)
            mlflow.log_dict({"confusion_matrix": cm}, "confusion_matrix.json")
            if len(drift):
                dm, _ = metrics(model, drift[FEATURES], drift.next_shift_jam)
                mlflow.log_metrics({f"drift_{k}": v for k, v in dm.items()})
            
            # Trust scikit-learn tree types explicitly to avoid skops loading issues
            info = mlflow.sklearn.log_model(
                model, 
                name="model", 
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )
            results.append({"name": name, "model": model, "uri": info.model_uri,
                            "cm": cm, **m})
            print(f"{name}: {m}\n  confusion matrix: {cm}")

    # Missing a jam is costlier than a false alarm, so rank by recall, then F1
    best = max(results, key=lambda r: (r["recall"], r["f1"]))
    mv = mlflow.register_model(best["uri"], MODEL_NAME)
    MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    os.makedirs("models", exist_ok=True)
    joblib.dump(best["model"], "models/model.joblib")
    json.dump({"selected": best["name"], "version": mv.version,
               "recall": best["recall"], "f1": best["f1"]},
              open("models/selection.json", "w"), indent=2)
    print(f"\nRegistered '{best['name']}' as {MODEL_NAME} v{mv.version} (alias: champion)")

if __name__ == "__main__":
    main()
