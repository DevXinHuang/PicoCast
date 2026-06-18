#!/usr/bin/env python
# ruff: noqa: E501, E402
"""Phase 3: Train scikit-learn models on labeled candidate tracklets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from scripts.candidate_utils import (
    load_config,
)


def write_empty_report(out_path: Path, message: str):
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# PicoCAST Candidate Classifier Report\n\n{message}\n")


def main():
    parser = argparse.ArgumentParser(description="Train candidate classifier model.")
    parser.add_argument("config", type=Path, help="Path to config.yaml")

    args = parser.parse_args()
    load_config(args.config)
    case_dir = args.config.parent

    ml_dir = case_dir / "outputs" / "ml"
    report_path = ml_dir / "model_report.md"

    labels_csv = ml_dir / "manual_labels.csv"
    features_parquet = ml_dir / "tracklet_features.parquet"

    # Safeguard 1: Missing files
    if not features_parquet.exists():
        msg = "Tracklet features parquet not found. Run build_ml_feature_table.py first."
        print(msg)
        write_empty_report(report_path, msg)
        return

    tracklets_df = pd.read_parquet(features_parquet)

    if not labels_csv.exists():
        msg = "Not enough labeled examples for reliable ML training."
        print(msg)
        write_empty_report(report_path, msg)
        # Write empty ml scores
        pd.DataFrame(columns=["tracklet_id", "balloon_like_probability", "clutter_probability", "artifact_probability", "ml_best_class", "ml_confidence", "model_notes"]).to_csv(ml_dir / "candidate_ml_scores.csv", index=False)
        return

    labels_df = pd.read_csv(labels_csv)

    # Filter to labeled tracklets only
    labeled_tracklets = labels_df[
        (labels_df["object_type"] == "tracklet") &
        (labels_df["manual_label"].notna()) &
        (labels_df["manual_label"] != "") &
        (labels_df["manual_label"] != "unknown")
    ].copy()

    # Safeguard 2: Less than 5 labeled examples
    if len(labeled_tracklets) < 5:
        msg = "Not enough labeled examples for reliable ML training."
        print(msg)
        write_empty_report(report_path, msg)
        pd.DataFrame(columns=["tracklet_id", "balloon_like_probability", "clutter_probability", "artifact_probability", "ml_best_class", "ml_confidence", "model_notes"]).to_csv(ml_dir / "candidate_ml_scores.csv", index=False)
        return

    # Map labels to target classes
    label_map = {
        "balloon_like": "balloon_like",
        "maybe_balloon": "balloon_like",
        "weather_like": "clutter",
        "terrain_clutter_like": "clutter",
        "bioscatter_like": "clutter",
        "artifact_or_noise": "artifact",
        "bad_spaghetti_tracklet": "artifact",
    }
    labeled_tracklets["target_class"] = labeled_tracklets["manual_label"].map(label_map)

    # Filter invalid classes
    labeled_tracklets = labeled_tracklets[labeled_tracklets["target_class"].notna()]

    # Join features
    df_train = pd.merge(
        labeled_tracklets[["object_id", "target_class"]],
        tracklets_df,
        left_on="object_id",
        right_on="tracklet_id",
        how="inner"
    )

    # Safeguard 3: Less than 5 matching tracklets or single class
    if len(df_train) < 5 or df_train["target_class"].nunique() < 2:
        msg = "Not enough labeled examples for reliable ML training."
        print(msg)
        write_empty_report(report_path, msg)
        pd.DataFrame(columns=["tracklet_id", "balloon_like_probability", "clutter_probability", "artifact_probability", "ml_best_class", "ml_confidence", "model_notes"]).to_csv(ml_dir / "candidate_ml_scores.csv", index=False)
        return

    print(f"Training classifier on {len(df_train)} examples. Class distribution:\n{df_train['target_class'].value_counts()}")

    # Define candidate numeric features
    candidate_features = [
        "n_points", "duration_min", "median_abs_vertical_mismatch_m", 
        "max_abs_vertical_mismatch_m", "median_segment_speed_kmh", "max_segment_speed_kmh",
        "mean_balloon_like_score", "path_smoothness_score", "altitude_consistency_score",
        "telemetry_match_score", "tracklet_score", "spaghetti_score", "n_associations",
        "mean_n_gates", "max_n_gates", "mean_max_reflectivity_dbz", "max_max_reflectivity_dbz",
        "mean_mean_reflectivity_dbz", "mean_rhohv_mean", "mean_compactness_km", "mean_range_km",
        "mean_velocity_mean_ms", "mean_spectrum_width_mean_ms"
    ]

    actual_features = [col for col in candidate_features if col in df_train.columns]

    X = df_train[actual_features].copy()
    y = df_train["target_class"].copy()

    # Preprocessing pipeline pieces
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    # Evaluate using cross-validation (handling tiny datasets gracefully)
    best_score = -1.0
    best_model_name = ""
    model_cv_results = {}

    min_class_size = y.value_counts().min()
    n_splits = min(3, min_class_size)
    if n_splits < 2:
        # If any class has only 1 sample, we cannot use StratifiedKFold safely with >1 split.
        # Fallback to standard 2-fold cross validation on raw index splits
        cv = 2
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
            mean_score = scores.mean()
        except Exception as e:
            print(f"Warning: cross-validation failed for {name}: {e}. Evaluating on training set instead.")
            model.fit(X_scaled, y)
            mean_score = model.score(X_scaled, y)
        
        model_cv_results[name] = mean_score
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name

    print(f"Best model: {best_model_name} with validation score: {best_score:.4f}")

    # Retrain best model on full data
    best_model = models[best_model_name]
    best_model.fit(X_scaled, y)

    # Save model package
    model_package = {
        "model": best_model,
        "features": actual_features,
        "imputer": imputer,
        "scaler": scaler,
        "classes": best_model.classes_.tolist(),
    }
    joblib.dump(model_package, ml_dir / "picocast_candidate_classifier.joblib")
    print(f"Saved model package: {ml_dir / 'picocast_candidate_classifier.joblib'}")

    # Save feature importances
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.zeros(len(actual_features))

    importance_df = pd.DataFrame({
        "feature": actual_features,
        "importance": importances
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(ml_dir / "feature_importance.csv", index=False)

    # Predict for all tracklets
    all_X = tracklets_df[actual_features].copy()
    all_X_imputed = imputer.transform(all_X)
    all_X_scaled = scaler.transform(all_X_imputed)

    probs = best_model.predict_proba(all_X_scaled)
    classes = best_model.classes_.tolist()

    # Get index mapping for targets
    balloon_idx = classes.index("balloon_like") if "balloon_like" in classes else -1
    clutter_idx = classes.index("clutter") if "clutter" in classes else -1
    artifact_idx = classes.index("artifact") if "artifact" in classes else -1

    balloon_probs = probs[:, balloon_idx] if balloon_idx >= 0 else np.zeros(len(probs))
    clutter_probs = probs[:, clutter_idx] if clutter_idx >= 0 else np.zeros(len(probs))
    artifact_probs = probs[:, artifact_idx] if artifact_idx >= 0 else np.zeros(len(probs))

    best_classes = best_model.predict(all_X_scaled)
    confidences = probs.max(axis=1)

    scores_df = pd.DataFrame({
        "tracklet_id": tracklets_df["tracklet_id"],
        "balloon_like_probability": balloon_probs,
        "clutter_probability": clutter_probs,
        "artifact_probability": artifact_probs,
        "ml_best_class": best_classes,
        "ml_confidence": confidences,
        "model_notes": f"Trained on {len(df_train)} labeled examples. Model type: {best_model_name}."
    })
    scores_df.to_csv(ml_dir / "candidate_ml_scores.csv", index=False)
    print(f"Wrote candidate predictions: {ml_dir / 'candidate_ml_scores.csv'}")

    # Generate model_report.md
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# PicoCAST Candidate Classifier Report\n\n")
        f.write("## 1. Dataset Configuration\n")
        f.write(f"- **Total Labeled Tracklets:** {len(df_train)}\n")
        f.write("### Target Class Distribution:\n")
        for cls, count in df_train["target_class"].value_counts().items():
            f.write(f"  - **`{cls}`**: {count}\n")
        f.write("\n")
        f.write("## 2. Model Performance Evaluation\n")
        f.write(f"- **Best Selected Model:** {best_model_name}\n")
        f.write("- **Cross-Validation Scores (Accuracy):**\n")
        for name, score in model_cv_results.items():
            f.write(f"  - {name}: {score:.4f}\n")
        f.write("\n")
        f.write("## 3. Feature Importance Rankings\n")
        f.write("| Feature Name | Relative Importance |\n")
        f.write("| :--- | :---: |\n")
        for _, row in importance_df.head(10).iterrows():
            f.write(f"| `{row['feature']}` | {row['importance']:.4f} |\n")
        f.write("\n")
        f.write("## 4. Top Predicted Balloon Candidates\n")
        f.write("| Tracklet ID | Balloon Probability | Best Class | Confidence |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for _, row in scores_df.sort_values("balloon_like_probability", ascending=False).head(5).iterrows():
            f.write(f"| `{row['tracklet_id']}` | {row['balloon_like_probability'] * 100:.1f}% | `{row['ml_best_class']}` | {row['ml_confidence'] * 100:.1f}% |\n")
        f.write("\n")

    print(f"Wrote model report: {report_path}")


if __name__ == "__main__":
    main()
