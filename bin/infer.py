from subprocess import call
from os.path import join
import json
import sys
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.io import write_json
from scipy.spatial import cKDTree


def load_artifacts():
    model = joblib.load("artifacts/model")
    scaler = joblib.load("artifacts/scaler")
    imputer = joblib.load("artifacts/imputer")
    anomaly_detector = joblib.load("artifacts/anomaly_detector")
    manifest = pd.read_parquet("artifacts/manifest.parquet")
    return model, scaler, imputer, anomaly_detector, manifest


def parse_raw_data() -> None:
    call(["Rscript", "preprocess.R", "."], shell=False)


def load_beta_values() -> pd.DataFrame:
    beta = pd.read_parquet(join("results", "mynorm.parquet"))
    return beta.set_index("CpG")


def global_norm(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.div(data.mean())
    normalized = normalized.map(lambda x: np.log2(x))
    normalized.index = [f"{cpg}g" for cpg in normalized.index]
    return normalized


def local_norm(
    mynorm: pd.DataFrame, manifest: pd.DataFrame, window: int = 1000
) -> pd.DataFrame:
    common_cpgs = manifest.index.intersection(mynorm.index)

    # Filter both DataFrames to keep only common CpGs
    manifest_filtered = manifest.loc[common_cpgs]
    mynorm_filtered = mynorm.loc[common_cpgs]

    # Sort manifest by chromosome and position
    manifest_sorted = manifest_filtered.sort_values(["CHR", "MAPINFO"])

    # Initialize standardized DataFrame
    normalized = mynorm_filtered.copy()

    # Process each chromosome independently
    for chr_name, chr_df in manifest_sorted.groupby("CHR"):
        positions = chr_df["MAPINFO"].values
        cpg_indices = chr_df.index

        # Using cKDTree for efficient local window lookup
        tree = cKDTree(positions.reshape(-1, 1))

        for idx, pos in zip(cpg_indices, positions):
            indices_within_window = tree.query_ball_point([pos], r=window)

            if len(indices_within_window) <= 1:
                continue

            window_indices = cpg_indices[indices_within_window]
            local_mean = mynorm_filtered.loc[window_indices].mean(axis=0)
            normalized.loc[idx] = np.log2((mynorm_filtered.loc[idx] / local_mean))

    normalized.index = [f"{cpg}l" for cpg in normalized.index]
    return normalized


def impute(data: pd.DataFrame, scaler, imputer) -> tuple[pd.DataFrame, float]:
    if not data.isna().any().any():
        print("No NaN in data, skipping imputation.")
        return data, 0.0

    nan_fraction = sum(data.isna().sum(axis=1)) / data.shape[1]
    data_scaled = scaler.transform(data)
    data_scaled = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)

    imputed_data = imputer.transform(data_scaled)
    imputed_data = scaler.inverse_transform(imputed_data)
    return (
        pd.DataFrame(imputed_data, index=data.index, columns=data.columns),
        nan_fraction,
    )


def predict(sample: pd.DataFrame, model) -> tuple:
    return (
        model.predict(sample)[0],
        model.predict_proba(sample).flatten().tolist(),
        model.classes_.tolist(),
    )


def anomaly_detection(
    sample: pd.DataFrame, anomaly_detector, baseline_threshold: float = 1.5
) -> tuple:
    anomaly_score = abs(anomaly_detector.score_samples(sample)[0])
    threshold = abs(anomaly_detector["localoutlierfactor"].offset_)

    if anomaly_score >= threshold:
        status = "High-risk sample"
    elif baseline_threshold < anomaly_score < threshold:
        status = "Medium-risk sample"
    else:
        status = "Low-risk sample"

    return (
        status,
        anomaly_score,
        {"Medium-risk sample": baseline_threshold, "High-risk sample": threshold},
    )


def asses_confidence_status(proba: float) -> str:
    if proba > 0.8:
        return "High"
    elif 0.65 < proba <= 0.8:
        return "Medium"
    elif 0.5 < proba <= 0.65:
        return "Low"
    else:
        return "Uncertain"


def probability_plot(classes: list, probabilities: list, n_top: int = 10):
    df = pd.concat(
        (
            pd.Series(classes, name="Class"),
            pd.Series(probabilities, name="Probability"),
        ),
        axis=1,
    )
    df = df.sort_values("Probability", ascending=False).iloc[:n_top]

    fig = px.bar(
        data_frame=df,
        x="Probability",
        y="Class",
        orientation="h",
        title="Prediction probabilities [TOP10]",
    )
    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis={"range": (0, 1)},
        yaxis={"categoryorder": "total ascending"},
    )
    for t, name in zip(
        [0.5, 0.65, 0.8],
        ["Low", "Medium", "High"],
    ):
        fig.add_vline(
            x=t,
            line_color="red",
            line_width=2,
            line_dash="dot",
            annotation_text=name,
        )
    return fig


def anomaly_plot(scores: float, thresholds: dict):
    fig = px.bar(x=["Sample"], y=[scores], title="Anomaly score")

    for name, value in thresholds.items():
        fig.add_hline(
            y=value,
            line_color="red",
            line_width=2,
            line_dash="dot",
            annotation_text=name,
        )

    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis_title="",
        yaxis_title="Anomaly score",
    )
    return fig


def summarize(
    prediction,
    confidence,
    classes,
    nan_fraction,
    anomaly_status,
    anomaly_score,
    anomaly_t,
) -> None:

    with open(join("results", "predicted.json"), "r") as file:
        predicted = json.load(file)
        predicted["Prediction"] = prediction
        predicted["Confidence_status"] = asses_confidence_status(max(confidence))
        predicted["Confidence"] = confidence
        predicted["Classes"] = classes
        predicted["Nan_fraction"] = nan_fraction
        predicted["Anomaly_status"] = anomaly_status
        predicted["Anomaly_score"] = anomaly_score
        predicted["Anomaly_thresholds"] = anomaly_t

    with open(join("results", "predicted.json"), "w") as file:
        json.dump(predicted, file)


if __name__ == "__main__":
    if not os.path.exists("idats/"):
        print(f"idats/ dir not found, skipping execution.")
        sys.exit(-1)

    os.makedirs("results", exist_ok=True)
    model, scaler, imputer, anomaly_detector, manifest = load_artifacts()

    assert all(
        np.array_equal(arr, scaler.feature_names_in_)
        for arr in [
            imputer.feature_names_in_,
            model.feature_names_in_,
            anomaly_detector.feature_names_in_,
        ]
    )

    parse_raw_data()

    beta = load_beta_values()
    beta_g_norm = global_norm(beta)
    beta_l_norm = local_norm(beta, manifest)

    data = pd.concat((beta, beta_g_norm, beta_l_norm)).T
    data = data[scaler.feature_names_in_]

    data, nan_fraction = impute(data, scaler, imputer)

    data = data[model.feature_names_in_]
    data.to_parquet(join("results", "sample.parquet"))

    prediction, confidence, classes = predict(data, model)
    confidence_status = asses_confidence_status(max(confidence))
    anomaly_status, anomaly_score, anomaly_t = anomaly_detection(data, anomaly_detector)

    ap = anomaly_plot(anomaly_score, anomaly_t)
    write_json(ap, join("results", "ap.json"))

    pp = probability_plot(classes, confidence)
    write_json(pp, join("results", "pp.json"))

    summarize(
        prediction,
        confidence,
        classes,
        nan_fraction,
        anomaly_status,
        anomaly_score,
        anomaly_t,
    )
