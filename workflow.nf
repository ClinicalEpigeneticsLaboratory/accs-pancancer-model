#!/usr/bin/env nextflow

params.working_dir = ""

params.model = "artifacts/model"
params.imputer = "artifacts/imputer"
params.scaler = "artifacts/scaler"
params.manifest = "artifacts/manifest.parquet"
params.anomaly_detector = "artifacts/anomaly_detector"

process parseRawData {
    publishDir "$params.working_dir/results", mode: 'copy', overwrite: true, pattern: 'mynorm.parquet'
    publishDir "$params.working_dir/results", mode: 'copy', overwrite: true, pattern: 'predicted.json'

    input:
    path working_dir

    output:
    path 'mynorm.parquet', emit: 'mynorm'
    path 'predicted.json', emit: 'predictions'

    script:
    """
    preprocess.R $working_dir
    """
}

process normalizeData {
    publishDir "$params.working_dir/results", mode: 'copy', overwrite: true, pattern: 'normalized.parquet'

    input:
    path mynorm
    path manifest

    output:
    path 'normalized.parquet'

    script:
    """
    normalize.py $mynorm $manifest
    """
}

process imputeData {
    publishDir "$params.working_dir/results", mode: 'copy', overwrite: true, pattern: 'imputed.parquet'

    input:
    path normalized_mynorm
    path scaler
    path imputer

    output:
    path 'imputed.parquet'

    script:
    """
    #!/usr/local/bin/python3.10

    import pandas as pd
    import joblib

    data = pd.read_parquet("${normalized_mynorm}")
    scaler = joblib.load("${scaler}")
    imputer = joblib.load("${imputer}")

    if not data.isna().any().any():
        data.to_parquet('imputed.parquet')

    else:
        data = data.loc[scaler.feature_names_in_].T
        data_scaled = scaler.transform(data)
        imputed_data = imputer.transform(data_scaled)
        data_imputed = scaler.inverse_transform(imputed_data)
        pd.DataFrame(data_imputed, index=data.index, columns=data.columns).to_parquet('imputed.parquet')
    """
}

process predictData {
    publishDir "$params.working_dir/results", mode: 'copy', overwrite: true, pattern: 'predicted.json'

    input:
    path mynorm
    path model
    path predictions

    output:
    path 'predicted.json'

    script:
    """
    #!/usr/local/bin/python3.10

    import pandas as pd
    import joblib
    import json

    data = pd.read_parquet("${mynorm}")
    model = joblib.load("${model}")

    prediction = model.predict(data)[0]
    classes = model.classes_.tolist()
    proba = model.predict_proba(data).flatten().tolist()

    with open('${predictions}', 'r') as f:
        result = json.load(f)

    with open('predicted.json', 'w') as f:
        result["Prediction"] = prediction
        result["Probabilities"] = proba
        result["Classes"] = classes

        json.dump(result, f)
    """
}

process anomalyDetection {
    publishDir "$params.working_dir/results", mode: 'copy', overwrite: true, pattern: 'predicted.json'

    input:
    path mynorm
    path detector
    path predictions

    output:
    path 'predicted.json'

    script:
    """
    #!/usr/local/bin/python3.10

    import pandas as pd
    import joblib
    import json

    data = pd.read_parquet("${mynorm}")
    detector = joblib.load("${detector}")

    anomaly_score  = abs(detector.score_samples(data)[0])
    threshold = abs(detector["localoutlierfactor"].offset_)

    if anomaly_score >= threshold:
        status = "High-risk sample"

    elif 1.5 < anomaly_score < threshold:
        status = "Medium-risk sample"

    else:
        status = "Low-risk sample"

    with open('${predictions}', 'r') as f:
        result = json.load(f)

    with open('predicted.json', 'w') as f:
        result["Anomaly_score"] = anomaly_score
        result["Anomaly_thresholds"] = {"Medium-risk sample": 1.5, "High-risk sample": threshold}
        json.dump(result, f)

    """

}

process generatePP {
    publishDir "$params.working_dir/results", mode: 'copy', overwrite: true, pattern: 'pp.json'

    input:
    path predicted

    output:
    path "pp.json"

    script:
    """
    #!/usr/local/bin/python3.10

    import json
    import pandas as pd
    import plotly.express as px
    from plotly.io import write_json

    with open("${predicted}", "r") as file:
        predicted = json.load(file)

    n_top = 10
    classes = predicted["Classes"]
    probabilities = predicted["Probabilities"]

    df = pd.concat(
    (
        pd.Series(classes, name="Class"),
        pd.Series(probabilities, name="Probability"),
    ),
    axis=1,
    )

    df = df.sort_values("Probability", ascending=False).iloc[:n_top]

    fig = px.bar(data_frame=df, x="Probability", y="Class", orientation="h", title=f"Prediction probabilities [TOP{n_top}]")
    fig.update_layout(height=500, showlegend=False, xaxis={"range": (0, 1)}, yaxis={"categoryorder": "total ascending"})

    for t, name in zip([0.5, 0.65, 0.8], ["Low", "Medium", "High"]):
        fig.add_vline(
            x=t,
            line_color="red",
            line_width=2,
            line_dash="dot",
            annotation_text=name,
        )

    write_json(fig, "pp.json")
    """
}

process generateAP {
    publishDir "$params.working_dir/results", mode: 'copy', overwrite: true, pattern: 'ap.json'

    input:
    path predicted

    output:
    path "ap.json"

    script:
    """
    #!/usr/local/bin/python3.10
    import json
    import plotly.express as px
    from plotly.io import write_json

    with open("${predicted}", "r") as file:
        predicted = json.load(file)

    thresholds = predicted["Anomaly_thresholds"]
    score = predicted["Anomaly_score"]
    fig = px.bar(x=["Sample"], y=[score], title="Anomaly score")

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

    write_json(fig, "ap.json")
    """
}

process cnvsEstimation {
    input:
    path working_dir

    output:
    path 'cnvs.json'

    script:
    """
    CNVs.R $working_dir
    """
}

process updateCNVsPlot {
    publishDir "$params.working_dir/results", mode: 'copy', overwrite: true, pattern: 'cnvs.json'

    input:
    path cnvs_plot

    output:
    path 'cnvs.json'

    script:
    """
    #!/usr/local/bin/python3.10

    import json
    import plotly.express as px
    from plotly.io import write_json
    from plotly.io import read_json

    cnvs_plot = read_json("${cnvs_plot}", skip_invalid=True)
    cnvs_plot = cnvs_plot.update_layout(title="Estimated CNVs", yaxis={"title": "log2 ratio of normalized intensities"})
    write_json(cnvs_plot, "cnvs.json")
    """
}

workflow {
    wd = file( params.working_dir )
    model = file( params.model )
    scaler = file( params.scaler )
    imputer = file( params.imputer )
    manifest = file( params.manifest )
    anomaly_detector = file( params.anomaly_detector )

    data = parseRawData(wd)
    mynorm_normalized = normalizeData(data.mynorm, manifest)
    mynorm_normalized_imputed = imputeData(mynorm_normalized, scaler, imputer)

    predictions_model = predictData(mynorm_normalized_imputed, model, data.predictions)
    generatePP(predictions_model)

    predictions_anomaly_detector = anomalyDetection(mynorm_normalized_imputed, anomaly_detector, data.predictions)
    generateAP(predictions_anomaly_detector)

    cnvs_plot = cnvsEstimation(wd)
    updateCNVsPlot(cnvs_plot)
}
