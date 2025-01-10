#!/usr/bin/env nextflow

params.input = ""

params.model = "artifacts/model"
params.imputer = "artifacts/imputer"
params.scaler = "artifacts/scaler"
params.manifest = "artifacts/manifest.parquet"
params.detail_regions = "artifacts/regions.bed"
params.anomaly_detector = "artifacts/anomaly_detector"


log.info """\
==============
Input:
==============
Input directory [--input <path>]: ${params.input}
""".stripIndent()

process parseRawData {
    publishDir "$params.input/results", mode: 'copy', overwrite: true, pattern: 'mynorm.parquet'
    publishDir "$params.input/results", mode: 'copy', overwrite: true, pattern: 'results.json'

    input:
    path input

    output:
    path 'mynorm.parquet', emit: 'mynorm'
    path 'results.json', emit: 'results'

    script:
    """
    preprocess.R $input
    """
}

process normalizeData {
    publishDir "$params.input/results", mode: 'copy', overwrite: true, pattern: 'normalized.parquet'

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
    publishDir "$params.input/results", mode: 'copy', overwrite: true, pattern: 'imputed.parquet'
    publishDir "$params.input/results", mode: 'copy', overwrite: true, pattern: 'results.json'

    input:
    path normalized_mynorm
    path scaler
    path imputer
    path results

    output:
    path 'imputed.parquet'
    path 'results.json'

    script:
    """
    #!/usr/local/bin/python3.10

    import pandas as pd
    import joblib
    import json
    
    data = pd.read_parquet("${normalized_mynorm}")
    if "CpG" in data.columns:
        data = data.set_index("CpG")

    scaler = joblib.load("${scaler}")
    imputer = joblib.load("${imputer}")
    data = data.loc[scaler.feature_names_in_]

    with open('${results}', 'r') as f:
        result = json.load(f)

    nan_freq = data.isna().value_counts(normalize=True).to_dict()
    nan_freq = {str(k[0]): v for k, v in nan_freq.items()}

    with open('results.json', 'w') as f:
        result["NaN_frequency"] = nan_freq
        json.dump(result, f)

    if not data.isna().any().squeeze():
        data.to_parquet('imputed.parquet')

    else:
        data = data.T
        data_scaled = scaler.transform(data)
        imputed_data = imputer.transform(data_scaled)
        data_imputed = scaler.inverse_transform(imputed_data)
        pd.DataFrame(data_imputed, index=data.index, columns=data.columns).to_parquet('imputed.parquet')
    """
}

process plotNaNfreq {
    publishDir "$params.input/results", mode: 'copy', overwrite: true, pattern: 'nanf.json'

    input:
    path results

    output:
    path "nanf.json"

    script:
    """
    #!/usr/local/bin/python3.10

    import json
    import pandas as pd
    import plotly.express as px
    from plotly.io import write_json

    with open('${results}', 'r') as f:
        result = json.load(f)
        nan_freq = result["NaN_frequency"]

    df = pd.DataFrame.from_records(nan_freq, index=["NaN frequency"]).T.reset_index()
    df.columns = ["Missing data", "Frequency"]

    fig = px.pie(df, names="Missing data", values="Frequency", title="Missing data")
    fig.update_layout(height=500, legend={"title": "Missing data"})
    write_json(fig, "nanf.json")
    """
}

process predictData {
    publishDir "$params.input/results", mode: 'copy', overwrite: true, pattern: 'results.json'

    input:
    path mynorm
    path model
    path results

    output:
    path 'results.json'

    script:
    """
    #!/usr/local/bin/python3.10

    import pandas as pd
    import joblib
    import json

    data = pd.read_parquet("${mynorm}")
    model = joblib.load("${model}")

    prediction = model.predict(data)[0]
    proba = model.predict_proba(data).flatten().tolist()

    confidence = max(proba)
    classes = model.classes_.tolist()

    confidence_thresholds = {"High": 0.8, "Medium": 0.65, "Low": 0.5}

    if confidence >= confidence_thresholds["High"]:
        confidence_status = "High"
    elif confidence_thresholds["Medium"] < confidence < confidence_thresholds["High"]:
        confidence_status = "Medium"
    else:
        confidence_status = "Low"

    with open('${results}', 'r') as f:
        result = json.load(f)

    with open('results.json', 'w') as f:
        result["Prediction"] = prediction
        result["Confidence"] = round(confidence, 2)
        result["Confidence_status"] = confidence_status
        result["Confidence_thresholds"] = confidence_thresholds
        result["Probabilities"] = proba
        result["Classes"] = classes

        json.dump(result, f)
    """
}

process anomalyDetection {
    publishDir "$params.input/results", mode: 'copy', overwrite: true, pattern: 'results.json'

    input:
    path mynorm
    path detector
    path results

    output:
    path 'results.json'

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

    with open('${results}', 'r') as f:
        result = json.load(f)

    with open('results.json', 'w') as f:
        result["Anomaly_status"] = status
        result["Anomaly_score"] = anomaly_score
        result["Anomaly_thresholds"] = {"Medium-risk sample": 1.5, "High-risk sample": threshold}

        json.dump(result, f)
    """

}

process generatePP {
    publishDir "$params.input/results", mode: 'copy', overwrite: true, pattern: 'pp.json'

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
    thresholds = predicted["Confidence_thresholds"]

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

    for name, t in thresholds.items():
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
    publishDir "$params.input/results", mode: 'copy', overwrite: true, pattern: 'ap.json'

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
    path input
    path regions

    output:
    path 'cnvs.json'

    script:
    """
    CNVs.R $input $regions
    """
}

process updateCNVsPlot {
    publishDir "$params.input/results", mode: 'copy', overwrite: true, pattern: 'cnvs.json'

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
    wd = file( params.input )
    model = file( params.model )
    scaler = file( params.scaler )
    imputer = file( params.imputer )
    manifest = file( params.manifest )
    detail_regions = file( params.detail_regions )
    anomaly_detector = file( params.anomaly_detector )

    data = parseRawData(wd)
    mynorm_normalized = normalizeData(data.mynorm, manifest)
    (mynorm_normalized_imputed, results) = imputeData(mynorm_normalized, scaler, imputer, data.results)

    plotNaNfreq(results)

    results_model = predictData(mynorm_normalized_imputed, model, results)
    generatePP(results_model)

    results_anomaly_detector = anomalyDetection(mynorm_normalized_imputed, anomaly_detector, data.results)
    generateAP(results_anomaly_detector)

    cnvs_plot = cnvsEstimation(wd, detail_regions)
    updateCNVsPlot(cnvs_plot)
}
