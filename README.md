### Pancancer-model v1
This is a Nextflow-based implementation of a predictive model compatible with the [MbCC](https://mbcc.pum.edu.pl/) model registry.

#### Run locally

1. Install Java with SDKMAN
```
curl -s https://get.sdkman.io | bash
sdk install java 17.0.10-tem
java -version
```

2. Install Nextflow

```
curl -s https://get.nextflow.io | bash
chmod +x nextflow
mkdir -p $HOME/.local/bin/
mv nextflow $HOME/.local/bin/
```

3. Configure Nextflow [nextflow.config]

```
docker {
  enabled = true
}

process {
  container = "janbinkowski96/accs-model-v1"
}
```

#### Start
```
nextflow workflow.nf --input <dir> -work-dir /temp/work
```

#### Content
Nextflow scripts compatible with the MbCC model registry must include:

- `workflow.nf` - Nextflow scripts with a `--input` flag to specify the task directory. 
By default, the workflow expects an 'idats/' subdirectory within the task directory.
  - `metadata.json` - A file containing model metadata, including these required fields: 
    - `Model`: str - Name of model e.g. logistic regression
    - `Number` of features: int - Number of features used by the model
    - `Number` of classes: int - Number of classes used to train the model (for classifiers only)
    - `Classes`: str - List of all supported classes (for classifiers only)
    - `bin/` - A subdirectory with Python or R executables
    - `artifacts/` - A subdirectory containing necessary artifacts, such as the model and imputer.


#### Inference workflow
The inference workflow includes the following five main steps (they are not obligatory and different models
registered within MbCC could differ in number of processing steps):
1. Loading, normalizing, and masking probes from Idat files
2. Imputing missing data
3. Data engineering 
4. Anomaly detection 
5. Inference


#### Expected output
To be compatible with the MbCC model registry, Nextflow scripts must generate `results/` within `input` directory, MbCC expects
that `results/` subdirectory comprises:
1. pp.json - Probability plot
2. ap.json - Anomaly plot
3. nanf.json - Nan frequency plot 
4. cnvs.json - CNVs estimation plot generated using [conumee2.0](https://github.com/hovestadtlab/conumee2)
5. results.json - JSON file comprising:

   - `Predicted_sex` - <str> - Predicted sex of sample (either "Female" or "Male")
   - `Predicted_platform` - <str> - Predicted platform (450K, EPIC or EPICv2)
   - `Prediction` - <str> - Predicted class
   - `Probabilities` - <list> - List of all predicted probabilities 
   - `Confidence_thresholds` - <dict> Threshold values for probabilities e.g. {"High": 0.8, "Medium": 0.65, "Low": 0.5}
   - `Classes` - <list> - List of all classes supported by the model (in the same order as `Probabilities`)
   - `Anomaly_score` - <float> - Numerical value indicating the likelihood of anomaly (novelty/outlier) 
   - `Anomaly_thresholds` - <dict> - Threshold values for anomaly scores e.g. {"Medium-risk sample": 1.5, "High-risk sample": 1.86}

All plots should be generated using Plotly and exported to json using write_json from [plotly.io](https://plotly.com/python-api-reference/generated/plotly.io.html).