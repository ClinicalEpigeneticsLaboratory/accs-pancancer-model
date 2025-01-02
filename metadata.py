import json
import joblib


def export_metadata():
    model = joblib.load("artifacts/model")
    model_name = model.steps[-1][0]

    n_features = model.n_features_in_
    classes = ", ".join(model.classes_)
    n_classes = len(model.classes_)

    metadata = {"Model": model_name,
                "Number of features": n_features,
                "Number of classes": n_classes,
                "Classes": classes}

    with open("metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)


if __name__ == "__main__":
    export_metadata()