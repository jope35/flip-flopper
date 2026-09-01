# Databricks notebook source
# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("table_name", "")
dbutils.widgets.text("model_catalog", "")
dbutils.widgets.text("model_schema", "")
dbutils.widgets.text("model_name", "")

catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()
table_name = dbutils.widgets.get("table_name").strip()
model_catalog = dbutils.widgets.get("model_catalog").strip()
model_schema = dbutils.widgets.get("model_schema").strip()
model_name = dbutils.widgets.get("model_name").strip()

if not all([catalog, schema, table_name, model_catalog, model_schema, model_name]):
    raise ValueError("catalog, schema, table_name, model_catalog, model_schema, and model_name must all be non-empty")

full_table = f"{catalog}.{schema}.{table_name}"
full_model_name = f"{model_catalog}.{model_schema}.{model_name}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data

# COMMAND ----------

pdf = spark.table(full_table).toPandas()

y = pdf["label"]
X = pdf[[c for c in pdf.columns if c.startswith("feature_")]]

print(f"Loaded {len(pdf)} rows with {X.shape[1]} features from {full_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train & convert to ONNX

# COMMAND ----------

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X, y)

initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    options={id(model): {"zipmap": False}},
)

print(f"Trained LogisticRegression on {len(X)} rows and converted to ONNX")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register in UC Model Registry

# COMMAND ----------

from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.pyfunc import PythonModel
from mlflow.types.schema import ColSpec, Schema


class OnnxLogisticRegressionModel(PythonModel):
    def __init__(self, feature_names, model_name):
        self.feature_names = feature_names
        self.model_name = model_name

    def load_context(self, context):
        self.session = ort.InferenceSession(
            context.artifacts["onnx_model"],
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, context, model_input):
        features = model_input[self.feature_names].to_numpy(dtype=np.float32)
        proba = np.asarray(self.session.run(None, {self.input_name: features})[1])[:, 1]
        return pd.DataFrame(
            {
                "target": (proba >= 0.5).astype(np.int64),
                "proba": proba.astype(np.float64),
                "model_name": self.model_name,
            }
        )


mlflow.set_registry_uri("databricks-uc")

input_schema = Schema([ColSpec("double", column) for column in X.columns])
output_schema = Schema(
    [
        ColSpec("long", "target"),
        ColSpec("double", "proba"),
        ColSpec("string", "model_name"),
    ]
)
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

with TemporaryDirectory() as tmpdir:
    onnx_path = Path(tmpdir) / "model.onnx"
    onnx_path.write_bytes(onnx_model.SerializeToString())

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=OnnxLogisticRegressionModel(
                feature_names=list(X.columns),
                model_name=model_name,
            ),
            artifacts={"onnx_model": str(onnx_path)},
            signature=signature,
            pip_requirements=[
                "mlflow",
                "onnxruntime",
                "numpy",
                "pandas",
                "cloudpickle",
            ],
        )

    sample = mlflow.pyfunc.load_model(model_info.model_uri).predict(X.head(2))
    assert list(sample.columns) == ["target", "proba", "model_name"] and len(sample) == 2
    assert sample["model_name"].eq(model_name).all()

    registered = mlflow.register_model(model_info.model_uri, full_model_name)

print(f"Registered {registered.name} version {registered.version}")
