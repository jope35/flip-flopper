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

params = {
    "catalog": catalog,
    "schema": schema,
    "table_name": table_name,
    "model_catalog": model_catalog,
    "model_schema": model_schema,
    "model_name": model_name,
}
if not all(params.values()):
    missing = ", ".join(name for name, value in params.items() if not value)
    raise ValueError(f"All parameters must be non-empty; missing: {missing}")

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

import os
import tempfile

import numpy as np
import onnx
import torch
import torch.nn as nn

HIDDEN = 15
EPOCHS = 10
BATCH_SIZE = 4096
LEARNING_RATE = 0.001
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)


class SimpleMLP(nn.Module):
    """Six Linear layers: five hidden (15 units) + one output (1 unit)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

model = SimpleMLP(input_dim=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCELoss()

model.train()
for _ in range(EPOCHS):
    permutation = torch.randperm(X_tensor.size(0))
    for start in range(0, X_tensor.size(0), BATCH_SIZE):
        idx = permutation[start : start + BATCH_SIZE]
        batch_x = X_tensor[idx]
        batch_y = y_tensor[idx]
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)
        loss.backward()
        optimizer.step()

model.eval()
with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
    onnx_path = f.name

dummy_input = torch.randn(1, X.shape[1], dtype=torch.float32)
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["float_input"],
    output_names=["output"],
    dynamic_axes={
        "float_input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    opset_version=17,
    dynamo=False,
)
onnx_model = onnx.load(onnx_path)
os.unlink(onnx_path)

print(f"Trained SimpleMLP (6 Linear layers, {HIDDEN} hidden units) on {len(X)} rows and converted to ONNX")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register in UC Model Registry

# COMMAND ----------

from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import onnxruntime as ort
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.pyfunc import PythonModel
from mlflow.types.schema import ColSpec, Schema


class PytorchMlpOnnxPyFunc(PythonModel):
    def __init__(self, feature_names, model_name):
        self.feature_names = feature_names
        self.model_name = model_name

    def load_context(self, context):
        self.session = ort.InferenceSession(context.artifacts["onnx_model"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, context, model_input):
        features = model_input[self.feature_names].to_numpy(dtype=np.float32)
        proba = self.session.run(None, {self.input_name: features})[0].squeeze(-1)
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
            python_model=PytorchMlpOnnxPyFunc(
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
