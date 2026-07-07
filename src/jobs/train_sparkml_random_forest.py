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

df = spark.table(full_table)
feature_cols = sorted(c for c in df.columns if c.startswith("feature_"))

print(f"Loaded table {full_table} with {len(feature_cols)} feature columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train & convert to ONNX

# COMMAND ----------

import time

from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.convert.sparkml.operator_converters import (
    random_forest_classifier,
    tree_ensemble_common,
)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col


def _join_storage_path(base, *parts):
    return f"{base.rstrip('/')}/{'/'.join(parts)}"


assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_df = assembler.transform(df).select("features", col("label").cast("double").alias("label"))
rf_model = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=10,
    maxDepth=5,
    seed=42,
).fit(train_df)

# ponytail: UC Volume scratch space — managed storage paths reject ad-hoc ML writes on serverless
onnx_scratch_root = _join_storage_path(
    f"/Volumes/{catalog}/{schema}/onnx_scratch",
    f"_onnx_tmp_{int(time.time() * 1000)}",
)


def _serverless_save_read_sparkml_model_data(spark_session, model):
    # ponytail: onnxmltools reads spark.master (blocked on serverless) to pick ONNX_DFS_PATH vs local tmp
    path = _join_storage_path(
        onnx_scratch_root,
        f"{type(model).__name__}_{int(time.time() * 1000)}",
    )
    model.write().overwrite().save(path)
    return spark_session.read.parquet(_join_storage_path(path, "data"))


# ponytail: random_forest_classifier imports save_read at module load; patch both bindings
random_forest_classifier.save_read_sparkml_model_data = _serverless_save_read_sparkml_model_data
tree_ensemble_common.save_read_sparkml_model_data = _serverless_save_read_sparkml_model_data

from onnxmltools import convert_sparkml

target_opset = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())
initial_types = [("features", FloatTensorType([None, len(feature_cols)]))]
onnx_model = convert_sparkml(
    rf_model,
    "sparkml_random_forest",
    initial_types,
    spark_session=spark,
    target_opset=target_opset,
)

print(f"Trained SparkML RandomForest on Spark and converted to ONNX (opset {target_opset})")

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


class SparkRFOnnxPyFunc(PythonModel):
    def __init__(self, feature_names, model_name):
        self.feature_names = feature_names
        self.model_name = model_name

    def load_context(self, context):
        self.session = ort.InferenceSession(
            context.artifacts["onnx_model"],
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.proba_output_name = next(output.name for output in self.session.get_outputs() if len(output.shape) == 2)

    def predict(self, context, model_input):
        features = model_input[self.feature_names].to_numpy(dtype=np.float32)
        raw = self.session.run([self.proba_output_name], {self.input_name: features})[0]
        if raw.ndim == 2 and raw.shape[1] >= 2:
            proba = raw[:, 1]
        else:
            proba = np.asarray(raw).reshape(-1)
        return pd.DataFrame(
            {
                "target": (proba >= 0.5).astype(np.int64),
                "proba": proba.astype(np.float64),
                "model_name": self.model_name,
            }
        )


mlflow.set_registry_uri("databricks-uc")

sample_pdf = df.select(feature_cols).limit(2).toPandas()
input_schema = Schema([ColSpec("double", column) for column in feature_cols])
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
            python_model=SparkRFOnnxPyFunc(
                feature_names=feature_cols,
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

    sample = mlflow.pyfunc.load_model(model_info.model_uri).predict(sample_pdf)
    assert list(sample.columns) == ["target", "proba", "model_name"] and len(sample) == 2
    assert sample["model_name"].eq(model_name).all()

    registered = mlflow.register_model(model_info.model_uri, full_model_name)

print(f"Registered {registered.name} version {registered.version}")
