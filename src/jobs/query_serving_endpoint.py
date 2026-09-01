# Databricks notebook source
# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

dbutils.widgets.text("endpoint", "flip_flopper_serving")
dbutils.widgets.text("served_model", "")

endpoint = dbutils.widgets.get("endpoint").strip()
served_model = dbutils.widgets.get("served_model").strip() or None

if not endpoint:
    raise ValueError("endpoint must be non-empty")

if served_model:
    print(f"Querying {endpoint} served model: {served_model}")
else:
    print(f"Querying {endpoint} (traffic split)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query endpoint

# COMMAND ----------

import json
import urllib.request

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host = ctx.apiUrl().get().rstrip("/")
token = ctx.apiToken().get()

record = {f"feature_{i:03d}": 0.0 for i in range(33)}
payload = {"dataframe_records": [record]}

if served_model:
    path = f"/serving-endpoints/{endpoint}/served-models/{served_model}/invocations"
else:
    path = f"/serving-endpoints/{endpoint}/invocations"

request = urllib.request.Request(
    f"{host}{path}",
    headers={"Authorization": f"Bearer {token}"},
    data=json.dumps(payload).encode("utf-8"),
    method="POST",
)
request.add_header("Content-Type", "application/json")

with urllib.request.urlopen(request, timeout=60) as response:
    result = json.loads(response.read())

output = json.dumps(result, indent=2)
print(output)
dbutils.notebook.exit(output)
