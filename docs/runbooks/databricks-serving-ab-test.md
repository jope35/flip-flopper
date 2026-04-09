## Deploy

1. Run `databricks bundle validate -t dev`
2. Run `databricks bundle deploy -t dev`
3. Run SQL bootstrap on your catalog/schema (replace placeholders), e.g. run `src/sql/create_online_tables.sql` with `${catalog}` and `${schema}` substituted, or use a SQL task in a job.
4. Run `databricks bundle run bootstrap_endpoint -t dev`

## Shadow mode

1. Keep the traffic controller job parameters with `--shadow-mode true` for at least one full feedback window.
2. Inspect job logs for proposed splits (`shadow` decisions do not change the endpoint).
3. Switch `--shadow-mode false` only after you trust metrics and guardrails.

## Rollback

1. Re-run the bootstrap job with `--control-percent` / `--challenger-percent` set to the last-known-good split (adjust job parameters or override at run time).
2. Confirm routes in the Model Serving UI.
3. Pause the traffic controller schedule until the challenger issue is understood.

## Config

Application settings live in `config/app.yaml` at the bundle root. Jobs pass `--config-yaml` with path `config/app.yaml` by default (`var.app_config_path` in `databricks.yml`).
