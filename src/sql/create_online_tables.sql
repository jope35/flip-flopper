CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.flip_flopper_feedback (
  client_request_id STRING NOT NULL,
  label INT NOT NULL,
  label_timestamp TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.flip_flopper_controller_state (
  decision_ts TIMESTAMP NOT NULL,
  endpoint_name STRING NOT NULL,
  action STRING NOT NULL,
  reason STRING NOT NULL,
  control_percent INT NOT NULL,
  challenger_percent INT NOT NULL,
  support_control BIGINT,
  support_challenger BIGINT
);
