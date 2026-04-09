from pathlib import Path


def test_bundle_files_exist() -> None:
    assert Path("databricks.yml").exists()
    assert Path("resources/jobs.yml").exists()
    assert Path("src/jobs/bootstrap_endpoint.py").exists()
    assert Path("src/jobs/run_traffic_controller.py").exists()
    assert Path("config/app.yaml").exists()
    assert Path("src/sql/create_online_tables.sql").exists()
    assert Path("docs/runbooks/databricks-serving-ab-test.md").exists()
