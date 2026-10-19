import os

from arize_batch_experiment.config import Settings

REQUIRED = {
    "ARIZE_API_KEY": "key",
    "ARIZE_SPACE_ID": "space",
    "GOOGLE_CLOUD_PROJECT": "project",
    "GOOGLE_CLOUD_LOCATION": "us-central1",
    "BQ_DATASET": "project.dataset",
    "MODEL": "gemini-x",
}


def test_empty_arize_host_vars_are_removed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name, value in REQUIRED.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ARIZE_API_HOST", "")
    monkeypatch.setenv("ARIZE_FLIGHT_HOST", "flight.example.com")

    settings = Settings.from_env()

    assert "ARIZE_API_HOST" not in os.environ
    assert os.environ["ARIZE_FLIGHT_HOST"] == "flight.example.com"
    assert settings.arize_api_host is None
