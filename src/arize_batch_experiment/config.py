import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable {name} (see .env.example)")
    return value


def _optional(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


@dataclass(frozen=True)
class Settings:
    arize_api_key: str
    arize_space_id: str
    arize_api_host: str | None
    arize_flight_host: str | None
    arize_single_host: str | None
    arize_ssl_ca_cert: str | None
    gcp_project: str
    gcp_location: str
    bq_dataset: str
    gemini_model: str
    runs_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            arize_api_key=_required("ARIZE_API_KEY"),
            arize_space_id=_required("ARIZE_SPACE_ID"),
            arize_api_host=_optional("ARIZE_API_HOST"),
            arize_flight_host=_optional("ARIZE_FLIGHT_HOST"),
            arize_single_host=_optional("ARIZE_SINGLE_HOST"),
            arize_ssl_ca_cert=_optional("ARIZE_SSL_CA_CERT"),
            gcp_project=_required("GOOGLE_CLOUD_PROJECT"),
            gcp_location=_required("GOOGLE_CLOUD_LOCATION"),
            bq_dataset=_required("BQ_DATASET"),
            gemini_model=_required("GEMINI_MODEL"),
            runs_dir=Path(os.environ.get("RUNS_DIR", "runs")),
        )
