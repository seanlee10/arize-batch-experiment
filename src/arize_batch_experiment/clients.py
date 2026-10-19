from arize import ArizeClient
from google import genai
from google.cloud import bigquery
from google.genai.types import HttpOptions

from arize_batch_experiment.config import Settings


def arize_client(settings: Settings) -> ArizeClient:
    kwargs: dict = {"api_key": settings.arize_api_key}
    if settings.arize_single_host:
        kwargs["single_host"] = settings.arize_single_host
    else:
        if settings.arize_api_host:
            kwargs["api_host"] = settings.arize_api_host
        if settings.arize_flight_host:
            kwargs["flight_host"] = settings.arize_flight_host
    if settings.arize_ssl_ca_cert:
        kwargs["ssl_ca_cert"] = settings.arize_ssl_ca_cert
    return ArizeClient(**kwargs)


def gemini_client(settings: Settings) -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=settings.gcp_project,
        location=settings.gcp_location,
        http_options=HttpOptions(api_version="v1"),
    )


def bigquery_client(settings: Settings) -> bigquery.Client:
    return bigquery.Client(project=settings.gcp_project)
