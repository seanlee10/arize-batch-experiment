import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
from google import genai
from google.cloud import bigquery
from google.genai.types import BatchJob, CreateBatchJobConfig, JobState

SUCCEEDED_STATES = {JobState.JOB_STATE_SUCCEEDED, JobState.JOB_STATE_PARTIALLY_SUCCEEDED}
FAILED_STATES = {
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_CANCELLED,
    JobState.JOB_STATE_EXPIRED,
    JobState.JOB_STATE_PAUSED,
}
TERMINAL_STATES = SUCCEEDED_STATES | FAILED_STATES

# Vertex AI writes these into the output table; an input column with either name is rejected.
RESERVED_COLUMNS = {"response", "status"}


def gemini_request(
    prompt: str,
    *,
    system: str | None = None,
    json_output: bool = False,
    temperature: float | None = None,
) -> str:
    request: dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    if system:
        request["systemInstruction"] = {"parts": [{"text": system}]}
    generation_config: dict[str, Any] = {}
    if json_output:
        generation_config["responseMimeType"] = "application/json"
    if temperature is not None:
        generation_config["temperature"] = temperature
    if generation_config:
        request["generationConfig"] = generation_config
    return json.dumps(request, ensure_ascii=False)


def response_text(response: Any) -> str | None:
    if response is None or (isinstance(response, float) and pd.isna(response)):
        return None
    if isinstance(response, str):
        if not response.strip():
            return None
        response = json.loads(response)
    if not isinstance(response, Mapping):
        return None
    candidates = response.get("candidates") or []
    if not candidates:
        return None
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts if isinstance(part, Mapping))
    return text or None


def status_error(status: Any) -> str | None:
    if status is None or (isinstance(status, float) and pd.isna(status)):
        return None
    status = str(status).strip()
    return status or None


def parse_output(rows: pd.DataFrame, id_column: str) -> pd.DataFrame:
    out = pd.DataFrame({id_column: rows[id_column]})
    out["output"] = rows["response"].map(response_text)
    out["error"] = rows["status"].map(status_error)
    missing = out["output"].isna() & out["error"].isna()
    out.loc[missing, "error"] = "Empty response"
    return out.reset_index(drop=True)


@dataclass
class VertexBatch:
    gemini: genai.Client
    bq: bigquery.Client
    model: str
    dataset: str

    def table(self, name: str) -> str:
        return f"{self.dataset}.{name}"

    def write_input(self, df: pd.DataFrame, table: str) -> str:
        clash = RESERVED_COLUMNS & set(df.columns)
        if clash:
            raise ValueError(f"Input columns {sorted(clash)} are reserved by Vertex AI")
        if "request" not in df.columns:
            raise ValueError("Input needs a `request` column")
        self.bq.load_table_from_dataframe(
            df,
            table,
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
        ).result()
        return f"bq://{table}"

    def submit(self, src: str, dest_table: str, display_name: str) -> str:
        job = self.gemini.batches.create(
            model=self.model,
            src=src,
            config=CreateBatchJobConfig(dest=f"bq://{dest_table}", display_name=display_name),
        )
        return job.name

    def get(self, job_name: str) -> BatchJob:
        return self.gemini.batches.get(name=job_name)

    def wait(self, job_name: str, poll_seconds: int = 60) -> BatchJob:
        job = self.get(job_name)
        last_state = None
        while job.state not in TERMINAL_STATES:
            if job.state != last_state:
                print(f"  {job_name}: {job.state.name}")
                last_state = job.state
            time.sleep(poll_seconds)
            job = self.get(job_name)
        print(f"  {job_name}: {job.state.name}")
        return job

    def read_output(self, table: str, id_column: str) -> pd.DataFrame:
        rows = self.bq.query(f"SELECT {id_column}, response, status FROM `{table}`").to_dataframe()
        return parse_output(rows, id_column)
