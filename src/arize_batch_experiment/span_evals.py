"""Workflow B: judge exported spans with Gemini batch inference and log evals back to Arize."""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from arize_batch_experiment import prompts
from arize_batch_experiment.clients import arize_client, bigquery_client, gemini_client
from arize_batch_experiment.config import Settings
from arize_batch_experiment.pipeline import batch_stage
from arize_batch_experiment.state import RunState
from arize_batch_experiment.vertex_batch import VertexBatch, gemini_request

SPAN_ID_COLUMN = "context.span_id"
# BigQuery column names cannot contain dots.
SPAN_ID = "span_id"


@dataclass
class SpanEvalOptions:
    project: str
    run_id: str
    start: datetime
    end: datetime
    eval_name: str = "correctness"
    where: str = ""
    sample_rate: float | None = None
    limit: int | None = None
    wait: bool = True
    poll_seconds: int = 60
    resubmit_failed: bool = False


def _export_spans(arize, settings: Settings, opts: SpanEvalOptions, state: RunState):
    path = state.dir / "spans.parquet"
    if path.exists():
        return pd.read_parquet(path)
    columns = sorted(prompts.placeholders(prompts.SPAN_JUDGE_PROMPT) | {SPAN_ID_COLUMN})
    spans = arize.spans.export_to_df(
        space_id=settings.arize_space_id,
        project_name=opts.project,
        start_time=opts.start,
        end_time=opts.end,
        where=opts.where,
        columns=columns,
        sample_rate=opts.sample_rate,
    )
    if opts.limit:
        spans = spans.head(opts.limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    spans.to_parquet(path)
    print(f"Exported {len(spans)} spans from project {opts.project!r}")
    return spans


def _judge_requests(spans: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            SPAN_ID: spans[SPAN_ID_COLUMN],
            "request": [
                gemini_request(
                    prompts.render(prompts.SPAN_JUDGE_PROMPT, row),
                    system=prompts.JUDGE_SYSTEM,
                    json_output=True,
                    temperature=0,
                )
                for row in spans.to_dict("records")
            ],
        }
    )


def build_evals(verdicts: pd.DataFrame, eval_name: str) -> tuple[pd.DataFrame, int]:
    parsed = pd.DataFrame(
        [prompts.parse_verdict(text) for text in verdicts["output"]], index=verdicts.index
    )
    ok = parsed["parse_error"].isna() & (parsed["label"].notna() | parsed["score"].notna())
    evals = pd.DataFrame(
        {
            SPAN_ID_COLUMN: verdicts.loc[ok, SPAN_ID],
            f"eval.{eval_name}.label": parsed.loc[ok, "label"],
            f"eval.{eval_name}.score": parsed.loc[ok, "score"],
            f"eval.{eval_name}.explanation": parsed.loc[ok, "explanation"],
        }
    ).reset_index(drop=True)
    return evals, int((~ok).sum())


def run(settings: Settings, opts: SpanEvalOptions) -> None:
    arize = arize_client(settings)
    vb = VertexBatch(
        gemini=gemini_client(settings),
        bq=bigquery_client(settings),
        model=settings.gemini_model,
        dataset=settings.bq_dataset,
    )
    state = RunState.load(settings.runs_dir, opts.run_id)

    if state.get("evals_uploaded"):
        print(f"Evals for run {opts.run_id!r} were already uploaded")
        return

    spans = _export_spans(arize, settings, opts, state)
    if spans.empty:
        raise SystemExit("No spans matched the time range and filter")

    verdicts = batch_stage(
        state=state,
        vb=vb,
        stage="span_judge",
        id_column=SPAN_ID,
        build_input=lambda: _judge_requests(spans),
        wait=opts.wait,
        poll_seconds=opts.poll_seconds,
        resubmit_failed=opts.resubmit_failed,
    )
    if verdicts is None:
        return

    evals, skipped = build_evals(verdicts, opts.eval_name)
    if evals.empty:
        raise SystemExit("No usable judge verdicts to upload")
    arize.spans.update_evaluations(
        space_id=settings.arize_space_id,
        project_name=opts.project,
        dataframe=evals,
    )
    state.set("evals_uploaded", len(evals))
    print(f"Logged eval {opts.eval_name!r} on {len(evals)} spans ({skipped} skipped)")
