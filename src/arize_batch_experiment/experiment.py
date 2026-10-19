"""Workflow A: run a dataset experiment through Vertex AI batch inference and log it to Arize."""

from dataclasses import dataclass

import pandas as pd
from arize.experiments.evaluators.types import EvaluationResultFieldNames
from arize.experiments.types import ExperimentTaskFieldNames

from arize_batch_experiment import prompts
from arize_batch_experiment.clients import arize_client, bigquery_client, vertex_client
from arize_batch_experiment.config import Settings
from arize_batch_experiment.pipeline import batch_stage
from arize_batch_experiment.state import RunState
from arize_batch_experiment.vertex_batch import VertexBatch

EXAMPLE_ID = "example_id"


@dataclass
class ExperimentOptions:
    dataset: str
    run_id: str
    experiment_name: str
    eval_name: str = "correctness"
    judge: bool = True
    limit: int | None = None
    wait: bool = True
    poll_seconds: int = 60
    resubmit_failed: bool = False


def _export_examples(arize, settings: Settings, opts: ExperimentOptions, state: RunState):
    path = state.dir / "examples.parquet"
    if path.exists():
        return pd.read_parquet(path)
    examples = arize.datasets.list_examples(
        dataset=opts.dataset, space=settings.arize_space_id, all=True
    ).to_df()
    if opts.limit:
        examples = examples.head(opts.limit)
    missing = prompts.placeholders(prompts.TASK_PROMPT) - set(examples.columns)
    if missing:
        raise SystemExit(
            f"TASK_PROMPT uses {sorted(missing)}, which are not dataset columns: "
            f"{sorted(examples.columns)}"
        )
    examples = examples.rename(columns={"id": EXAMPLE_ID})
    path.parent.mkdir(parents=True, exist_ok=True)
    examples.to_parquet(path)
    print(f"Exported {len(examples)} examples from dataset {opts.dataset!r}")
    return examples


def _task_requests(vb: VertexBatch, examples: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            EXAMPLE_ID: examples[EXAMPLE_ID],
            "request": [
                vb.request(prompts.render(prompts.TASK_PROMPT, row), system=prompts.TASK_SYSTEM)
                for row in examples.to_dict("records")
            ],
        }
    )


def _judge_requests(vb: VertexBatch, examples: pd.DataFrame, task: pd.DataFrame) -> pd.DataFrame:
    rows = examples.merge(task[[EXAMPLE_ID, "output"]], on=EXAMPLE_ID)
    rows = rows[rows["output"].notna()]
    return pd.DataFrame(
        {
            EXAMPLE_ID: rows[EXAMPLE_ID],
            "request": [
                vb.request(
                    prompts.render(prompts.EXPERIMENT_JUDGE_PROMPT, row),
                    system=prompts.JUDGE_SYSTEM,
                    json_output=True,
                    temperature=0,
                )
                for row in rows.to_dict("records")
            ],
        }
    )


def build_runs(
    task: pd.DataFrame, verdicts: pd.DataFrame | None, eval_name: str, model: str
) -> pd.DataFrame:
    runs = task.rename(columns={"error": "task_error"}).copy()
    runs["model"] = model
    if verdicts is not None:
        parsed = pd.DataFrame(
            [prompts.parse_verdict(text) for text in verdicts["output"]], index=verdicts.index
        )
        parsed[EXAMPLE_ID] = verdicts[EXAMPLE_ID]
        parsed["judge_error"] = verdicts["error"].fillna(parsed["parse_error"])
        parsed = parsed.drop(columns="parse_error").rename(
            columns={c: f"{eval_name}.{c}" for c in ("label", "score", "explanation")}
        )
        runs = runs.merge(parsed, on=EXAMPLE_ID, how="left")
    runs["output"] = runs["output"].fillna("")
    return runs


def run(settings: Settings, opts: ExperimentOptions) -> None:
    arize = arize_client(settings)
    vb = VertexBatch(
        client=vertex_client(settings),
        bq=bigquery_client(settings),
        model=settings.model,
        dataset=settings.bq_dataset,
    )
    state = RunState.load(settings.runs_dir, opts.run_id)

    if state.get("experiment_id"):
        print(f"Already uploaded as experiment {state.get('experiment_id')}")
        return

    examples = _export_examples(arize, settings, opts, state)
    common = dict(
        state=state,
        vb=vb,
        id_column=EXAMPLE_ID,
        wait=opts.wait,
        poll_seconds=opts.poll_seconds,
        resubmit_failed=opts.resubmit_failed,
    )

    task = batch_stage(stage="task", build_input=lambda: _task_requests(vb, examples), **common)
    if task is None:
        return

    verdicts = None
    if opts.judge:
        verdicts = batch_stage(
            stage="judge", build_input=lambda: _judge_requests(vb, examples, task), **common
        )
        if verdicts is None:
            return

    runs = build_runs(task, verdicts, opts.eval_name, settings.model)
    evaluator_columns = None
    if verdicts is not None:
        evaluator_columns = {
            opts.eval_name: EvaluationResultFieldNames(
                label=f"{opts.eval_name}.label",
                score=f"{opts.eval_name}.score",
                explanation=f"{opts.eval_name}.explanation",
            )
        }

    experiment = arize.experiments.create(
        name=opts.experiment_name,
        dataset=opts.dataset,
        space=settings.arize_space_id,
        experiment_runs=runs,
        task_fields=ExperimentTaskFieldNames(example_id=EXAMPLE_ID, output="output"),
        evaluator_columns=evaluator_columns,
    )
    state.set("experiment_id", experiment.id)
    failed = int(runs["task_error"].notna().sum())
    print(
        f"Created experiment {opts.experiment_name!r} ({experiment.id}): {len(runs)} runs, "
        f"{failed} task failures"
    )
