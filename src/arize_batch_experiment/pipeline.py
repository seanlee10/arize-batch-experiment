import re
from collections.abc import Callable

import pandas as pd

from arize_batch_experiment.state import RunState
from arize_batch_experiment.vertex_batch import FAILED_STATES, SUCCEEDED_STATES, VertexBatch


def bq_safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def batch_stage(
    *,
    state: RunState,
    vb: VertexBatch,
    stage: str,
    id_column: str,
    build_input: Callable[[], pd.DataFrame],
    wait: bool,
    poll_seconds: int,
    resubmit_failed: bool,
) -> pd.DataFrame | None:
    """Run one batch job for a stage, resuming from saved state.

    Returns the parsed output, or None while the job is still running.
    """
    output_path = state.dir / f"{stage}_output.parquet"
    if output_path.exists():
        return pd.read_parquet(output_path)

    run_id = bq_safe(state.data["run_id"])
    out_table = state.get(f"{stage}_output_table") or vb.table(f"{run_id}_{stage}_out")
    job_name = state.get(f"{stage}_job")

    if job_name is None:
        request_df = build_input()
        if request_df.empty:
            raise SystemExit(f"[{stage}] nothing to submit")
        src = vb.write_input(request_df, vb.table(f"{run_id}_{stage}_in"), id_column)
        job_name = vb.submit(src, out_table, display_name=f"{run_id}-{stage}")
        state.set(f"{stage}_input", src)
        state.set(f"{stage}_output_table", out_table)
        state.set(f"{stage}_job", job_name)
        print(f"[{stage}] submitted {len(request_df)} requests as {job_name}")

    job = vb.wait(job_name, poll_seconds) if wait else vb.get(job_name)

    if job.state in FAILED_STATES:
        if not resubmit_failed:
            raise SystemExit(
                f"[{stage}] job ended in {job.state.name}: {job.error}. "
                "Fix the cause and rerun with --resubmit-failed to resume it."
            )
        # Vertex AI resumes a job submitted with the same input and output: only requests
        # that did not complete are processed again.
        job_name = vb.submit(state.get(f"{stage}_input"), out_table, f"{run_id}-{stage}-resume")
        state.set(f"{stage}_job", job_name)
        print(f"[{stage}] resubmitted as {job_name}")
        job = vb.wait(job_name, poll_seconds) if wait else vb.get(job_name)

    if job.state not in SUCCEEDED_STATES:
        print(f"[{stage}] {job.state.name}; rerun the same command later to continue")
        return None

    if job.completion_stats:
        stats = job.completion_stats
        print(f"[{stage}] done: {stats.successful_count} succeeded, {stats.failed_count} failed")
    output = vb.read_output(out_table, id_column)
    output.to_parquet(output_path)
    return output
