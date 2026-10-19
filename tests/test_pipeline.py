from dataclasses import dataclass, field
from types import SimpleNamespace

import pandas as pd
import pytest
from google.genai.types import JobState

from arize_batch_experiment.pipeline import batch_stage
from arize_batch_experiment.state import RunState


@dataclass
class FakeBatch:
    states: list[JobState]
    submitted: list[tuple[str, str]] = field(default_factory=list)

    def table(self, name):
        return f"proj.ds.{name}"

    def write_input(self, df, table, id_column):
        return f"bq://{table}"

    def submit(self, src, dest_table, display_name):
        self.submitted.append((src, dest_table))
        return f"jobs/{len(self.submitted)}"

    def get(self, job_name):
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return SimpleNamespace(state=state, error="boom", completion_stats=None)

    wait = None

    def read_output(self, table, id_column):
        return pd.DataFrame({id_column: ["a"], "output": ["ok"], "error": [None]})


def _stage(state, vb, resubmit_failed=False):
    return batch_stage(
        state=state,
        vb=vb,
        stage="task",
        id_column="example_id",
        build_input=lambda: pd.DataFrame({"example_id": ["a"], "request": ["{}"]}),
        wait=False,
        poll_seconds=0,
        resubmit_failed=resubmit_failed,
    )


def test_pending_job_is_not_resubmitted_on_rerun(tmp_path):
    vb = FakeBatch(states=[JobState.JOB_STATE_RUNNING])
    state = RunState.load(tmp_path, "r1")
    assert _stage(state, vb) is None
    assert _stage(RunState.load(tmp_path, "r1"), vb) is None
    assert len(vb.submitted) == 1


def test_succeeded_job_output_is_cached(tmp_path):
    vb = FakeBatch(states=[JobState.JOB_STATE_SUCCEEDED])
    out = _stage(RunState.load(tmp_path, "r1"), vb)
    assert out["output"].tolist() == ["ok"]
    vb.states = [JobState.JOB_STATE_FAILED]
    again = _stage(RunState.load(tmp_path, "r1"), vb)
    assert again["output"].tolist() == ["ok"]
    assert len(vb.submitted) == 1


def test_failed_job_stops_unless_resubmit_requested(tmp_path):
    vb = FakeBatch(states=[JobState.JOB_STATE_EXPIRED])
    with pytest.raises(SystemExit, match="JOB_STATE_EXPIRED"):
        _stage(RunState.load(tmp_path, "r1"), vb)

    vb.states = [JobState.JOB_STATE_EXPIRED, JobState.JOB_STATE_SUCCEEDED]
    out = _stage(RunState.load(tmp_path, "r1"), vb, resubmit_failed=True)
    assert out is not None
    assert vb.submitted[0] == vb.submitted[1]
    assert RunState.load(tmp_path, "r1").get("task_job") == "jobs/2"
