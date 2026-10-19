import json

import pandas as pd
import pytest

from arize_batch_experiment import prompts
from arize_batch_experiment.experiment import build_runs
from arize_batch_experiment.span_evals import build_evals
from arize_batch_experiment.vertex_batch import (
    CLAUDE_MAX_TOKENS,
    VertexBatch,
    claude_request,
    gemini_request,
    is_claude,
    model_resource,
    parse_output,
    response_text,
)


def test_gemini_request_shape():
    req = json.loads(gemini_request("hi", system="sys", json_output=True, temperature=0))
    assert req["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert req["systemInstruction"] == {"parts": [{"text": "sys"}]}
    assert req["generationConfig"] == {"responseMimeType": "application/json", "temperature": 0}


def test_gemini_request_minimal_has_no_optional_fields():
    assert set(json.loads(gemini_request("hi"))) == {"contents"}


def _response(text):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}


@pytest.mark.parametrize(
    "value,expected",
    [
        (_response("hello"), "hello"),
        (json.dumps(_response("hello")), "hello"),
        ({"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}, "ab"),
        ("", None),
        (None, None),
        (float("nan"), None),
        ({"candidates": []}, None),
    ],
)
def test_response_text(value, expected):
    assert response_text(value) == expected


def test_parse_output_marks_failures_and_empty_responses():
    rows = pd.DataFrame(
        {
            "example_id": ["a", "b", "c"],
            "response": [_response("ok"), None, {"candidates": []}],
            "status": ["", "Bad Request: invalid role", ""],
        }
    )
    out = parse_output(rows, "example_id")
    assert out["output"].tolist() == ["ok", None, None]
    assert out["error"].tolist() == [None, "Bad Request: invalid role", "Empty response"]


def test_render_supports_dotted_names_and_json_values():
    row = {"attributes.input.value": "q", "meta": {"k": 1}, "empty": None}
    assert prompts.render("{attributes.input.value}|{meta}|{empty}", row) == 'q|{"k": 1}|'


def test_render_missing_column_is_explicit():
    with pytest.raises(KeyError, match="question"):
        prompts.render("{question}", {})


@pytest.mark.parametrize(
    "text,label,score,error",
    [
        ('{"label": "correct", "score": 1, "explanation": "x"}', "correct", 1.0, False),
        ('```json\n{"label": "incorrect", "score": "0"}\n```', "incorrect", 0.0, False),
        ("not json", None, None, True),
        ("[1, 2]", None, None, True),
        (None, None, None, True),
    ],
)
def test_parse_verdict(text, label, score, error):
    verdict = prompts.parse_verdict(text)
    assert verdict["label"] == label
    assert verdict["score"] == score
    assert (verdict["parse_error"] is not None) == error


def test_build_runs_merges_verdicts_and_keeps_failures():
    task = pd.DataFrame(
        {"example_id": ["a", "b"], "output": ["answer", None], "error": [None, "Bad Request"]}
    )
    verdicts = pd.DataFrame(
        {
            "example_id": ["a"],
            "output": ['{"label": "correct", "score": 1, "explanation": "fine"}'],
            "error": [None],
        }
    )
    runs = build_runs(task, verdicts, "correctness", "gemini-x").set_index("example_id")
    assert runs.loc["a", "correctness.label"] == "correct"
    assert runs.loc["a", "correctness.score"] == 1.0
    assert runs.loc["b", "output"] == ""
    assert runs.loc["b", "task_error"] == "Bad Request"
    assert pd.isna(runs.loc["b", "correctness.label"])
    assert (runs["model"] == "gemini-x").all()


def test_build_evals_drops_unusable_verdicts():
    verdicts = pd.DataFrame(
        {
            "span_id": ["s1", "s2", "s3"],
            "output": ['{"label": "correct", "score": 1}', "garbage", None],
            "error": [None, None, "Bad Request"],
        }
    )
    evals, skipped = build_evals(verdicts, "correctness")
    assert skipped == 2
    assert evals.to_dict("records") == [
        {
            "context.span_id": "s1",
            "eval.correctness.label": "correct",
            "eval.correctness.score": 1.0,
            "eval.correctness.explanation": None,
        }
    ]


def test_claude_request_shape():
    req = json.loads(claude_request("hi", system="sys"))
    assert req == {
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": CLAUDE_MAX_TOKENS,
        "messages": [{"role": "user", "content": "hi"}],
        "system": "sys",
    }


@pytest.mark.parametrize(
    "model,claude,resource",
    [
        ("gemini-2.5-flash", False, "gemini-2.5-flash"),
        ("claude-sonnet-5-5", True, "publishers/anthropic/models/claude-sonnet-5-5"),
        (
            "publishers/anthropic/models/claude-haiku-5-5",
            True,
            "publishers/anthropic/models/claude-haiku-5-5",
        ),
    ],
)
def test_model_routing(model, claude, resource):
    assert is_claude(model) is claude
    assert model_resource(model) == resource


def test_vertex_batch_builds_request_for_model():
    def vb(model):
        return VertexBatch(client=None, bq=None, model=model, dataset="p.d")

    gemini = json.loads(vb("gemini-2.5-flash").request("hi", json_output=True, temperature=0))
    claude = json.loads(vb("claude-sonnet-5-5").request("hi", json_output=True, temperature=0))
    assert "contents" in gemini
    assert "temperature" not in claude and claude["messages"][0]["content"] == "hi"


def _claude_response(*blocks, stop_reason="end_turn", **extra):
    return {"content": list(blocks), "stop_reason": stop_reason, **extra}


def test_response_text_reads_claude_text_blocks_only():
    response = _claude_response(
        {"type": "thinking", "thinking": ""},
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    )
    assert response_text(response) == "ab"
    assert response_text(json.dumps(response)) == "ab"
    assert response_text(_claude_response()) is None


def test_parse_output_reports_claude_refusals():
    rows = pd.DataFrame(
        {
            "example_id": ["a", "b", "c"],
            "response": [
                _claude_response({"type": "text", "text": "3"}),
                _claude_response(stop_reason="refusal", stop_details={"category": "cyber"}),
                _claude_response(stop_reason="refusal"),
            ],
            "status": ["", "", ""],
        }
    )
    out = parse_output(rows, "example_id")
    assert out["output"].tolist() == ["3", None, None]
    assert out["error"].tolist() == [None, "Refused (cyber)", "Refused"]
