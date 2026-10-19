import json

import pandas as pd
import pytest

from arize_batch_experiment import prompts
from arize_batch_experiment.experiment import build_runs
from arize_batch_experiment.span_evals import build_evals
from arize_batch_experiment.vertex_batch import gemini_request, parse_output, response_text


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
