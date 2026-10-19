"""Prompt templates. Edit these for your use case.

Placeholders are `{column name}` and are filled from the row being processed: a dataset
example (its columns), the task output (`{output}`), or an exported span (for example
`{attributes.input.value}`). Dotted names are allowed.
"""

import json
import re
from collections.abc import Mapping
from typing import Any

TASK_SYSTEM = (
    "You answer multiple-choice questions. Reply with the 0-based index of the correct "
    "choice only, as a single digit."
)
TASK_PROMPT = "{question}\n\nChoices (JSON list, 0-based):\n{choices}"

JUDGE_SYSTEM = (
    "You are grading the answer of an AI assistant to a multiple-choice question. "
    "The answer is correct if it selects the choice at the correct 0-based index. "
    'Reply with JSON only: {"label": "correct" | "incorrect", "score": 1 | 0, '
    '"explanation": "<one sentence>"}'
)
EXPERIMENT_JUDGE_PROMPT = (
    "Question:\n{question}\n\nChoices (JSON list, 0-based):\n{choices}\n\n"
    "Correct index:\n{answer}\n\nAssistant answer:\n{output}"
)
SPAN_JUDGE_PROMPT = (
    "User input:\n{attributes.input.value}\n\nAssistant output:\n{attributes.output.value}"
)

_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def render(template: str, row: Mapping[str, Any]) -> str:
    def value(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in row:
            raise KeyError(f"Template placeholder {{{key}}} has no matching column")
        v = row[key]
        if v is None:
            return ""
        if isinstance(v, dict | list):
            return json.dumps(v, ensure_ascii=False)
        return str(v)

    return _PLACEHOLDER.sub(value, template)


def placeholders(template: str) -> set[str]:
    return set(_PLACEHOLDER.findall(template))


def parse_verdict(text: str | None) -> dict[str, Any]:
    empty = {"label": None, "score": None, "explanation": None}
    if not text:
        return {**empty, "parse_error": "No judge output"}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned)
    try:
        verdict = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {**empty, "parse_error": f"Judge output is not JSON: {e}"}
    if not isinstance(verdict, Mapping):
        return {**empty, "parse_error": "Judge output is not a JSON object"}
    score = verdict.get("score")
    try:
        score = None if score is None else float(score)
    except (TypeError, ValueError):
        score = None
    label = verdict.get("label")
    return {
        "label": None if label is None else str(label),
        "score": score,
        "explanation": verdict.get("explanation"),
        "parse_error": None,
    }
