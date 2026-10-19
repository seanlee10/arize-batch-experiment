import argparse
from datetime import UTC, datetime, timedelta

from arize_batch_experiment import experiment, span_evals
from arize_batch_experiment.config import Settings


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-id",
        help="Name for this run; reuse it to resume. Defaults to a timestamp.",
    )
    parser.add_argument("--eval-name", default="correctness")
    parser.add_argument("--limit", type=int, help="Only use the first N rows (pilot runs)")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit or check the batch job and exit instead of waiting for it",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--resubmit-failed",
        action="store_true",
        help="Resume a failed, cancelled or expired batch job",
    )


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="arize-batch",
        description="Run Arize experiments and span evals through Vertex AI batch inference",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("experiment", help="Run a dataset experiment as a batch job")
    exp.add_argument("--dataset", required=True, help="Arize dataset name or ID")
    exp.add_argument("--experiment-name", help="Defaults to batch-<run id>")
    exp.add_argument("--no-judge", action="store_true", help="Skip the LLM judge stage")
    _common(exp)

    spans = sub.add_parser("span-evals", help="Judge production spans as a batch job")
    spans.add_argument("--project", required=True, help="Arize tracing project name")
    spans.add_argument("--start", type=_parse_time, help="ISO time; default 24 hours ago")
    spans.add_argument("--end", type=_parse_time, help="ISO time; default now")
    spans.add_argument("--where", default="", help="Span filter, same syntax as the Arize UI")
    spans.add_argument("--sample-rate", type=float)
    _common(spans)

    args = parser.parse_args()
    settings = Settings.from_env()
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    print(f"Run id: {run_id} (pass --run-id {run_id} to resume)")

    if args.command == "experiment":
        experiment.run(
            settings,
            experiment.ExperimentOptions(
                dataset=args.dataset,
                run_id=run_id,
                experiment_name=args.experiment_name or f"batch-{run_id}",
                eval_name=args.eval_name,
                judge=not args.no_judge,
                limit=args.limit,
                wait=not args.no_wait,
                poll_seconds=args.poll_seconds,
                resubmit_failed=args.resubmit_failed,
            ),
        )
    else:
        end = args.end or datetime.now(UTC)
        span_evals.run(
            settings,
            span_evals.SpanEvalOptions(
                project=args.project,
                run_id=run_id,
                start=args.start or end - timedelta(days=1),
                end=end,
                eval_name=args.eval_name,
                where=args.where,
                sample_rate=args.sample_rate,
                limit=args.limit,
                wait=not args.no_wait,
                poll_seconds=args.poll_seconds,
                resubmit_failed=args.resubmit_failed,
            ),
        )


if __name__ == "__main__":
    main()
