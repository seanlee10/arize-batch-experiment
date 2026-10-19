"""Create a small sample dataset in Arize whose columns match the default prompts."""

import argparse
import csv
from pathlib import Path

from arize_batch_experiment.clients import arize_client
from arize_batch_experiment.config import Settings

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_questions.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="batch-experiment-sample")
    parser.add_argument("--csv", type=Path, default=SAMPLE)
    args = parser.parse_args()

    settings = Settings.from_env()
    with args.csv.open(newline="", encoding="utf-8") as f:
        examples = list(csv.DictReader(f))

    dataset = arize_client(settings).datasets.create(
        name=args.name, space=settings.arize_space_id, examples=examples
    )
    print(f"Created dataset {args.name!r} ({dataset.id}) with {len(examples)} examples")


if __name__ == "__main__":
    main()
