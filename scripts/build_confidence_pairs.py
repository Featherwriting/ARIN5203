import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

OPTION_PREFIX_RE = re.compile(r"^[A-Z]\s*[\.:\)]\s*")
SKIP_ANSWER_TEXT = {"i don't know", "none of the above"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build confidence/unconfidence pairs from multiple-choice predictions."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more JSONL files containing model predictions with probabilities.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional list of dataset names to keep (matches the 'dataset' field).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum probability threshold for the confident answer.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination CSV path for the generated dataset.",
    )
    return parser.parse_args()


def iter_records(paths: Iterable[str]) -> Iterator[Dict]:
    for path in paths:
        input_path = Path(path)
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Failed to parse JSON in {input_path} at line {line_number}: {exc}"
                    ) from exc


def normalize_choice(choice_text: Optional[str]) -> str:
    if choice_text is None:
        return ""
    text = str(choice_text).strip()
    return OPTION_PREFIX_RE.sub("", text).strip()


def select_answers(
    record: Dict,
    min_confidence: float,
    dataset_filter: Optional[List[str]] = None,
) -> Optional[Dict]:
    dataset_name = record.get("dataset") or record.get("source")
    if dataset_filter and dataset_name not in dataset_filter:
        return None

    choices = record.get("choices") or {}
    probabilities = record.get("probabilities") or {}
    if not choices or not probabilities:
        return None

    overlapping_keys = [key for key in probabilities if key in choices]
    if not overlapping_keys:
        return None

    normalized_choices = {key: normalize_choice(choices[key]) for key in overlapping_keys}
    filtered_keys = [
        key for key in overlapping_keys if normalized_choices[key].casefold() not in SKIP_ANSWER_TEXT
    ]
    if len(filtered_keys) < 2:
        return None

    scored_keys = {key: float(probabilities[key]) for key in filtered_keys}

    top_key = max(scored_keys, key=scored_keys.get)
    bottom_key = min(scored_keys, key=scored_keys.get)

    top_prob = scored_keys[top_key]
    bottom_prob = scored_keys[bottom_key]

    if top_prob < min_confidence:
        return None

    question = str(record.get("question", "")).strip()
    confident_answer = normalized_choices[top_key]
    unconfident_answer = normalized_choices[bottom_key]

    if not question or not confident_answer or not unconfident_answer:
        return None

    return {
        "question": question,
        "confident_answer": confident_answer,
        "unconfident_answer": unconfident_answer,
        "dataset": dataset_name,
        "sample_id": record.get("sample_id"),
        "confident_probability": f"{top_prob:.6f}",
        "unconfident_probability": f"{bottom_prob:.6f}",
    }


def main() -> None:
    args = parse_args()
    dataset_filter = args.datasets if args.datasets else None

    rows: List[Dict] = []
    seen = kept = 0

    for record in iter_records(args.inputs):
        seen += 1
        result = select_answers(record, args.min_confidence, dataset_filter)
        if result is None:
            continue
        rows.append(result)
        kept += 1

    if not rows:
        raise RuntimeError("No records met the filtering criteria.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "question",
        "confident_answer",
        "unconfident_answer",
        "dataset",
        "sample_id",
        "confident_probability",
        "unconfident_probability",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {seen} records, kept {kept} with confidence >= {args.min_confidence}.")
    print(f"Saved dataset with {len(rows)} rows to {output_path}.")


if __name__ == "__main__":
    main()
