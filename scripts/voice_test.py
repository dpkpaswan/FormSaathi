"""Voice accuracy test script for FormSaathi.

Takes a folder of WAV files plus an expected-values CSV, calls /api/fill,
and prints the percentage of names, dates and phone numbers that were correct.

Usage:
    python scripts/voice_test.py \\
        --audio-dir ./test_audio \\
        --expected ./test_audio/expected.csv \\
        --api-url http://localhost:8000 \\
        --language ta \\
        --bank-id demo_bank

CSV format:
    filename,full_name,date_of_birth,mobile_number,father_or_spouse_name,...
    test1.wav,ராஜா குமார்,15-03-1990,9876543210,குமார்,...

Any column matching a form field id will be compared. Empty cells are skipped.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Voice accuracy test for FormSaathi /api/fill endpoint.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing WAV test audio files.",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        required=True,
        help="CSV file with expected field values per audio file.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the FormSaathi API (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "ta"],
        default="ta",
        help="Language code for the audio (default: ta).",
    )
    parser.add_argument(
        "--bank-id",
        type=str,
        default="demo_bank",
        help="Bank form ID (default: demo_bank).",
    )
    return parser.parse_args()


def normalize(value: str) -> str:
    """Normalize a value for comparison (strip, lowercase, collapse spaces)."""
    return " ".join(value.strip().lower().split())


def call_fill_api(
    api_url: str,
    audio_path: Path,
    language: str,
    bank_id: str,
) -> dict:
    """Send an audio file to /api/fill and return the response JSON.

    Args:
        api_url: Base API URL.
        audio_path: Path to the WAV file.
        language: Language code.
        bank_id: Bank form ID.

    Returns:
        Parsed JSON response dict.
    """
    url = f"{api_url}/api/fill"
    with open(audio_path, "rb") as f:
        files = {"audio": (audio_path.name, f, "audio/wav")}
        data = {"language": language, "bank_id": bank_id}
        response = requests.post(url, files=files, data=data, timeout=90)

    response.raise_for_status()
    return response.json()


def evaluate_results(
    expected_rows: list[dict[str, str]],
    actual_results: list[dict],
    key_fields: list[str],
) -> dict[str, dict[str, int]]:
    """Compare expected vs actual extracted fields.

    Args:
        expected_rows: List of expected value dicts (from CSV).
        actual_results: List of API response dicts.
        key_fields: Field IDs to evaluate.

    Returns:
        Dict of field_id -> {"correct": int, "total": int}.
    """
    scores: dict[str, dict[str, int]] = {
        f: {"correct": 0, "total": 0} for f in key_fields
    }

    for expected, actual in zip(expected_rows, actual_results):
        fields = actual.get("fields", {})
        for field_id in key_fields:
            expected_val = expected.get(field_id, "").strip()
            if not expected_val:
                continue  # Skip empty expected values

            scores[field_id]["total"] += 1

            actual_field = fields.get(field_id, {})
            actual_val = actual_field.get("value", "") or ""

            if normalize(expected_val) == normalize(actual_val):
                scores[field_id]["correct"] += 1

    return scores


def main() -> None:
    """Run the voice accuracy test."""
    args = parse_args()

    # Validate paths
    if not args.audio_dir.is_dir():
        print(f"ERROR: Audio directory not found: {args.audio_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.expected.is_file():
        print(f"ERROR: Expected CSV not found: {args.expected}", file=sys.stderr)
        sys.exit(1)

    # Read expected values
    with open(args.expected, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected_rows = list(reader)

    if not expected_rows:
        print("ERROR: CSV is empty.", file=sys.stderr)
        sys.exit(1)

    # Key fields to track (names, dates, phones)
    key_fields = [
        "full_name",
        "father_or_spouse_name",
        "date_of_birth",
        "mobile_number",
        "nominee_name",
    ]

    # Available fields from CSV headers (intersect with key_fields)
    csv_fields = set(expected_rows[0].keys())
    active_fields = [f for f in key_fields if f in csv_fields]

    if not active_fields:
        print(
            f"WARNING: No key fields found in CSV. "
            f"CSV columns: {list(csv_fields)}",
            file=sys.stderr,
        )
        active_fields = [f for f in csv_fields if f != "filename"]

    print(f"Testing {len(expected_rows)} audio file(s)")
    print(f"Evaluating fields: {', '.join(active_fields)}")
    print(f"API: {args.api_url}")
    print(f"Language: {args.language}")
    print("-" * 60)

    # Process each audio file
    actual_results: list[dict] = []
    errors: list[str] = []

    for i, row in enumerate(expected_rows):
        filename = row.get("filename", "").strip()
        if not filename:
            print(f"  Row {i + 1}: skipped (no filename)")
            continue

        audio_path = args.audio_dir / filename
        if not audio_path.is_file():
            msg = f"  Row {i + 1}: {filename} — FILE NOT FOUND"
            print(msg)
            errors.append(msg)
            actual_results.append({})
            continue

        print(f"  Row {i + 1}: {filename} ... ", end="", flush=True)

        try:
            result = call_fill_api(
                api_url=args.api_url,
                audio_path=audio_path,
                language=args.language,
                bank_id=args.bank_id,
            )
            actual_results.append(result)
            print("OK")

            # Show transcript
            transcript = result.get("transcript", "(no transcript)")
            print(f"    Transcript: {transcript[:100]}...")

        except Exception as exc:
            msg = f"API ERROR: {exc}"
            print(msg)
            errors.append(f"  {filename}: {msg}")
            actual_results.append({})

    # Evaluate
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    scores = evaluate_results(expected_rows, actual_results, active_fields)

    total_correct = 0
    total_tested = 0

    for field_id, score in scores.items():
        correct = score["correct"]
        total = score["total"]
        pct = (correct / total * 100) if total > 0 else 0.0
        total_correct += correct
        total_tested += total
        print(f"  {field_id:30s}  {correct:3d}/{total:3d}  ({pct:5.1f}%)")

    overall_pct = (total_correct / total_tested * 100) if total_tested > 0 else 0.0
    print("-" * 60)
    print(f"  {'OVERALL':30s}  {total_correct:3d}/{total_tested:3d}  ({overall_pct:5.1f}%)")

    if errors:
        print(f"\n{len(errors)} error(s) encountered.")

    # Exit code: 0 if ≥80% overall, 1 otherwise
    sys.exit(0 if overall_pct >= 80.0 else 1)


if __name__ == "__main__":
    main()
