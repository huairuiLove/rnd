"""Validate and preprocess the 54-label AAPD dataset used by CoMAL."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from nltk.tokenize import word_tokenize
from tqdm import tqdm
from transformers import BertTokenizer


STANDARD_SPLIT_SIZES = {"train": 53_840, "val": 1_000, "test": 1_000}
PAPER_SPLIT_SIZES = {"train": 45_840, "val": 5_000, "test": 5_000}
EXPECTED_LABEL_COUNT = 54
SOURCE_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Example:
    text: str
    labels: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the 54-label AAPD files expected by CoMAL. The source directory "
            "must contain text_{train,val,test} and label_{train,val,test}."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/aapd_54/raw"),
        help="Directory containing the standard 53,840/1,000/1,000 AAPD split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/aapd_54"),
        help="Directory for the paper split and classifier JSON files.",
    )
    parser.add_argument(
        "--bert-path",
        type=Path,
        default=Path("bert/bert-base-uncased"),
        help="Local BERT tokenizer directory.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the source files and print statistics without writing output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace previously generated split and JSON files in output-dir.",
    )
    return parser.parse_args()


def read_split(source_dir: Path, split: str) -> list[Example]:
    text_path = source_dir / f"text_{split}"
    label_path = source_dir / f"label_{split}"
    missing_paths = [path for path in (text_path, label_path) if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing AAPD source file(s): " + ", ".join(map(str, missing_paths))
        )

    with text_path.open(encoding="utf-8") as text_stream:
        texts = [line.rstrip("\n") for line in text_stream]
    with label_path.open(encoding="utf-8") as label_stream:
        labels = [tuple(line.strip().split()) for line in label_stream]

    if len(texts) != len(labels):
        raise ValueError(
            f"{split}: text/label line count mismatch ({len(texts)} != {len(labels)})"
        )
    if len(texts) != STANDARD_SPLIT_SIZES[split]:
        raise ValueError(
            f"{split}: expected {STANDARD_SPLIT_SIZES[split]:,} examples, "
            f"found {len(texts):,}"
        )

    examples = []
    for line_number, (text, one_example_labels) in enumerate(
        zip(texts, labels), start=1
    ):
        if not text.strip():
            raise ValueError(f"{text_path}:{line_number}: empty text")
        if not one_example_labels:
            raise ValueError(f"{label_path}:{line_number}: empty label list")
        examples.append(Example(text=text.strip(), labels=one_example_labels))
    return examples


def load_and_validate_source(source_dir: Path) -> list[Example]:
    all_examples = []
    for split in SOURCE_SPLITS:
        all_examples.extend(read_split(source_dir, split))

    unique_labels = {label for example in all_examples for label in example.labels}
    if len(unique_labels) != EXPECTED_LABEL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LABEL_COUNT} unique labels, found {len(unique_labels)}"
        )
    return all_examples


def print_statistics(examples: Sequence[Example], prefix: str) -> None:
    label_total = sum(len(example.labels) for example in examples)
    unique_labels = {label for example in examples for label in example.labels}
    print(
        f"{prefix}: examples={len(examples):,}, labels={len(unique_labels)}, "
        f"labels/example={label_total / len(examples):.5f}"
    )


def make_paper_splits(
    examples: Sequence[Example], seed: int
) -> dict[str, list[Example]]:
    expected_total = sum(PAPER_SPLIT_SIZES.values())
    if len(examples) != expected_total:
        raise ValueError(
            f"Paper split needs {expected_total:,} examples, found {len(examples):,}"
        )

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    test_end = PAPER_SPLIT_SIZES["test"]
    validation_end = test_end + PAPER_SPLIT_SIZES["val"]
    splits = {
        "test": shuffled[:test_end],
        "val": shuffled[test_end:validation_end],
        "train": shuffled[validation_end:],
    }
    for split, expected_size in PAPER_SPLIT_SIZES.items():
        if len(splits[split]) != expected_size:
            raise AssertionError(
                f"{split}: expected {expected_size:,}, got {len(splits[split]):,}"
            )
    return splits


def generated_paths(output_dir: Path, max_length: int) -> list[Path]:
    raw_split_paths = [
        output_dir / f"{kind}_{split}"
        for split in SOURCE_SPLITS
        for kind in ("text", "label")
    ]
    classifier_paths = [
        output_dir / f"clf_{split}_data_{max_length}.json"
        for split in SOURCE_SPLITS
    ]
    return raw_split_paths + classifier_paths + [
        output_dir / "label_freq.json",
        output_dir / "split_manifest.json",
    ]


def refuse_implicit_overwrite(
    output_dir: Path, max_length: int, force: bool
) -> None:
    existing_paths = [
        path for path in generated_paths(output_dir, max_length) if path.exists()
    ]
    if existing_paths and not force:
        preview = ", ".join(map(str, existing_paths[:3]))
        raise FileExistsError(
            f"Generated files already exist ({preview}). Pass --force to replace them."
        )


def write_raw_splits(
    output_dir: Path, splits: dict[str, list[Example]]
) -> None:
    for split in SOURCE_SPLITS:
        text_path = output_dir / f"text_{split}"
        label_path = output_dir / f"label_{split}"
        with text_path.open("w", encoding="utf-8") as text_stream, label_path.open(
            "w", encoding="utf-8"
        ) as label_stream:
            for example in splits[split]:
                text_stream.write(example.text + "\n")
                label_stream.write(" ".join(example.labels) + "\n")


def normalize_for_repo_tokenizer(text: str, max_length: int) -> str:
    tokens = [
        token.lower()
        for token in word_tokenize(text)
        if re.sub(r"[^\w]", "", token)
    ]
    return " ".join(tokens[:max_length])


def encode_examples(
    examples: Iterable[Example], tokenizer: BertTokenizer, max_length: int, split: str
) -> list[dict[str, object]]:
    encoded_examples = []
    for example in tqdm(examples, desc=f"Tokenizing {split}"):
        normalized_text = normalize_for_repo_tokenizer(example.text, max_length)
        input_tokens = tokenizer.tokenize(normalized_text)[:max_length]
        encoded_examples.append(
            {
                "input_ids": tokenizer.convert_tokens_to_ids(input_tokens),
                "label": list(example.labels),
            }
        )
    return encoded_examples


def write_classifier_data(
    output_dir: Path,
    splits: dict[str, list[Example]],
    tokenizer: BertTokenizer,
    max_length: int,
) -> None:
    for split in SOURCE_SPLITS:
        encoded_examples = encode_examples(
            splits[split], tokenizer, max_length, split
        )
        output_path = output_dir / f"clf_{split}_data_{max_length}.json"
        with output_path.open("w", encoding="utf-8") as output_stream:
            json.dump(
                encoded_examples,
                output_stream,
                ensure_ascii=False,
                separators=(",", ":"),
            )


def label_frequencies(examples: Iterable[Example]) -> list[tuple[str, int]]:
    counts = Counter(
        label for example in examples for label in example.labels
    )
    return counts.most_common()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_metadata(
    source_dir: Path,
    output_dir: Path,
    splits: dict[str, list[Example]],
    seed: int,
    max_length: int,
    bert_path: Path,
) -> None:
    all_examples = [example for split in SOURCE_SPLITS for example in splits[split]]
    frequencies = label_frequencies(all_examples)
    with (output_dir / "label_freq.json").open("w", encoding="utf-8") as stream:
        json.dump(frequencies, stream, ensure_ascii=False, indent=2)

    source_hashes = {
        path.name: sha256(path)
        for split in SOURCE_SPLITS
        for path in (source_dir / f"text_{split}", source_dir / f"label_{split}")
    }
    manifest = {
        "source": {
            "url": "https://git.uwaterloo.ca/jimmylin/Castor-data/-/tree/master/datasets/AAPD",
            "commit": "16599a322e6bdeb8d4a72ea982dc3a309c3f85ce",
            "standard_split_sizes": STANDARD_SPLIT_SIZES,
            "sha256": source_hashes,
        },
        "paper_split_sizes": PAPER_SPLIT_SIZES,
        "split_seed": seed,
        "split_algorithm": (
            "Concatenate standard train/val/test, random.Random(seed).shuffle, "
            "then take test, validation, and remaining train examples."
        ),
        "label_count": EXPECTED_LABEL_COUNT,
        "average_labels_per_example": sum(
            len(example.labels) for example in all_examples
        )
        / len(all_examples),
        "tokenizer": str(bert_path),
        "max_length": max_length,
    }
    with (output_dir / "split_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if args.max_length < 3:
        raise ValueError("--max-length must be at least 3")

    examples = load_and_validate_source(args.source_dir)
    print_statistics(examples, "Standard AAPD source")
    if args.validate_only:
        return

    refuse_implicit_overwrite(args.output_dir, args.max_length, args.force)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = make_paper_splits(examples, args.seed)
    for split in SOURCE_SPLITS:
        print_statistics(splits[split], f"Paper {split}")

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    write_raw_splits(args.output_dir, splits)
    write_classifier_data(args.output_dir, splits, tokenizer, args.max_length)
    write_metadata(
        args.source_dir,
        args.output_dir,
        splits,
        args.seed,
        args.max_length,
        args.bert_path,
    )
    print(f"AAPD preprocessing complete: {args.output_dir}")


if __name__ == "__main__":
    main()
