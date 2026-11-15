"""SUBTLEX-US Ingestion and Normalization Script.

Processes the raw 9-column SUBTLEX-US dataset into a sanitized,
deterministic, 2-column TSV dictionary of top N words (default: 25,000).

Strict adherence to Phase 3.10.3 requirements:
- Read 9-column format: Word, FREQcount, ...
- Only tokens matching regex ^[A-Za-z]+$
- Uppercase normalization
- Valid single-letter words (A, I) preserved
- Contractions with apostrophes excluded
- Sum frequencies upon case collision/duplicate with logging
- Deterministic sort key: (-frequency, word)
- Exactly top 25,000 words
- TSV format: WORD<TAB>INTEGER_FREQUENCY (no header)
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ingest_subtlex")

ALPHA_REGEX = re.compile(r"^[A-Za-z]+$")


def ingest_subtlex(
    input_path: str | Path,
    output_path: str | Path,
    limit: int = 25000,
) -> dict:
    """Ingest, filter, aggregate, sort, and write SUBTLEX-US dictionary.

    Args:
        input_path: Path to the raw SUBTLEX-US 9-column text/tsv file.
        output_path: Path to the destination 2-column TSV file.
        limit: Maximum number of top words to write (default: 25,000).

    Returns:
        Summary statistics dictionary.
    """
    in_file = Path(input_path)
    out_file = Path(output_path)

    if not in_file.is_file():
        raise FileNotFoundError(f"Source file not found: {in_file}")

    logger.info(f"Starting ingestion from: {in_file}")
    logger.info(f"Target limit: {limit} words")

    total_lines = 0
    header_skipped = 0
    invalid_tokens = 0
    malformed_lines = 0
    valid_entries = 0
    collisions = 0

    # Store aggregated frequencies: word_upper -> total_freq
    aggregated: dict[str, int] = {}
    collision_details: list[tuple[str, str, int, int]] = []

    with open(in_file, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total_lines += 1

            # Split on tab
            parts = line.split("\t")
            if len(parts) < 2:
                # Try comma or whitespace if tab split fails
                malformed_lines += 1
                logger.warning(f"Line {line_num} malformed (less than 2 parts): {line[:50]}")
                continue

            raw_word = parts[0].strip()
            raw_freq = parts[1].strip()

            # Detect header row
            if line_num == 1 and raw_word.lower() == "word" and "freq" in raw_freq.lower():
                header_skipped += 1
                logger.info(f"Detected and skipped header at line 1: {parts[:4]}")
                continue

            # Validate alphabetic characters only: ^[A-Za-z]+$
            if not ALPHA_REGEX.match(raw_word):
                invalid_tokens += 1
                logger.debug(f"Skipping non-alphabetic token '{raw_word}' at line {line_num}")
                continue

            # Parse frequency as positive integer
            try:
                freq = int(raw_freq)
                if freq < 0:
                    freq = 0
            except ValueError:
                malformed_lines += 1
                logger.warning(f"Malformed frequency '{raw_freq}' for word '{raw_word}' at line {line_num}")
                continue

            valid_entries += 1
            word_upper = raw_word.upper()

            if word_upper in aggregated:
                collisions += 1
                prev_freq = aggregated[word_upper]
                aggregated[word_upper] += freq
                collision_details.append((word_upper, raw_word, prev_freq, freq))
                logger.info(
                    f"Collision for '{word_upper}': prev={prev_freq}, add={freq}, total={aggregated[word_upper]}"
                )
            else:
                aggregated[word_upper] = freq

    logger.info(f"Total lines read: {total_lines}")
    logger.info(f"Header lines skipped: {header_skipped}")
    logger.info(f"Valid entries processed: {valid_entries}")
    logger.info(f"Invalid tokens rejected: {invalid_tokens}")
    logger.info(f"Malformed lines rejected: {malformed_lines}")
    logger.info(f"Unique uppercase words in corpus: {len(aggregated)}")
    logger.info(f"Collisions merged: {collisions}")

    # Deterministic sort: key = (-frequency, word)
    logger.info("Sorting words deterministically by (-frequency, word)...")
    sorted_words = sorted(
        aggregated.items(),
        key=lambda item: (-item[1], item[0])
    )

    # Slice to top `limit`
    selected_words = sorted_words[:limit]
    logger.info(f"Selected {len(selected_words)} words (limit: {limit})")

    # Ensure parent directory exists
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Write output TSV (no header)
    logger.info(f"Writing output TSV to: {out_file}")
    with open(out_file, "w", encoding="utf-8", newline="\n") as out_f:
        for word, freq in selected_words:
            out_f.write(f"{word}\t{freq}\n")

    logger.info(f"Successfully generated {out_file} with {len(selected_words)} entries.")

    return {
        "input_path": str(in_file),
        "output_path": str(out_file),
        "total_lines": total_lines,
        "header_skipped": header_skipped,
        "valid_entries": valid_entries,
        "invalid_tokens": invalid_tokens,
        "malformed_lines": malformed_lines,
        "unique_corpus_words": len(aggregated),
        "collisions_merged": collisions,
        "selected_word_count": len(selected_words),
        "top_10": selected_words[:10],
        "tail_5": selected_words[-5:] if len(selected_words) >= 5 else selected_words,
    }


def main() -> None:
    """CLI entrypoint for ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest and clean SUBTLEX-US dataset into 2-column TSV dictionary."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/dictionary/source/SUBTLEXus74286wordstextversion.txt",
        help="Path to raw SUBTLEX-US text file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dictionary/english_words_25k.tsv",
        help="Path to output TSV dictionary file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25000,
        help="Maximum number of top words to retain (default: 25000).",
    )

    args = parser.parse_args()

    # Resolve relative to project root if needed
    project_root = Path(__file__).resolve().parents[1]
    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = project_root / in_path

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = project_root / out_path

    stats = ingest_subtlex(in_path, out_path, args.limit)
    print("\n--- INGESTION SUMMARY ---")
    print(f"Source: {stats['input_path']}")
    print(f"Destination: {stats['output_path']}")
    print(f"Corpus Unique Words: {stats['unique_corpus_words']}")
    print(f"Collisions Merged: {stats['collisions_merged']}")
    print(f"Selected Words: {stats['selected_word_count']}")
    print("\nTop 5 Words:")
    for w, f in stats["top_10"][:5]:
        print(f"  {w}: {f}")
    print("\nTail 5 Words:")
    for w, f in stats["tail_5"]:
        print(f"  {w}: {f}")


if __name__ == "__main__":
    main()
