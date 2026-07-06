"""
Script to help build the golden evaluation set.
Run this interactively to annotate real documents.

Usage:
  python scripts/build_golden_set.py --doc data/raw/RBI_Circular_2024.pdf
"""
import json
import argparse
from pathlib import Path

GOLDEN_PATH = Path("data/eval/golden_set.json")

def add_sample():
    print("\n=== Add Golden Eval Sample ===")
    sample = {
        "id": f"gs_{Path(GOLDEN_PATH).exists() and len(json.loads(GOLDEN_PATH.read_text())) + 1:03d}",
        "question":    input("Question: "),
        "ground_truth": input("Ground truth answer: "),
        "source_doc":  input("Source doc filename: "),
        "page_numbers": list(map(int, input("Page numbers (comma-sep): ").split(","))),
        "category":    input("Category (numeric_extraction/multi_page_synthesis/table_extraction): "),
        "difficulty":  input("Difficulty (easy/medium/hard): "),
        "notes":       input("Notes: "),
    }

    existing = json.loads(GOLDEN_PATH.read_text()) if GOLDEN_PATH.exists() else []
    existing.append(sample)
    GOLDEN_PATH.write_text(json.dumps(existing, indent=2))
    print(f"✓ Saved. Total samples: {len(existing)}")

if __name__ == "__main__":
    add_sample()
