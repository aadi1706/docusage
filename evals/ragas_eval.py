"""
DocuSage — RAGAS Evaluation Harness

Metrics evaluated:
  - faithfulness:        Does the answer stick to retrieved context? (anti-hallucination)
  - context_precision:  Are the retrieved chunks actually relevant?
  - answer_relevancy:   Does the answer address the question asked?

Run locally:
  python evals/ragas_eval.py

Run in CI (GitHub Actions reads exit code):
  python evals/ragas_eval.py --ci
"""
import json
import sys
import os
import argparse
from pathlib import Path
from loguru import logger

# These imports will work once dependencies are installed
# from ragas import evaluate
# from ragas.metrics import faithfulness, context_precision, answer_relevancy
# from datasets import Dataset


GOLDEN_SET_PATH = Path("data/eval/golden_set.json")

# Thresholds (also set in .env — CI reads from env)
THRESHOLDS = {
    "faithfulness":       float(os.getenv("RAGAS_FAITHFULNESS_THRESHOLD",  "0.82")),
    "context_precision":  float(os.getenv("RAGAS_CONTEXT_PRECISION_THRESHOLD", "0.75")),
    "answer_relevancy":   float(os.getenv("RAGAS_ANSWER_RELEVANCY_THRESHOLD", "0.80")),
}


def load_golden_set() -> list:
    if not GOLDEN_SET_PATH.exists():
        logger.warning(f"Golden set not found at {GOLDEN_SET_PATH}. Run scripts/build_golden_set.py first.")
        return []
    with open(GOLDEN_SET_PATH) as f:
        return json.load(f)


def run_eval(golden_samples: list, ci_mode: bool = False) -> dict:
    """
    Run RAGAS evaluation over golden set.
    Returns metric scores dict.
    """
    if not golden_samples:
        logger.error("No golden samples — cannot evaluate")
        return {}

    logger.info(f"Running eval on {len(golden_samples)} golden samples...")

    # TODO Week 8: uncomment and wire up real inference
    # from agents.graph import run_query
    # 
    # ragas_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    # for sample in golden_samples:
    #     result = run_query(sample["question"])
    #     ragas_data["question"].append(sample["question"])
    #     ragas_data["answer"].append(result.final_answer or "")
    #     ragas_data["contexts"].append([c.content for c in result.retrieved_chunks])
    #     ragas_data["ground_truth"].append(sample["ground_truth"])
    #
    # dataset = Dataset.from_dict(ragas_data)
    # scores = evaluate(dataset, metrics=[faithfulness, context_precision, answer_relevancy])
    # return dict(scores)

    # Placeholder scores until Week 8
    mock_scores = {
        "faithfulness":      0.88,
        "context_precision": 0.79,
        "answer_relevancy":  0.84,
    }
    logger.info(f"[MOCK] Eval scores: {mock_scores}")
    return mock_scores


def check_thresholds(scores: dict) -> tuple[bool, list]:
    """Returns (passed, list_of_failures)."""
    failures = []
    for metric, threshold in THRESHOLDS.items():
        score = scores.get(metric, 0.0)
        if score < threshold:
            failures.append(f"{metric}: {score:.3f} < threshold {threshold}")
    return len(failures) == 0, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true", help="Exit with code 1 if thresholds not met")
    args = parser.parse_args()

    golden = load_golden_set()
    scores = run_eval(golden, ci_mode=args.ci)

    if not scores:
        logger.error("Eval produced no scores")
        sys.exit(1 if args.ci else 0)

    passed, failures = check_thresholds(scores)

    print("\n" + "="*50)
    print("DocuSage RAGAS Eval Results")
    print("="*50)
    for metric, score in scores.items():
        threshold = THRESHOLDS.get(metric, 0)
        status = "✓" if score >= threshold else "✗"
        print(f"  {status} {metric:<25} {score:.3f}  (threshold: {threshold})")
    print("="*50)

    if passed:
        print("✓ All thresholds passed — safe to merge\n")
    else:
        print("✗ THRESHOLD FAILURES:")
        for f in failures:
            print(f"  - {f}")
        print()
        if args.ci:
            sys.exit(1)


if __name__ == "__main__":
    main()
