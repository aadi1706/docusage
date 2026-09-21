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

from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, answer_relevancy
from datasets import Dataset


GOLDEN_SET_PATH = Path("data/eval/golden_set.json")

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
    """Run RAGAS evaluation over golden set. Returns metric scores dict."""
    if not golden_samples:
        logger.error("No golden samples — cannot evaluate")
        return {}

    logger.info(f"Running eval on {len(golden_samples)} golden samples...")

    from agents.graph import run_query

    ragas_data: dict = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for i, sample in enumerate(golden_samples):
        logger.info(f"  [{i+1}/{len(golden_samples)}] {sample['id']}: {sample['question'][:60]}...")
        try:
            result = run_query(sample["question"])
            answer = result.final_answer or ""
            contexts = [c.content for c in result.retrieved_chunks if c.content.strip()]
        except Exception as e:
            logger.warning(f"    run_query failed for {sample['id']}: {e} — using empty answer")
            answer = ""
            contexts = []

        # RAGAS requires at least one non-empty context string
        if not contexts:
            contexts = ["[no context retrieved]"]

        ragas_data["question"].append(sample["question"])
        ragas_data["answer"].append(answer)
        ragas_data["contexts"].append(contexts)
        ragas_data["ground_truth"].append(sample["ground_truth"])

    dataset = Dataset.from_dict(ragas_data)
    logger.info("Calling RAGAS evaluate (uses OpenAI) …")

    result = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, answer_relevancy],
        raise_exceptions=False,
    )

    scores = {
        "faithfulness":      float(result["faithfulness"]),
        "context_precision": float(result["context_precision"]),
        "answer_relevancy":  float(result["answer_relevancy"]),
    }
    logger.info(f"Eval scores: {scores}")
    return scores


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
