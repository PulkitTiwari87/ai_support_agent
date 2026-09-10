"""One-command reproduction: fetch data -> process -> build golden set ->
train classifier -> build retrieval index -> evaluate -> judge replies.
Quickstart target: under 15 minutes on a laptop (no GPU, no API key needed)."""
import subprocess
import sys

STEPS = [
    [sys.executable, "scripts/fetch_data.py"],
    [sys.executable, "scripts/process_data.py"],
    [sys.executable, "scripts/build_golden_set.py"],
    [sys.executable, "src/intent_classifier.py"],
    [sys.executable, "src/retrieval.py"],
    [sys.executable, "scripts/evaluate.py"],
    [sys.executable, "scripts/judge_replies.py"],
]
# sys.executable, not a bare "python" on PATH: a hardcoded "python" silently
# picked up a different, unrelated system interpreter (with an unpinned,
# older scikit-learn) instead of the project's own .venv in this
# environment -- found this session (planning/18_DECISION_LOG.md). Using
# sys.executable guarantees every step runs under whatever interpreter
# actually launched run_all.py.

def main():
    for step in STEPS:
        print(f"\n$ {' '.join(step)}")
        r = subprocess.run(step)
        if r.returncode != 0:
            print(f"FAILED: {step}")
            sys.exit(1)
    print("\nAll steps completed. See data/processed/eval_results.json for headline metrics.")

if __name__ == "__main__":
    main()
