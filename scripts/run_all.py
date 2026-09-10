"""One-command reproduction: fetch data -> process -> build golden set ->
train classifier -> build retrieval index -> evaluate -> judge replies.
Quickstart target: under 15 minutes on a laptop (no GPU, no API key needed)."""
import subprocess
import sys

STEPS = [
    ["python", "scripts/fetch_data.py"],
    ["python", "scripts/process_data.py"],
    ["python", "scripts/build_golden_set.py"],
    ["python", "src/intent_classifier.py"],
    ["python", "src/retrieval.py"],
    ["python", "scripts/evaluate.py"],
    ["python", "scripts/judge_replies.py"],
]

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
