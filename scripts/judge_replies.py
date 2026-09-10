"""Runs the heuristic judge over system predictions on the golden set.
Attempts the LLM judge first if ANTHROPIC_API_KEY is set; falls back to
heuristic and records which path actually ran."""
import os
import json
import sys
sys.path.insert(0, "src")
import pandas as pd
from judge import heuristic_judge, llm_judge
from retrieval import load_index, retrieve

def main():
    gold = pd.read_csv("data/processed/golden_set.csv")
    preds = pd.read_csv("data/processed/system_predictions.csv")
    df = pd.concat([gold[["pair_id", "customer_msg"]], preds], axis=1)
    vectorizer, matrix, corpus = load_index()

    used_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    scores = []
    for _, row in df.iterrows():
        evidence = retrieve(row["customer_msg"], vectorizer, matrix, corpus, k=3)
        if used_llm:
            try:
                s = llm_judge(row["customer_msg"], row["draft_reply"], evidence)
            except Exception:
                used_llm = False
                s = heuristic_judge(row["customer_msg"], row["draft_reply"], evidence)
        else:
            s = heuristic_judge(row["customer_msg"], row["draft_reply"], evidence)
        scores.append(s)

    method = "llm" if used_llm else "heuristic"
    n_grounded_scored = sum(s.get("grounded_score", 0) for s in scores if "grounded_score" in s)
    n_length_ok = sum(1 for s in scores if s.get("length_ok"))
    n_with_unsupported_claims = sum(1 for s in scores if s.get("unsupported_claims"))

    summary = {
        "judge_method_actually_used": method,
        "note": "LLM judge requires ANTHROPIC_API_KEY; not configured in this environment -- heuristic judge used instead."
                if method == "heuristic" else "LLM judge ran successfully.",
        "n": len(scores),
        "heuristic_grounded_count": n_grounded_scored if method == "heuristic" else None,
        "heuristic_length_ok_count": n_length_ok if method == "heuristic" else None,
        "heuristic_replies_with_unsupported_claims": n_with_unsupported_claims if method == "heuristic" else None,
        "unsupported_claims_caveat": (
            "This check is close to vacuous for the current extractive generator "
            "(the reply IS the evidence by construction) -- it becomes a real "
            "signal only once the LLM paraphrasing path actually runs. See "
            "src/judge.py::_unsupported_claims."
        ) if method == "heuristic" else None,
    }
    print(json.dumps(summary, indent=2))
    with open("data/processed/judge_results.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
