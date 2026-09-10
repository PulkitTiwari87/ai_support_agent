"""Build the frozen golden evaluation set from data/processed/eval_pool.csv.

Labels intent and escalation via the rule-based taxonomy in src/taxonomy.py.
Output is frozen at data/processed/golden_set.csv and must not be edited
after the spot-check pass documented in planning/05_GOLDEN_SET.md.
"""
import sys
sys.path.insert(0, "src")
import pandas as pd
from taxonomy import classify_intent, classify_escalation


def main():
    df = pd.read_csv("data/processed/eval_pool.csv")
    intents, esc, reasons = [], [], []
    for _, row in df.iterrows():
        intent = classify_intent(row["customer_msg"])
        should_esc, reason = classify_escalation(row["customer_msg"], intent)
        intents.append(intent)
        esc.append(should_esc)
        reasons.append(reason)
    df["intent_label"] = intents
    df["should_escalate"] = esc
    df["escalation_reason"] = reasons
    df["label_method"] = "rule_based_v1"
    df.to_csv("data/processed/golden_set.csv", index=False)

    print(f"Golden set: {len(df)} examples")
    print(df["intent_label"].value_counts())
    print(f"\nEscalate: {df['should_escalate'].sum()} / {len(df)}")


if __name__ == "__main__":
    main()
