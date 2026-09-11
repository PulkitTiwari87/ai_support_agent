"""Evaluation harness: runs the system and both baselines on the frozen
golden set and reports intent + escalation metrics for all three.

Golden labels come from src/taxonomy.py rules (spot-checked, see
planning/05_GOLDEN_SET.md) -- NOT from the system under test, so this is not
circular for the learned intent classifier or the retrieval-grounded
generator. It IS a soft ceiling for the rule-based escalation logic, since
golden escalation labels use the same rule family; this is disclosed as a
limitation in the README/report, not hidden.
"""
import json
import sys
sys.path.insert(0, "src")
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from pipeline import Pipeline
from baselines import baseline_trivial, baseline_simple


def run_system(df, pipeline):
    intents, escalates, reasons, replies, grounded = [], [], [], [], []
    for msg in df["customer_msg"]:
        r = pipeline.run(msg)
        intents.append(r["intent"])
        escalates.append(r["escalate"])
        reasons.append(r["escalation_reason"])
        replies.append(r["draft_reply"])
        grounded.append(r["grounded"])
    return pd.DataFrame({
        "pred_intent": intents, "pred_escalate": escalates,
        "escalation_reason": reasons, "draft_reply": replies, "grounded": grounded,
    })


def run_baseline(df, fn):
    intents, escalates = [], []
    for msg in df["customer_msg"]:
        r = fn(msg)
        intents.append(r["intent"])
        escalates.append(r["escalate"])
    return pd.DataFrame({"pred_intent": intents, "pred_escalate": escalates})


def compute_metrics(gold, preds, name):
    y_true_intent = gold["intent_label"]
    y_pred_intent = preds["pred_intent"]
    y_true_esc = gold["should_escalate"]
    y_pred_esc = preds["pred_escalate"]

    labels = sorted(set(y_true_intent) | set(y_pred_intent))
    cm = confusion_matrix(y_true_intent, y_pred_intent, labels=labels)

    # false auto-handle: gold says escalate, system says don't -> high-risk failure
    false_auto_handle = int(((y_true_esc == True) & (y_pred_esc == False)).sum())
    false_escalation = int(((y_true_esc == False) & (y_pred_esc == True)).sum())

    per_class_f1 = f1_score(y_true_intent, y_pred_intent, average=None, labels=labels, zero_division=0)

    return {
        "name": name,
        "intent_accuracy": round(accuracy_score(y_true_intent, y_pred_intent), 4),
        "intent_macro_f1": round(f1_score(y_true_intent, y_pred_intent, average="macro", zero_division=0), 4),
        "intent_weighted_f1": round(f1_score(y_true_intent, y_pred_intent, average="weighted", zero_division=0), 4),
        "intent_per_class_f1": {label: round(float(f1), 4) for label, f1 in zip(labels, per_class_f1)},
        "intent_confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "escalation_precision": round(precision_score(y_true_esc, y_pred_esc, zero_division=0), 4),
        "escalation_recall": round(recall_score(y_true_esc, y_pred_esc, zero_division=0), 4),
        "escalation_f1": round(f1_score(y_true_esc, y_pred_esc, zero_division=0), 4),
        "false_auto_handle_count": false_auto_handle,
        "false_auto_handle_rate": round(false_auto_handle / max((y_true_esc == True).sum(), 1), 4),
        "false_escalation_count": false_escalation,
        "false_escalation_rate": round(false_escalation / max((y_true_esc == False).sum(), 1), 4),
        "n": len(gold),
    }


def main():
    gold = pd.read_csv("data/processed/golden_set.csv")
    gold["should_escalate"] = gold["should_escalate"].astype(bool)

    pipeline = Pipeline()
    sys_preds = run_system(gold, pipeline)
    trivial_preds = run_baseline(gold, baseline_trivial)
    simple_preds = run_baseline(gold, baseline_simple)

    results = [
        compute_metrics(gold, trivial_preds, "baseline_trivial"),
        compute_metrics(gold, simple_preds, "baseline_simple"),
        compute_metrics(gold, sys_preds, "system_full"),
    ]

    for r in results:
        print(f"\n=== {r['name']} (n={r['n']}) ===")
        print(f"  intent_accuracy={r['intent_accuracy']}  intent_macro_f1={r['intent_macro_f1']}")
        print(f"  escalation P={r['escalation_precision']} R={r['escalation_recall']} F1={r['escalation_f1']}")
        print(f"  false_auto_handle_rate={r['false_auto_handle_rate']} ({r['false_auto_handle_count']} cases)")
        print(f"  false_escalation_rate={r['false_escalation_rate']} ({r['false_escalation_count']} cases)")

    grounded_rate = sys_preds["grounded"].mean()
    print(f"\nSystem reply groundedness: {grounded_rate:.1%} of replies had sufficient retrieval evidence")

    with open("data/processed/eval_results.json", "w") as f:
        json.dump({"results": results, "grounded_rate": round(float(grounded_rate), 4)}, f, indent=2)

    sys_preds.to_csv("data/processed/system_predictions.csv", index=False)
    print("\nSaved: data/processed/eval_results.json, data/processed/system_predictions.csv")


if __name__ == "__main__":
    main()
