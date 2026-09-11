"""Phase 1 diagnosis: current classifier on TRAIN / held-out DEV / GOLDEN.

DEV here means a fixed 20% stratified split carved out of dev_pool.csv and
held out from all training in this diagnosis (never seen during fit).
GOLDEN is data/processed/golden_set.csv, evaluated once, read-only.

This script only measures the CURRENT shipped model's regime -- it does not
change anything.
"""
import sys
sys.path.insert(0, "src")
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from taxonomy import classify_intent

SEED = 42


def build_current_pipe():
    features = FeatureUnion([
        ("word", TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)),
    ])


def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n=== {name} === n={len(y_true)}  acc={acc:.4f}  macro-F1={macro_f1:.4f}  weighted-F1={weighted_f1:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("labels:", labels)
    print(cm)
    return {"n": len(y_true), "accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


def main():
    dev_pool = pd.read_csv("data/processed/dev_pool.csv")
    dev_pool["intent_label"] = dev_pool["customer_msg"].apply(classify_intent)

    train_df, devval_df = train_test_split(
        dev_pool, test_size=0.2, random_state=SEED, stratify=dev_pool["intent_label"]
    )
    print(f"dev_pool total={len(dev_pool)}  train={len(train_df)}  held-out dev-val={len(devval_df)}")
    print("\nTrain class counts:")
    print(train_df["intent_label"].value_counts())
    print("\nHeld-out dev-val class counts:")
    print(devval_df["intent_label"].value_counts())

    pipe = build_current_pipe()
    pipe.fit(train_df["customer_msg"], train_df["intent_label"])

    results = {}
    results["train"] = report("TRAIN (in-sample)", train_df["intent_label"], pipe.predict(train_df["customer_msg"]))
    results["dev"] = report("HELD-OUT DEV (from dev_pool, never trained on)", devval_df["intent_label"], pipe.predict(devval_df["customer_msg"]))

    golden = pd.read_csv("data/processed/golden_set.csv")
    results["golden"] = report("GOLDEN (frozen, evaluated once here for diagnosis)", golden["intent_label"], pipe.predict(golden["customer_msg"]))

    with open("data/processed/diagnosis_current_model.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
