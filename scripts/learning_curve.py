"""Phase: learning curve. Does more real SpotifyCares data (from the same
dataset, via retrieval_corpus.csv -- currently unused for classifier
training) improve held-out DEV performance, or does it plateau?

Protocol:
- Fixed held-out DEV_VAL: the same 20% stratified split of dev_pool used in
  scripts/diagnose_classifier.py (seed=42), held out from ALL training here.
- TRAIN UNIVERSE: dev_pool's 80% train split + all of retrieval_corpus,
  weakly labeled with the same taxonomy rules. retrieval_corpus is already
  conversation-level disjoint from golden_set and dev_pool (verified by
  tests/test_data_pipeline.py) -- expanding into it does not touch golden.
- Train on increasing fractions of the combined universe, evaluate on the
  SAME fixed DEV_VAL each time. GOLDEN is not touched in this script.
"""
import sys
sys.path.insert(0, "src")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, f1_score
from taxonomy import classify_intent

SEED = 42
FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.0]


def build_pipe():
    features = FeatureUnion([
        ("word", TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)),
    ])


def main():
    dev_pool = pd.read_csv("data/processed/dev_pool.csv")
    dev_pool["intent_label"] = dev_pool["customer_msg"].apply(classify_intent)
    dev_train, dev_val = train_test_split(
        dev_pool, test_size=0.2, random_state=SEED, stratify=dev_pool["intent_label"]
    )

    retrieval = pd.read_csv("data/processed/retrieval_corpus.csv")
    retrieval["intent_label"] = retrieval["customer_msg"].apply(classify_intent)

    universe = pd.concat([dev_train[["customer_msg", "intent_label"]],
                           retrieval[["customer_msg", "intent_label"]]], ignore_index=True)
    universe = universe.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"Train universe (dev_train + retrieval_corpus): {len(universe)}")
    print(f"Fixed held-out DEV_VAL (from dev_pool only, never in universe): {len(dev_val)}")
    print("\nUniverse class counts:")
    print(universe.intent_label.value_counts())

    print(f"\n{'frac':>6} {'n_train':>8} {'dev_acc':>8} {'dev_macroF1':>12}")
    for frac in FRACTIONS:
        n = int(len(universe) * frac)
        sub = universe.iloc[:n]
        pipe = build_pipe()
        pipe.fit(sub["customer_msg"], sub["intent_label"])
        preds = pipe.predict(dev_val["customer_msg"])
        acc = accuracy_score(dev_val["intent_label"], preds)
        f1 = f1_score(dev_val["intent_label"], preds, average="macro", zero_division=0)
        print(f"{frac:>6.0%} {n:>8} {acc:>8.4f} {f1:>12.4f}")


if __name__ == "__main__":
    main()
