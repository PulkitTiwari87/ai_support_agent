"""Controlled experiment grid on the EXPANDED training universe
(dev_train + retrieval_corpus), evaluated on the fixed held-out DEV_VAL
(from dev_pool only, seed=42, same split as diagnose_classifier.py /
learning_curve.py). GOLDEN is not touched anywhere in this script.

Sequential, not full cross-product, to keep this a small controlled grid:
  Group A: TF-IDF feature config (word n-grams, char n-grams, min_df)
  Group B: Logistic Regression C and class_weight, using winning features
  Group C: alternative simple classifiers, using winning features
"""
import sys
sys.path.insert(0, "src")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from taxonomy import classify_intent

SEED = 42


def load_universe():
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
    return universe, dev_val


def make_features(word_ngram, char_ngram, min_df):
    parts = [("word", TfidfVectorizer(min_df=min_df, ngram_range=word_ngram, sublinear_tf=True))]
    if char_ngram:
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram, min_df=min_df, sublinear_tf=True)))
    if len(parts) == 1:
        return parts[0][1]
    return FeatureUnion(parts)


def evaluate(features, clf, universe, dev_val, label):
    pipe = Pipeline([("features", features), ("clf", clf)])
    pipe.fit(universe["customer_msg"], universe["intent_label"])
    preds = pipe.predict(dev_val["customer_msg"])
    acc = accuracy_score(dev_val["intent_label"], preds)
    f1 = f1_score(dev_val["intent_label"], preds, average="macro", zero_division=0)
    print(f"{label:55s} acc={acc:.4f}  macro-F1={f1:.4f}")
    return acc, f1


def main():
    universe, dev_val = load_universe()
    print(f"universe n={len(universe)}  dev_val n={len(dev_val)}\n")

    print("=== Group A: TF-IDF feature config (LR C=2.0, balanced) ===")
    configs_a = [
        ("word(1,1) only",        (1, 1), None,    2),
        ("word(1,2) only",        (1, 2), None,    2),
        ("word(1,3) only",        (1, 3), None,    2),
        ("word(1,2)+char(3,5) [current]", (1, 2), (3, 5), 2),
        ("word(1,2)+char(3,6)",   (1, 2), (3, 6), 2),
        ("word(1,2)+char(4,6)",   (1, 2), (3, 6), 2),  # placeholder overwritten below
        ("word(1,2)+char(3,7)",   (1, 2), (3, 7), 2),
        ("word(1,2)+char(3,5) min_df=1", (1, 2), (3, 5), 1),
    ]
    # fix the (4,6) entry properly
    configs_a[5] = ("word(1,2)+char(4,6)", (1, 2), (4, 6), 2)

    results_a = []
    for label, word_ng, char_ng, min_df in configs_a:
        feats = make_features(word_ng, char_ng, min_df)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)
        acc, f1 = evaluate(feats, clf, universe, dev_val, label)
        results_a.append((label, word_ng, char_ng, min_df, acc, f1))

    best_a = max(results_a, key=lambda r: r[5])
    print(f"\nBest Group A by macro-F1: {best_a[0]} (macro-F1={best_a[5]:.4f})")
    best_word_ng, best_char_ng, best_min_df = best_a[1], best_a[2], best_a[3]

    print("\n=== Group B: Logistic Regression C / class_weight (winning features) ===")
    results_b = []
    for cw in [None, "balanced"]:
        for C in [0.1, 0.5, 1, 2, 5, 10]:
            feats = make_features(best_word_ng, best_char_ng, best_min_df)
            clf = LogisticRegression(max_iter=1000, class_weight=cw, C=C)
            label = f"LR C={C} class_weight={cw}"
            acc, f1 = evaluate(feats, clf, universe, dev_val, label)
            results_b.append((label, C, cw, acc, f1))

    best_b = max(results_b, key=lambda r: r[4])
    print(f"\nBest Group B by macro-F1: {best_b[0]} (macro-F1={best_b[4]:.4f})")

    print("\n=== Group C: alternative simple classifiers (winning features) ===")
    results_c = []

    feats = make_features(best_word_ng, best_char_ng, best_min_df)
    acc, f1 = evaluate(feats, LogisticRegression(max_iter=1000, class_weight=best_b[2], C=best_b[1]), universe, dev_val, "LogisticRegression (best config)")
    results_c.append(("LogisticRegression", acc, f1))

    feats2 = make_features(best_word_ng, best_char_ng, best_min_df)
    svc = LinearSVC(class_weight="balanced", C=1.0, max_iter=5000)
    acc, f1 = evaluate(feats2, svc, universe, dev_val, "LinearSVC (balanced, C=1.0)")
    results_c.append(("LinearSVC", acc, f1))

    feats3 = make_features(best_word_ng, best_char_ng, best_min_df)
    acc, f1 = evaluate(feats3, MultinomialNB(), universe, dev_val, "MultinomialNB")
    results_c.append(("MultinomialNB", acc, f1))

    best_c = max(results_c, key=lambda r: r[2])
    print(f"\nBest Group C by macro-F1: {best_c[0]} (macro-F1={best_c[2]:.4f})")


if __name__ == "__main__":
    main()
