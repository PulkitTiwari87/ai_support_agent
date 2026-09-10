"""Tunes the non-'other' override threshold using ONLY dev_pool (weak
labels), never the frozen golden set. Splits dev_pool 80/20 to get an
honest validation signal for the threshold choice, then reports the value
to hardcode into src/intent_classifier.py.

Rule: after predict_proba, if the max probability among non-'other' classes
clears `threshold`, predict that class instead of raw argmax (which
over-predicts 'other' due to its 65% share of weak training labels).
"""
import sys
sys.path.insert(0, "src")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from taxonomy import classify_intent


def predict_with_override(pipe, texts, threshold):
    proba = pipe.predict_proba(texts)
    classes = pipe.classes_
    other_idx = list(classes).index("other")
    preds = []
    for row in proba:
        non_other = [(c, p) for c, p in zip(classes, row) if c != "other"]
        best_c, best_p = max(non_other, key=lambda x: x[1])
        if best_p >= threshold:
            preds.append(best_c)
        else:
            preds.append(classes[row.argmax()])
    return preds


def main():
    df = pd.read_csv("data/processed/dev_pool.csv")
    df["intent_label"] = df["customer_msg"].apply(classify_intent)
    train, val = train_test_split(df, test_size=0.2, random_state=42, stratify=df["intent_label"])

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)),
    ])
    pipe.fit(train["customer_msg"], train["intent_label"])

    baseline_preds = pipe.predict(val["customer_msg"])
    baseline_f1 = f1_score(val["intent_label"], baseline_preds, average="macro", zero_division=0)
    print(f"baseline (raw argmax) macro-F1 on dev-val: {baseline_f1:.4f}")

    best_t, best_f1 = None, baseline_f1
    for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        preds = predict_with_override(pipe, val["customer_msg"], t)
        f1 = f1_score(val["intent_label"], preds, average="macro", zero_division=0)
        print(f"threshold={t}: macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_t, best_f1 = t, f1

    print(f"\nBest: threshold={best_t} (macro-F1={best_f1:.4f}) vs baseline {baseline_f1:.4f}")


if __name__ == "__main__":
    main()
