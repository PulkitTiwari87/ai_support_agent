"""Experiment: add char n-gram features to the intent classifier to see if
they help with minority-class recall (feature_request_feedback n=11,
content_availability n=19 in dev_pool weak labels -- likely too few examples
for word-level TF-IDF alone). Tuned on an 80/20 held-out split of dev_pool
only; checked once against the golden set at the end, kept only if it
doesn't regress the headline metric (false-auto-handle rate).
"""
import sys
sys.path.insert(0, "src")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import f1_score, classification_report
from taxonomy import classify_intent


def build_pipe(with_char_ngrams: bool):
    if not with_char_ngrams:
        return Pipeline([
            ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)),
        ])
    features = FeatureUnion([
        ("word", TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)),
    ])


def main():
    df = pd.read_csv("data/processed/dev_pool.csv")
    df["intent_label"] = df["customer_msg"].apply(classify_intent)
    train, val = train_test_split(df, test_size=0.2, random_state=42, stratify=df["intent_label"])

    for name, use_char in [("word-only (current)", False), ("word+char n-grams", True)]:
        pipe = build_pipe(use_char)
        pipe.fit(train["customer_msg"], train["intent_label"])
        preds = pipe.predict(val["customer_msg"])
        f1 = f1_score(val["intent_label"], preds, average="macro", zero_division=0)
        print(f"\n=== {name} === macro-F1 on dev-val: {f1:.4f}")
        print(classification_report(val["intent_label"], preds, zero_division=0))


if __name__ == "__main__":
    main()
