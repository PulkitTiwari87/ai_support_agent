"""TF-IDF + Logistic Regression intent classifier.

Trained on the dev_pool with weak labels from the rule-based taxonomy
(src/taxonomy.py), then evaluated against the frozen golden_set (a disjoint
split, see planning/05_GOLDEN_SET.md). This is a genuinely separate model
from the rule labeler: training on dev_pool and evaluating on eval_pool
means classifier errors are real, not circular.
"""
import sys
import pickle
sys.path.insert(0, "src")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from taxonomy import classify_intent

MODEL_PATH = "data/processed/intent_model.pkl"


def train():
    df = pd.read_csv("data/processed/dev_pool.csv")
    df["intent_label"] = df["customer_msg"].apply(classify_intent)

    # word n-grams + char n-grams: char n-grams help match word-form variants
    # (crash/crashes/crashing) without needing more labeled examples. Tuned
    # on an 80/20 held-out split of dev_pool (macro-F1 0.595 -> 0.606 there,
    # scripts/experiment_char_ngrams.py) and checked once against the
    # frozen golden set before keeping -- see planning/18_DECISION_LOG.md.
    features = FeatureUnion([
        ("word", TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    pipe = Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)),
    ])
    pipe.fit(df["customer_msg"], df["intent_label"])
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    return pipe


def load():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(pipe, text: str):
    # An "override other with best non-other class above a threshold" rule
    # was tried (scripts/tune_other_threshold.py) to counter 'other' being
    # 65% of dev_pool weak-training labels. Tuned honestly on an 80/20
    # held-out slice of dev_pool (macro-F1 0.595 -> 0.622 there), but
    # regressed the frozen golden set on every axis that matters, including
    # the headline false-auto-handle rate (0.095 -> 0.134, i.e. 12 -> 17
    # missed escalations). Rejected per the regression policy -- see
    # planning/18_DECISION_LOG.md. Kept as plain argmax.
    proba = pipe.predict_proba([text])[0]
    classes = pipe.classes_
    idx = proba.argmax()
    return classes[idx], float(proba[idx])


if __name__ == "__main__":
    pipe = train()
    print("Trained intent classifier on dev_pool (weak labels).")
    print("Classes:", list(pipe.classes_))
