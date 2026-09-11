"""TF-IDF (word + char n-grams) + LinearSVC intent classifier.

Trained on an EXPANDED pool: dev_pool.csv (529 examples) + retrieval_corpus.csv
(36,004 examples, previously used only for retrieval grounding), both weakly
labeled with the rule-based taxonomy (src/taxonomy.py). Both pools are
conversation-level disjoint from the frozen golden_set (verified by
tests/test_data_pipeline.py and a text-level exact-duplicate check --
zero overlap either way), so expanding into retrieval_corpus does not touch
golden-set isolation. Evaluated against golden_set separately (a disjoint
split, see planning/05_GOLDEN_SET.md) -- classifier errors are real, not
circular.

Why this config (see planning/18_DECISION_LOG.md #22-24 for the full
diagnosis): the previous model (trained on dev_pool's 529 examples alone)
showed textbook overfitting -- 99.8% train accuracy vs 60.6% held-out-dev
macro-F1 vs 59.1% golden macro-F1, with per-class F1 dropping in lockstep
with per-class training-example count (9 examples -> 0.00 dev F1 for
feature_request_feedback). A learning curve using retrieval_corpus as
additional real, same-brand, leakage-safe training data showed continued
DEV macro-F1 improvement up to ~50% of the expanded pool
(scripts/learning_curve.py), then a controlled feature/hyperparameter/
classifier grid (scripts/experiment_grid.py, scripts/experiment_svc_tuning.py)
on the expanded pool selected word(1,2)+char_wb(3,7) TF-IDF features with
LinearSVC over Logistic Regression and MultinomialNB.
"""
import sys
import pickle
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from taxonomy import classify_intent

MODEL_PATH = "data/processed/intent_model.pkl"


def _load_expanded_training_pool():
    dev_pool = pd.read_csv("data/processed/dev_pool.csv")
    retrieval = pd.read_csv("data/processed/retrieval_corpus.csv")
    df = pd.concat([dev_pool[["customer_msg"]], retrieval[["customer_msg"]]], ignore_index=True)
    df["intent_label"] = df["customer_msg"].apply(classify_intent)
    return df


def build_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 7), min_df=2, sublinear_tf=True)),
    ])


def train():
    df = _load_expanded_training_pool()

    pipe = Pipeline([
        ("features", build_features()),
        ("clf", LinearSVC(class_weight="balanced", C=1.0, max_iter=5000)),
    ])
    pipe.fit(df["customer_msg"], df["intent_label"])
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    return pipe


def load():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(pipe, text: str):
    """Returns (label, confidence). LinearSVC has no predict_proba; confidence
    is a softmax over the decision-function margins -- a display/introspection
    value only. Nothing in escalation logic or evaluation consumes this field
    (classify_escalation uses retrieval_confidence, not intent-model
    confidence), so this is a safe, cheap substitute for a true calibrated
    probability."""
    scores = pipe.decision_function([text])[0]
    exp_scores = np.exp(scores - scores.max())
    proba = exp_scores / exp_scores.sum()
    classes = pipe.classes_
    idx = proba.argmax()
    return classes[idx], float(proba[idx])


if __name__ == "__main__":
    pipe = train()
    print("Trained intent classifier on expanded pool (dev_pool + retrieval_corpus, weak labels).")
    print("Classes:", list(pipe.classes_))
