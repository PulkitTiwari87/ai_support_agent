"""Small, targeted follow-up: does LinearSVC's C / class_weight matter as
much as it did for Logistic Regression (Group B found class_weight=None +
higher C helped substantially once data was expanded)? Same fixed DEV_VAL
protocol as scripts/experiment_grid.py. Winning features from Group A:
word(1,2) + char(3,7).
"""
import sys
sys.path.insert(0, "src")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
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


def make_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 7), min_df=2, sublinear_tf=True)),
    ])


def main():
    universe, dev_val = load_universe()
    for cw in [None, "balanced"]:
        for C in [0.5, 1.0, 2.0]:
            feats = make_features()
            clf = LinearSVC(class_weight=cw, C=C, max_iter=5000)
            pipe = Pipeline([("features", feats), ("clf", clf)])
            pipe.fit(universe["customer_msg"], universe["intent_label"])
            preds = pipe.predict(dev_val["customer_msg"])
            acc = accuracy_score(dev_val["intent_label"], preds)
            f1 = f1_score(dev_val["intent_label"], preds, average="macro", zero_division=0)
            print(f"LinearSVC C={C} class_weight={str(cw):9s} acc={acc:.4f} macro-F1={f1:.4f}")


if __name__ == "__main__":
    main()
