"""TF-IDF retrieval over historical (customer_msg, support_reply) pairs.

Retrieves prior resolved conversations similar to the incoming message.
This is the grounding mechanism for reply generation: pairs retrieved carry
both the customer's problem AND Spotify's actual historical response, so
generation can be grounded in real resolution patterns rather than
paraphrasing a semantically-similar complaint with no known resolution.
"""
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = "data/processed/retrieval_index.pkl"


def build_index():
    corpus = pd.read_csv("data/processed/retrieval_corpus.csv")
    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(corpus["customer_msg"])
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix, "corpus": corpus}, f)
    return vectorizer, matrix, corpus


def load_index():
    with open(INDEX_PATH, "rb") as f:
        d = pickle.load(f)
    return d["vectorizer"], d["matrix"], d["corpus"]


def retrieve(query: str, vectorizer, matrix, corpus, k: int = 3):
    qvec = vectorizer.transform([query])
    sims = cosine_similarity(qvec, matrix)[0]
    top_idx = sims.argsort()[::-1][:k]
    results = []
    for i in top_idx:
        results.append({
            "customer_msg": corpus.iloc[i]["customer_msg"],
            "support_reply": corpus.iloc[i]["support_reply"],
            "similarity": float(sims[i]),
        })
    return results


if __name__ == "__main__":
    v, m, c = build_index()
    print(f"Retrieval index built: {m.shape[0]} historical pairs, {m.shape[1]} features")
