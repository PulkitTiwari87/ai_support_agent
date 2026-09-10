"""Parse SpotifyCares conversations into structured (customer_msg, support_reply) pairs.
Cleans handles/URLs, drops malformed/duplicate records, splits dev/retrieval/eval pools.
"""
import re
import json
import pandas as pd

BRAND = "SpotifyCares"

def clean_text(t: str) -> str:
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def parse_conversation(conv_id: str, conv: str):
    """Return list of (customer_msg, support_reply) adjacent pairs."""
    lines = conv.split("\n")
    turns = []
    for line in lines:
        m = re.match(r"^(Customer|Support):\s?(.*)$", line.strip())
        if m:
            turns.append((m.group(1), m.group(2)))
    pairs = []
    for i in range(len(turns) - 1):
        role, text = turns[i]
        nrole, ntext = turns[i + 1]
        if role == "Customer" and nrole == "Support":
            cust = clean_text(text)
            supp = clean_text(ntext)
            if len(cust) >= 15 and len(supp) >= 10:
                pairs.append({
                    "conversation_id": conv_id,
                    "customer_msg": cust,
                    "support_reply": supp,
                })
    return pairs

def main():
    df = pd.read_parquet("data/raw/customer_support_twitter.parquet")
    sub = df[df["company"] == BRAND].copy()
    print(f"{BRAND}: {len(sub)} raw conversations")

    all_pairs = []
    for _, row in sub.iterrows():
        all_pairs.extend(parse_conversation(row["conversation_id"], row["conversation"]))

    pairs_df = pd.DataFrame(all_pairs)
    print(f"Extracted {len(pairs_df)} (customer_msg, support_reply) pairs")

    before = len(pairs_df)
    pairs_df = pairs_df.drop_duplicates(subset=["customer_msg", "support_reply"])
    print(f"Deduped: {before} -> {len(pairs_df)}")

    # drop boilerplate-only replies (too generic to ground anything on)
    boiler = pairs_df["support_reply"].str.lower().isin({
        "thanks for reaching out.", "we're here to help.", "hi there!",
    })
    pairs_df = pairs_df[~boiler]

    # split at conversation_id level to avoid retrieval/eval leakage from
    # multiple pairs of the same conversation landing in different splits
    conv_ids = pairs_df["conversation_id"].drop_duplicates().sample(frac=1, random_state=42).tolist()
    n_conv = len(conv_ids)
    n_eval_c = min(250, int(n_conv * 0.08))
    n_dev_c = min(400, int(n_conv * 0.1))
    eval_ids = set(conv_ids[:n_eval_c])
    dev_ids = set(conv_ids[n_eval_c:n_eval_c + n_dev_c])

    pairs_df["pair_id"] = [f"sp_{i:06d}" for i in range(len(pairs_df))]
    eval_df = pairs_df[pairs_df["conversation_id"].isin(eval_ids)].groupby("conversation_id").first().reset_index()
    dev_df = pairs_df[pairs_df["conversation_id"].isin(dev_ids)]
    retrieval_df = pairs_df[~pairs_df["conversation_id"].isin(eval_ids | dev_ids)]

    eval_df.to_csv("data/processed/eval_pool.csv", index=False)
    dev_df.to_csv("data/processed/dev_pool.csv", index=False)
    retrieval_df.to_csv("data/processed/retrieval_corpus.csv", index=False)

    print(f"eval_pool={len(eval_df)} dev_pool={len(dev_df)} retrieval_corpus={len(retrieval_df)}")

    stats = {
        "brand": BRAND,
        "raw_conversations": int(len(sub)),
        "extracted_pairs": int(before),
        "deduped_pairs": int(len(pairs_df)) ,
        "eval_pool": int(len(eval_df)),
        "dev_pool": int(len(dev_df)),
        "retrieval_corpus": int(len(retrieval_df)),
    }
    with open("data/processed/data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
