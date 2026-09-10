"""Brand selection: score candidate brands on volume, resolution rate, and issue diversity.
Reproducible: run against data/raw/customer_support_twitter.parquet.
"""
import re
import pandas as pd

CANDIDATES = [
    "AmazonHelp", "AppleSupport", "Uber_Support", "SpotifyCares",
    "comcastcares", "TMobileHelp", "XboxSupport", "AskPlayStation",
    "AskPayPal", "VerizonSupport",
]

def has_support_reply(conv: str) -> bool:
    return "\nSupport:" in conv or conv.startswith("Support:")

def n_turns(conv: str) -> int:
    return len(re.findall(r"(?:^|\n)(?:Customer|Support):", conv))

def main():
    df = pd.read_parquet("data/raw/customer_support_twitter.parquet")
    rows = []
    for brand in CANDIDATES:
        sub = df[df["company"] == brand]
        n = len(sub)
        if n == 0:
            continue
        resolved = sub["conversation"].apply(has_support_reply)
        turns = sub["conversation"].apply(n_turns)
        # crude vocabulary diversity proxy: unique word count / total words in a 2000-conv sample
        sample = sub["conversation"].sample(min(2000, n), random_state=42)
        words = " ".join(sample).lower().split()
        vocab_ratio = len(set(words)) / max(len(words), 1)
        rows.append({
            "brand": brand,
            "n_conversations": n,
            "pct_with_support_reply": round(resolved.mean() * 100, 1),
            "avg_turns": round(turns.mean(), 2),
            "vocab_diversity": round(vocab_ratio, 4),
        })
    out = pd.DataFrame(rows).sort_values("n_conversations", ascending=False)
    print(out.to_string(index=False))
    out.to_csv("data/processed/brand_candidates.csv", index=False)

if __name__ == "__main__":
    main()
