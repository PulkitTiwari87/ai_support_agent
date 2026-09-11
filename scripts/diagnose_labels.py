"""Phase 1 continued: label-quality and 'other'-class diagnosis on
dev_pool.csv only (never golden). Read-only inspection, no label changes.
"""
import sys
sys.path.insert(0, "src")
import re
import pandas as pd
from taxonomy import _RULES, classify_intent

def main():
    dev = pd.read_csv("data/processed/dev_pool.csv")
    dev["intent_label"] = dev["customer_msg"].apply(classify_intent)

    # 1. First-match-wins ambiguity: how many examples match >1 rule?
    multi_match = 0
    match_order_effect = []
    for msg in dev["customer_msg"]:
        hits = [name for name, pat in _RULES if pat.search(msg)]
        if len(hits) > 1:
            multi_match += 1
            match_order_effect.append((msg[:80], hits))
    print(f"Examples matching >1 taxonomy rule (first-match-wins bias possible): {multi_match}/{len(dev)} ({multi_match/len(dev):.1%})")
    for msg, hits in match_order_effect[:8]:
        print(f"  {hits} <- {msg}")

    # 2. 'other' bucket: length distribution, short-fragment rate
    other = dev[dev.intent_label == "other"]
    print(f"\n'other' bucket: {len(other)}/{len(dev)} ({len(other)/len(dev):.1%})")
    word_counts = other["customer_msg"].str.split().str.len()
    print(f"  word count distribution: min={word_counts.min()} p25={word_counts.quantile(.25):.0f} "
          f"median={word_counts.median():.0f} p75={word_counts.quantile(.75):.0f} max={word_counts.max()}")
    short = (word_counts <= 5).sum()
    print(f"  <=5 words (likely fragments): {short}/{len(other)} ({short/len(other):.1%})")

    # 3. class counts (repeat, for record)
    print("\nClass counts (dev_pool weak labels):")
    print(dev.intent_label.value_counts())

    # 4. duplicate/near-duplicate customer_msg within dev_pool
    dup = dev.customer_msg.str.lower().str.strip().duplicated().sum()
    print(f"\nExact duplicate customer_msg (case/whitespace-normalized) within dev_pool: {dup}")

if __name__ == "__main__":
    main()
