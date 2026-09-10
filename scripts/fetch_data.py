"""Downloads the raw dataset (public HuggingFace mirror, no auth needed).
See planning/18_DECISION_LOG.md item 1 for why this mirror, not Kaggle."""
import os
import urllib.request

URL = "https://huggingface.co/api/datasets/TNE-AI/customer-support-on-twitter-conversation/parquet/default/train/0.parquet"
OUT = "data/raw/customer_support_twitter.parquet"

def main():
    os.makedirs("data/raw", exist_ok=True)
    if os.path.exists(OUT):
        print(f"Already present: {OUT}")
        return
    print(f"Downloading {URL} -> {OUT} (~210MB)...")
    urllib.request.urlretrieve(URL, OUT)
    print("Done.")

if __name__ == "__main__":
    main()
