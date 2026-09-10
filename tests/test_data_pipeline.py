"""Data integrity checks -- run after scripts/process_data.py and
scripts/build_golden_set.py have produced data/processed artifacts."""
import os
import pandas as pd
import pytest

PROCESSED = "data/processed"


def _need(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not built yet -- run scripts/process_data.py and scripts/build_golden_set.py")
    return path


def test_no_conversation_leakage_between_eval_and_retrieval():
    eval_df = pd.read_csv(_need(f"{PROCESSED}/eval_pool.csv"))
    retrieval_df = pd.read_csv(_need(f"{PROCESSED}/retrieval_corpus.csv"))
    overlap = set(eval_df["conversation_id"]) & set(retrieval_df["conversation_id"])
    assert len(overlap) == 0


def test_no_conversation_leakage_between_eval_and_dev():
    eval_df = pd.read_csv(_need(f"{PROCESSED}/eval_pool.csv"))
    dev_df = pd.read_csv(_need(f"{PROCESSED}/dev_pool.csv"))
    overlap = set(eval_df["conversation_id"]) & set(dev_df["conversation_id"])
    assert len(overlap) == 0


def test_golden_set_size_in_target_range():
    golden = pd.read_csv(_need(f"{PROCESSED}/golden_set.csv"))
    assert 150 <= len(golden) <= 250


def test_golden_set_no_duplicate_conversations():
    golden = pd.read_csv(_need(f"{PROCESSED}/golden_set.csv"))
    assert golden["conversation_id"].is_unique


def test_golden_set_has_required_columns():
    golden = pd.read_csv(_need(f"{PROCESSED}/golden_set.csv"))
    for col in ["customer_msg", "support_reply", "intent_label", "should_escalate"]:
        assert col in golden.columns
