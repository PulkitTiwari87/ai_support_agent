import sys
sys.path.insert(0, "src")
from reply_generation import generate_extractive, _clean_agent_signoff


def test_low_similarity_evidence_refuses_to_answer():
    result = generate_extractive("some issue", [{"support_reply": "reply", "similarity": 0.01, "customer_msg": "x"}])
    assert result["grounded"] is False
    assert result["draft_reply"] is None


def test_good_evidence_produces_reply():
    evidence = [{"support_reply": "Try restarting the app. /AB", "similarity": 0.5, "customer_msg": "x"}]
    result = generate_extractive("app won't start", evidence)
    assert result["grounded"] is True
    assert result["draft_reply"] is not None
    assert "/AB" not in result["draft_reply"]


def test_signoff_stripping():
    assert _clean_agent_signoff("Try this fix. ^CC") == "Try this fix."
    assert _clean_agent_signoff("No signoff here") == "No signoff here"


def test_no_evidence_refuses():
    result = generate_extractive("issue", [])
    assert result["grounded"] is False
