import sys
sys.path.insert(0, "src")
from taxonomy import classify_intent, classify_escalation, INTENTS


def test_classify_intent_returns_valid_label():
    assert classify_intent("I can't log into my account") in INTENTS


def test_account_access_keywords():
    assert classify_intent("I forgot my password and can't sign in") == "account_access"


def test_billing_keywords():
    assert classify_intent("I was charged twice for my premium subscription") == "billing_subscription"


def test_playback_keywords():
    assert classify_intent("The app crashes every time I try to play a song") == "playback_technical"


def test_unrecognized_falls_back_to_other():
    assert classify_intent("xyz") == "other"


def test_account_takeover_escalates():
    esc, reason = classify_escalation(
        "Someone accessed my account and changed the email to one I don't recognize",
        "account_access",
    )
    assert esc is True
    assert reason == "account_security"


def test_billing_dispute_escalates():
    esc, reason = classify_escalation("I was charged twice, please refund me", "billing_subscription")
    assert esc is True
    assert reason == "billing_dispute"


def test_low_retrieval_confidence_escalates():
    esc, reason = classify_escalation("hello", "other", retrieval_confidence=0.05)
    assert esc is True


def test_normal_request_does_not_escalate():
    esc, reason = classify_escalation(
        "How do I add a song to a playlist?", "content_availability", retrieval_confidence=0.5
    )
    assert esc is False
