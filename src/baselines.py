"""Two baselines evaluated on the same golden set as the full system.

Baseline 1 (trivial): majority-class intent, canned reply, never escalates.
Baseline 2 (simple): single-keyword-per-intent lookup (deliberately blunter
than the full taxonomy regex used to build the golden set -- reusing that
same regex here would make this baseline circularly "perfect" on intent),
one canned reply per intent, escalates only on a narrow set of explicit
distress/security keywords (no retrieval or model confidence signal).
"""
import re
import sys
sys.path.insert(0, "src")
from taxonomy import _ESCALATE_RULES

_SIMPLE_KEYWORDS = [
    ("account_access", re.compile(r"\b(login|log in|password|sign up|account)\b", re.I)),
    ("billing_subscription", re.compile(r"\b(premium|subscription|charge|billing|refund|cancel)\b", re.I)),
    ("playback_technical", re.compile(r"\b(crash|bug|error|not working|won'?t play|stopped)\b", re.I)),
    ("content_availability", re.compile(r"\b(not available|can'?t find|missing|catalog)\b", re.I)),
    ("feature_request_feedback", re.compile(r"\b(wish|suggestion|please add|feature)\b", re.I)),
]


def _simple_classify(text: str) -> str:
    for intent, pattern in _SIMPLE_KEYWORDS:
        if pattern.search(text):
            return intent
    return "other"

MAJORITY_INTENT = "other"  # most frequent class in dev_pool weak labels

CANNED_REPLIES = {
    "account_access": "Sorry to hear that! Please try resetting your password from the app, and let us know if the issue continues.",
    "billing_subscription": "Thanks for flagging this -- please DM us your account email so we can look into your billing.",
    "playback_technical": "Sorry for the trouble! Can you tell us your device, OS, and Spotify app version so we can look into it?",
    "content_availability": "Thanks for the suggestion -- content availability depends on licensing agreements in your region.",
    "feature_request_feedback": "Thanks for the feedback, we'll pass it along to the team!",
    "other": "Thanks for reaching out -- can you tell us a bit more about the issue so we can help?",
}


def baseline_trivial(customer_msg: str) -> dict:
    return {
        "intent": MAJORITY_INTENT,
        "draft_reply": CANNED_REPLIES[MAJORITY_INTENT],
        "escalate": False,
        "escalation_reason": "none",
    }


def baseline_simple(customer_msg: str) -> dict:
    intent = _simple_classify(customer_msg)
    escalate, reason = False, "none"
    for r_reason, pattern in _ESCALATE_RULES[:3]:  # keyword rules only, no confidence/retrieval signal
        if pattern.search(customer_msg):
            escalate, reason = True, r_reason
            break
    return {
        "intent": intent,
        "draft_reply": CANNED_REPLIES[intent],
        "escalate": escalate,
        "escalation_reason": reason,
    }
