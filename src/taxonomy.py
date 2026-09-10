"""Intent taxonomy and rule-based labeler for the SpotifyCares golden set.

The taxonomy was derived by reading ~150 sampled customer messages from
data/processed/dev_pool.csv (see planning/04_INTENT_TAXONOMY.md for the
sampling notes). Labeling is done with keyword/regex rules, not a trained
model or an LLM call -- this keeps golden-set construction reproducible and
free of circular dependency on the system being evaluated. Rules were tuned
against the dev pool only (never the eval pool) and then spot-checked by
hand against a random sample of the eval pool (see
planning/05_GOLDEN_SET.md for the spot-check results).
"""
import re

INTENTS = [
    "account_access",
    "billing_subscription",
    "playback_technical",
    "content_availability",
    "feature_request_feedback",
    "other",
]

_RULES = [
    ("account_access", re.compile(
        r"\b(log ?in|log ?on|logg?ing (in|out)|logs? (me )?out|password|sign ?up|sign ?in|hacked|"
        r"account.*(stolen|hacked|locked|access)|accessed my account|"
        r"can'?t access|verify my account|reset my (password|account)|email (was )?changed|"
        r"two.factor|2fa|locked out|link my \w+ (to|with) my account|connect(ed)? .*to my account)\b", re.I)),
    ("billing_subscription", re.compile(
        r"\b(premium|subscription|billing|payment|charged|refund|cancell?(ed|ing)?|invoice|price|pricing|"
        r"student (offer|discount|plan)|family (plan|member|package)|free trial|renew|card (declined|not accepting)|"
        r"paid|promo(tional)?( offer| code)?|discount code|\$\d|\d+\.\d\d\b|pesos|DRM|package)\b", re.I)),
    ("playback_technical", re.compile(
        r"\b(crash\w*|bug|not working|won'?t (play|open|load|connect|work)|error|glitch|freeze|"
        r"stop(s|ped|ping)? (working|playing|early)|won'?t (play|connect)|sync(ing)?|offline|"
        r"download(s|ed)? (disappear|delet)|no media control|"
        r"playlists? (disappear|gone|missing)|skip(ping)? (songs?|tracks?)|app (keeps|won'?t)|"
        r"sonos|chromecast|android auto|carplay|update (broke|to)|version \d|"
        r"cache issue|web player|album art|shuffle|ads? (stall|during|stop)|"
        r"lost spotify|reinstall|local librar(y|ies)|landscape mode|black bars)\b", re.I)),
    ("content_availability", re.compile(
        r"\b(can'?t find|not (on|available (on|in)) spotify|missing (from|on) spotify|"
        r"add (this|that) (song|album|artist|track)|"
        r"duplicate (page|album|artist)|catalog|licens(e|ing)|"
        r"available in \w+|spotify in \w+|spotify (blocked|banned)|"
        r"why (isn'?t|is|are).*(on spotify|available)|not letting me listen)\b", re.I)),
    ("feature_request_feedback", re.compile(
        r"\b(wish (you|spotify|had)|would be (nice|cool|great) if|feature request|suggestion|"
        r"please add|you should (add|make)|update the .*playlist|discover weekly|"
        r"i love|thanks for|great (app|job)|you guys rock|it would be cool|"
        r"stop getting|easier way to|is there a way to|any known issues|song limit)\b", re.I)),
]


def classify_intent(text: str) -> str:
    for intent, pattern in _RULES:
        if pattern.search(text):
            return intent
    return "other"


_ESCALATE_RULES = [
    ("account_security", re.compile(
        r"\b(hack\w*|stole\w*|stolen|account.*(hack\w*|stolen|compromised)|"
        r"email (was )?changed.*(not me|didn'?t|without)|unauthorized|fraud|"
        r"(accessed|logged in to|got into) my account|"
        r"changed (the |my )?(account )?email.*(recognize|not mine|wasn'?t me))\b", re.I)),
    ("billing_dispute", re.compile(
        r"\b(refund|charged (twice|again|wrong)|dispute|unauthorized (charge|payment)|"
        r"cancel my (subscription|account)|billing error)\b", re.I)),
    ("explicit_human_request", re.compile(
        r"\b(speak to (a|someone)|talk to (a|someone)|human|real person|manager|"
        r"escalate|representative)\b", re.I)),
    ("high_distress", re.compile(
        r"(!{3,}|\?{3,})|\b(furious|disgusted|lawsuit|lawyer|legal action)\b", re.I)),
]


def classify_escalation(text: str, intent: str, retrieval_confidence: float | None = None) -> tuple[bool, str]:
    """Returns (should_escalate, reason)."""
    for reason, pattern in _ESCALATE_RULES:
        if pattern.search(text):
            return True, reason
    if retrieval_confidence is not None and retrieval_confidence < 0.15:
        return True, "low_retrieval_confidence"
    if intent == "other":
        return True, "unclassified_intent"
    return False, "none"
