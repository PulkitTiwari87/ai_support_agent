"""Reply-quality judge.

LLM judge (src/judge.py:llm_judge) requires ANTHROPIC_API_KEY and calls
Claude to rate each (customer_msg, draft_reply, evidence) triple on
correctness/relevance/grounding/tone/hallucination per
planning/11_LLM_JUDGE.md. It was NOT run in this environment -- no API key
was configured -- and its absence is reported as a limitation, not papered
over with the heuristic scores below.

heuristic_judge is a cheap proxy actually used to produce the numbers in
this repo's evaluation artifacts: no semantic understanding, just grounding
and length checks. It is explicitly NOT presented as equivalent to an LLM
or human judge.
"""
import os
import re

RUBRIC_DIMENSIONS = ["correctness", "relevance", "grounding", "completeness", "tone", "hallucination"]


_CLAIM_PATTERN = re.compile(r"\$\d[\d,.]*|\b\d+%|\b\d+\s?(days?|hours?|minutes?|weeks?|months?)\b", re.I)


def _unsupported_claims(draft_reply: str, evidence_text: str) -> list[str]:
    """Numbers/amounts/durations in the reply not found anywhere in the
    evidence text -- a specific, checkable proxy for invented facts. NOTE:
    for the extractive generator this is close to vacuous (the reply IS the
    evidence, so nothing is ever "unsupported" by construction). It becomes
    meaningful once paraphrasing generation (the LLM path) is actually run,
    since paraphrasing is where a model can introduce a number/date/promise
    that wasn't in the source material. Kept here, run now, and disclosed as
    weak evidence in the current (extractive-only) results."""
    claims = _CLAIM_PATTERN.findall(draft_reply)
    reply_claims = set(re.findall(r"\$\d[\d,.]*|\b\d+%|\b\d+\s?\w+", draft_reply, re.I))
    return [c for c in reply_claims if c.lower() not in evidence_text]


def heuristic_judge(customer_msg: str, draft_reply: str, evidence: list[dict]) -> dict:
    if not draft_reply:
        return {
            "grounded_score": 0, "length_ok": False, "overlap_score": 0.0,
            "unsupported_claims": [], "method": "heuristic",
        }

    evidence_text = " ".join(e["support_reply"] for e in evidence).lower()
    reply_words = set(re.findall(r"\w+", draft_reply.lower()))
    evidence_words = set(re.findall(r"\w+", evidence_text))
    overlap = len(reply_words & evidence_words) / max(len(reply_words), 1)

    length_ok = 10 <= len(draft_reply.split()) <= 80
    unsupported = _unsupported_claims(draft_reply, evidence_text)
    return {
        "grounded_score": 1 if overlap > 0.25 else 0,
        "length_ok": length_ok,
        "overlap_score": round(overlap, 3),
        "unsupported_claims": unsupported,
        "method": "heuristic",
    }


def llm_judge(customer_msg: str, draft_reply: str, evidence: list[dict]) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set -- LLM judge unavailable in this environment")
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    evidence_text = "\n".join(f"- {e['support_reply']}" for e in evidence)
    prompt = (
        "Rate this customer support reply on 6 dimensions, each 1-5: "
        f"{', '.join(RUBRIC_DIMENSIONS)}. Reply with JSON only.\n\n"
        f"Customer message: {customer_msg}\nDraft reply: {draft_reply}\n"
        f"Historical evidence used:\n{evidence_text}\n"
    )
    resp = client.messages.create(
        model="claude-sonnet-5", max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"raw": resp.content[0].text, "method": "llm"}
