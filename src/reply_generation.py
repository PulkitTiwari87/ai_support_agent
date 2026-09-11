"""Grounded reply generation.

If ANTHROPIC_API_KEY is set, uses Claude to write a reply grounded strictly
in the retrieved historical evidence. Otherwise falls back to an extractive
template (adapts the best-matching historical reply) -- this keeps the
pipeline runnable end-to-end without API credentials, at the cost of fluency.
Both paths refuse to answer confidently when evidence is weak.

The two paths are NOT reported interchangeably in evaluation: the LLM path
is used only when its logs confirm the API was actually called.
"""
import os
import re

LOW_EVIDENCE_SIMILARITY = 0.12


def _clean_agent_signoff(text: str) -> str:
    text = re.sub(r"\s?[\^/][A-Z]{2,3}\.?$", "", text)
    return text.strip()


def generate_extractive(customer_msg: str, evidence: list[dict]) -> dict:
    if not evidence or evidence[0]["similarity"] < LOW_EVIDENCE_SIMILARITY:
        return {
            "draft_reply": None,
            "grounded": False,
            "evidence_used": [],
            "method": "extractive",
        }
    best = evidence[0]
    reply = _clean_agent_signoff(best["support_reply"])
    return {
        "draft_reply": reply,
        "grounded": True,
        "evidence_used": [best],
        "method": "extractive",
    }


def generate_llm(customer_msg: str, evidence: list[dict]) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    # Check evidence before attempting to import anthropic to avoid unnecessary import errors
    if not evidence or evidence[0]["similarity"] < LOW_EVIDENCE_SIMILARITY:
        return {"draft_reply": None, "grounded": False, "evidence_used": [], "method": "llm"}
    try:
        import anthropic
    except ModuleNotFoundError as e:
        raise RuntimeError("anthropic package is required for LLM generation but is not installed") from e
    client = anthropic.Anthropic(api_key=api_key)

    evidence_text = "\n".join(
        f"- Past customer issue: {e['customer_msg']}\n  Past Spotify reply: {e['support_reply']}"
        for e in evidence
    )
    prompt = (
        "You are a Spotify customer support agent. Write a short, helpful reply "
        "to the customer message below, grounded ONLY in the historical "
        "examples provided. Do not invent policies, promises, or facts not "
        "supported by the evidence. If the evidence does not clearly cover "
        "this issue, say you'll need to look into it rather than guessing.\n\n"
        f"Customer message: {customer_msg}\n\n"
        f"Historical evidence:\n{evidence_text}\n\n"
        "Reply (2-3 sentences, no policy invention):"
    )
    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    return {"draft_reply": text, "grounded": True, "evidence_used": evidence, "method": "llm"}


def generate_reply(customer_msg: str, evidence: list[dict]) -> dict:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return generate_llm(customer_msg, evidence)
        except Exception:
            pass
    return generate_extractive(customer_msg, evidence)
