"""End-to-end pipeline: intent -> retrieval -> grounded reply -> escalation."""
import sys
sys.path.insert(0, "src")
from taxonomy import classify_escalation
from intent_classifier import load as load_intent_model, predict as predict_intent
from retrieval import load_index, retrieve
from reply_generation import generate_reply


class Pipeline:
    def __init__(self):
        self.intent_model = load_intent_model()
        self.vectorizer, self.matrix, self.corpus = load_index()

    def run(self, customer_msg: str, k: int = 3) -> dict:
        intent, confidence = predict_intent(self.intent_model, customer_msg)
        evidence = retrieve(customer_msg, self.vectorizer, self.matrix, self.corpus, k=k)
        top_sim = evidence[0]["similarity"] if evidence else 0.0

        gen = generate_reply(customer_msg, evidence)

        should_escalate, reason = classify_escalation(customer_msg, intent, retrieval_confidence=top_sim)
        if not gen["grounded"] and not should_escalate:
            should_escalate, reason = True, "insufficient_evidence"

        return {
            "intent": intent,
            "confidence": round(confidence, 3),
            "evidence": evidence,
            "draft_reply": gen["draft_reply"],
            "grounded": gen["grounded"],
            "generation_method": gen["method"],
            "escalate": should_escalate,
            "escalation_reason": reason,
        }


if __name__ == "__main__":
    p = Pipeline()
    examples = [
        "Someone accessed my account and changed my email, please help!",
        "I was charged twice for premium this month, can I get a refund?",
        "The app keeps crashing on my Sonos speaker after the last update.",
    ]
    for ex in examples:
        result = p.run(ex)
        print(f"\n> {ex}")
        print(f"  intent={result['intent']} conf={result['confidence']} "
              f"escalate={result['escalate']} ({result['escalation_reason']})")
        print(f"  reply: {result['draft_reply']}")
