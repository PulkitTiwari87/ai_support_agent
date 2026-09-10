"""Verifies the LLM-backed code paths (reply generation, judge) are wired
correctly WITHOUT making a real API call -- no ANTHROPIC_API_KEY is
available in this environment (see planning/18_DECISION_LOG.md). Mocks the
anthropic client to prove the request/response contract is correct, so that
supplying a real key is the only thing needed to make these paths live."""
import sys
import types
from unittest.mock import patch, MagicMock

sys.path.insert(0, "src")


def _fake_anthropic_module(response_text):
    fake_module = types.ModuleType("anthropic")

    class FakeContent:
        def __init__(self, text):
            self.text = text

    class FakeResponse:
        def __init__(self, text):
            self.content = [FakeContent(text)]

    class FakeMessages:
        def create(self, **kwargs):
            assert "model" in kwargs
            assert "messages" in kwargs
            assert kwargs["messages"][0]["role"] == "user"
            return FakeResponse(response_text)

    class FakeAnthropic:
        def __init__(self, api_key):
            assert api_key == "fake-test-key"
            self.messages = FakeMessages()

    fake_module.Anthropic = FakeAnthropic
    return fake_module


def test_generate_llm_calls_api_and_parses_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    fake_mod = _fake_anthropic_module("Sorry about that! Please try restarting the app.")
    with patch.dict(sys.modules, {"anthropic": fake_mod}):
        from reply_generation import generate_llm
        evidence = [{"customer_msg": "app won't start", "support_reply": "Restart the app", "similarity": 0.6}]
        result = generate_llm("app crashes on launch", evidence)
        assert result["method"] == "llm"
        assert result["grounded"] is True
        assert "restarting" in result["draft_reply"].lower()


def test_generate_llm_without_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from reply_generation import generate_llm
    import pytest
    with pytest.raises(RuntimeError):
        generate_llm("issue", [{"customer_msg": "x", "support_reply": "y", "similarity": 0.5}])


def test_generate_llm_low_evidence_skips_api_call(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    from reply_generation import generate_llm
    result = generate_llm("issue", [{"customer_msg": "x", "support_reply": "y", "similarity": 0.01}])
    assert result["grounded"] is False
    assert result["draft_reply"] is None


def test_llm_judge_calls_api_and_returns_raw_text(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    fake_mod = _fake_anthropic_module('{"correctness": 4, "relevance": 5}')
    with patch.dict(sys.modules, {"anthropic": fake_mod}):
        from judge import llm_judge
        evidence = [{"customer_msg": "x", "support_reply": "Restart the app", "similarity": 0.6}]
        result = llm_judge("app crashes", "Please restart the app", evidence)
        assert result["method"] == "llm"
        assert "correctness" in result["raw"]


def test_llm_judge_without_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from judge import llm_judge
    import pytest
    with pytest.raises(RuntimeError):
        llm_judge("issue", "reply", [])


def test_generate_reply_falls_back_when_api_errors(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")

    fake_mod = types.ModuleType("anthropic")

    class BrokenAnthropic:
        def __init__(self, api_key):
            raise ConnectionError("simulated network failure")

    fake_mod.Anthropic = BrokenAnthropic
    with patch.dict(sys.modules, {"anthropic": fake_mod}):
        from reply_generation import generate_reply
        evidence = [{"customer_msg": "x", "support_reply": "Try this. /AB", "similarity": 0.5}]
        result = generate_reply("issue", evidence)
        assert result["method"] == "extractive"
        assert result["grounded"] is True
