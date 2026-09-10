# Evaluation Methodology

## What is measured, on what data
All three systems (`baseline_trivial`, `baseline_simple`, `system_full`)
run against the identical frozen `data/processed/golden_set.csv` (250
examples), using the identical metric functions in
`scripts/evaluate.py::compute_metrics`. No system gets a different
evaluation set or a different metric definition. Reproduce with
`python scripts/evaluate.py`.

## Metric denominators (verified, not assumed)
- `n = 250` for every system.
- `should_escalate == True`: 127/250 (50.8%) -- this is the denominator for
  escalation recall and false-auto-handle rate.
- `should_escalate == False`: 123/250 (49.2%) -- denominator for false
  escalation rate.
- Intent class distribution: `other` 118, `billing_subscription` 58,
  `playback_technical` 31, `account_access` 21, `feature_request_feedback`
  16, `content_availability` 6.

## Verified results (fresh model retrain, current sklearn 1.9.1)
| | intent acc | intent macro-F1 | escalation P/R/F1 | false-auto-handle | false-escalation |
|---|---|---|---|---|---|
| trivial | 0.472 | 0.107 | 0.00/0.00/0.00 | 1.000 (127/127) | 0.000 (0/123) |
| simple | 0.704 | 0.522 | 1.00/0.047/0.090 | 0.953 (121/127) | 0.000 (0/123) |
| **system** | **0.728** | **0.620** | 0.723/0.906/0.804 | **0.095 (12/127)** | 0.358 (44/123) |

## A reproducibility bug found and fixed during this pass
Re-running `scripts/evaluate.py` against the *previously committed* model
pickle (trained under scikit-learn 1.3.2, loaded under 1.9.1 installed in
this venv) produced **different** numbers (intent accuracy 0.68 instead of
0.728, false-auto-handle-rate 0.173 instead of 0.095) with an
`InconsistentVersionWarning`. This was a real correctness bug: an unpinned
`scikit-learn` version in `requirements.txt` meant the persisted model could
silently diverge from what a fresh install would train. Fixed by:
1. Pinning `scikit-learn==1.9.1` in `requirements.txt`.
2. Retraining fresh (`python src/intent_classifier.py && python
   src/retrieval.py`) and re-verifying the numbers above match the
   originally reported ones -- they do, exactly, confirming the *original*
   reported numbers were correct and the drift was purely a stale-pickle
   artifact, not a real regression.
3. `scripts/run_all.py` already retrains from scratch every run, so this
   bug could only have surfaced by running `scripts/evaluate.py` in
   isolation against a stale artifact -- documented here so it doesn't
   recur silently.

## Leakage audit (re-verified this pass)
- `eval_pool` / `dev_pool` / `retrieval_corpus`: zero `conversation_id`
  overlap between any pair of pools (checked exhaustively, not sampled).
- Zero exact-text `customer_msg` overlap between `golden_set` and
  `retrieval_corpus`.
- Zero exact-duplicate `customer_msg` within `golden_set` itself.
- One normalized (lowercased, punctuation-stripped) near-duplicate:
  `"can you help me please?"` appears in both the golden set and the
  retrieval corpus, from two different conversations. This is not leakage
  (different customers, different underlying issues) -- it's a generic
  boilerplate phrase, and a real limitation of TF-IDF retrieval on
  content-free short messages (see `planning/13_FAILURE_ANALYSIS.md`,
  failure mode #4).
- 80 exact-duplicate `customer_msg` rows exist *within* the 36,004-row
  retrieval corpus itself (different conversations, same generic phrase,
  e.g. "Thanks!"). Not a leakage issue -- retrieval corpus duplicates don't
  affect eval/dev independence -- but noted as a data-quality property.

## What "improve the largest failure mode" produced
Investigated why ~100% of escalation errors (56/56, both directions)
co-occur with the intent classifier or golden labeler assigning `other`
(more precise than the earlier ~85% estimate; see
`planning/13_FAILURE_ANALYSIS.md`). Root cause: `other` is 65% of
`dev_pool` weak training labels (344/529), so even with
`class_weight="balanced"`, the classifier over-predicts it.

**Experiment**: override raw argmax with the best non-`other` class when its
probability clears a threshold, tuned on an 80/20 held-out split of
`dev_pool` only (never the golden set) -- see
`scripts/tune_other_threshold.py`. Best threshold (0.30) improved macro-F1
on that held-out dev slice from 0.595 to 0.622.

**Result on the frozen golden set (checked once)**: this made every
headline number worse -- intent accuracy 0.728->0.716, macro-F1
0.620->0.605, and critically the false-auto-handle rate rose from 0.095 to
0.134 (12->17 missed escalations). **Rejected** per the regression policy;
reverted to plain argmax. See `planning/18_DECISION_LOG.md` for the full
writeup. The dev-tuned threshold did not generalize, most likely because
the held-out dev slice and the golden set -- despite both being drawn from
the same rule-labeling family -- have different enough composition that a
threshold fit to one doesn't transfer. This is left as a documented,
unresolved limitation rather than iterated further against the golden set.

## LLM judge and human validation: status
Both implemented, neither executed with real credentials/annotators in this
environment. Code-path correctness (request/response contract, fallback
behavior on missing key or API error) is verified with mocked API calls in
`tests/test_llm_paths_mocked.py` (4 tests, all passing) -- this proves the
integration is correct without fabricating outputs. See
`planning/12_HUMAN_VALIDATION.md` and `planning/18_DECISION_LOG.md`.
