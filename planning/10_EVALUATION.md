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

## Verified results (fresh model retrain via `scripts/run_all.py`,
## word+char n-gram classifier, sklearn 1.9.1, no version warnings)
| | intent acc | intent macro-F1 | escalation P/R/F1 | false-auto-handle | false-escalation |
|---|---|---|---|---|---|
| trivial | 0.472 | 0.107 | 0.00/0.00/0.00 | 1.000 (127/127) | 0.000 (0/123) |
| simple | 0.704 | 0.522 | 1.00/0.047/0.090 | 0.953 (121/127) | 0.000 (0/123) |
| **system** | **0.736** | **0.591** | 0.714/0.945/0.814 | **0.055 (7/127)** | 0.390 (48/123) |

(Superseded numbers from an earlier word-only classifier, kept for the
audit trail: accuracy 0.728, macro-F1 0.620, false-auto-handle 0.095
(12/127), false-escalation 0.358 (44/123) -- see "Char n-gram experiment"
below for why the numbers changed and why the change was kept.)

## Two reproducibility bugs found and fixed across sessions

**Bug 1 (session 2): unpinned scikit-learn.** Re-running
`scripts/evaluate.py` against a previously committed model pickle (trained
under scikit-learn 1.3.2, loaded under 1.9.1 installed in this venv)
produced different numbers with an `InconsistentVersionWarning`. Fixed by
pinning `scikit-learn==1.9.1` in `requirements.txt` and retraining fresh.

**Bug 2 (session 3, more serious): `run_all.py` used a bare `"python"`
instead of `sys.executable`.** In this environment, bare `python` on PATH
resolves to a completely unrelated system Python installation with
scikit-learn 1.3.2 -- *not* the project's `.venv` (1.9.1, matching
`requirements.txt`'s pin). This meant **the documented one-command
reproduction path (`python scripts/run_all.py`) silently ignored the pinned
dependency versions on every run**, regardless of the sklearn pin from
Bug 1's fix. It was internally self-consistent (train and evaluate both
happened under system Python 1.3.2), which is why it didn't surface as a
version-mismatch warning during `run_all.py` itself -- it only showed up
when directly re-running `scripts/evaluate.py` under `.venv` afterward and
finding a mismatch against whatever `run_all.py` had most recently trained.
Fixed by using `sys.executable` for every subprocess step in
`scripts/run_all.py`. Re-ran the full pipeline after the fix: zero version
warnings, fully self-consistent, and this is now the verified canonical
result.

This means the "byte-identical" reproducibility claim from the previous
session's audit was checking the wrong thing (manual `.venv`-only
invocations matching each other) rather than verifying that the actually
documented command (`python scripts/run_all.py`) used the pinned
environment at all. Both bugs are now fixed and the full run is verified
self-consistent under `sys.executable`.

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

Investigated why 100% of escalation errors (55/55 in the current run, 56/56
in the previous session -- both directions) co-occur with the intent
classifier or golden labeler assigning `other` (more precise than the
earlier ~85% estimate; see `planning/13_FAILURE_ANALYSIS.md`). Root cause:
`other` is 65% of `dev_pool` weak training labels (344/529), so even with
`class_weight="balanced"`, the classifier over-predicts it. Confirmed this
is a data-scarcity problem, not purely an architecture problem:
`feature_request_feedback` has only 11 `dev_pool` examples and
`content_availability` only 19 -- both too few for any TF-IDF classifier to
learn robustly regardless of features or thresholds.

**Experiment 1 (rejected): confidence-override threshold.** Override raw
argmax with the best non-`other` class when its probability clears a
threshold, tuned on an 80/20 held-out split of `dev_pool` only (never the
golden set) -- see `scripts/tune_other_threshold.py`. Best threshold (0.30)
improved macro-F1 on that held-out dev slice from 0.595 to 0.622, but
checked once against the frozen golden set, made every headline number
worse (accuracy 0.728->0.716, macro-F1 0.620->0.605, false-auto-handle
0.095->0.134). Reverted to plain argmax.

**Experiment 2 (kept): word+char n-gram features.** Added character
n-grams (3-5 chars, `analyzer="char_wb"`) alongside the existing word
n-grams via `FeatureUnion`, hypothesizing this would help match word-form
variants (crash/crashes/crashing) without needing more labeled examples --
see `scripts/experiment_char_ngrams.py`. Tuned/checked on the same 80/20
`dev_pool` split first (macro-F1 0.595->0.606 there, with
`feature_request_feedback` recall staying at 0 due to only 2 validation
examples -- confirming the scarcity diagnosis independent of features).
Checked once against the frozen golden set:

| | word-only | word+char n-grams |
|---|---|---|
| intent accuracy | 0.728 | 0.736 |
| intent macro-F1 | 0.620 | 0.591 |
| escalation F1 | 0.804 | 0.814 |
| false-auto-handle rate | 0.095 (12/127) | 0.055 (7/127) |
| false-escalation rate | 0.358 (44/123) | 0.390 (48/123) |

This is a genuine tradeoff, not a clean win: per-class inspection shows
char n-grams make the classifier predict `other` *more* often (160 vs. 150
predictions), which improves recall on the majority `other` class (helping
the headline false-auto-handle metric) at the cost of recall on minority
classes like `playback_technical` (recall dropped to 0.29) and
`feature_request_feedback` (recall dropped to 0.19) -- hence lower
macro-F1 and more false escalations. **Kept** because the assignment
explicitly weighs false-auto-handle as the worst failure mode, and this
change trades secondary metrics for exactly that one, unlike Experiment 1
which regressed everything including the primary metric. Documented with
full numbers, not presented as an unqualified improvement. See
`planning/18_DECISION_LOG.md` #15 and #18.

The underlying data-scarcity problem for minority intent classes remains
unresolved by either experiment -- a valid engineering conclusion (see
`planning/13_FAILURE_ANALYSIS.md`), not something threshold or feature
tuning on the same ~11-19-example classes can fix.

## LLM judge and human validation: status
Both implemented, neither executed with real credentials/annotators in this
environment. Code-path correctness (request/response contract, fallback
behavior on missing key or API error) is verified with mocked API calls in
`tests/test_llm_paths_mocked.py` (4 tests, all passing) -- this proves the
integration is correct without fabricating outputs. See
`planning/12_HUMAN_VALIDATION.md` and `planning/18_DECISION_LOG.md`.
