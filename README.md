# Spotify Support Agent Prototype

A grounded customer-support triage agent for `@SpotifyCares`: classifies
intent, retrieves similar historical support conversations, drafts a reply
grounded in that history, and decides whether to escalate to a human. Built
and evaluated end-to-end against real Twitter customer-support data.

**Read this first:** several components (LLM-based reply generation, LLM
judge, human validation) are implemented but did not run in this
environment because no `ANTHROPIC_API_KEY` was configured and no human
annotators were available. Every number below is real and reproducible;
where an LLM/human component was skipped, that's stated explicitly rather
than faked. See `planning/18_DECISION_LOG.md` for every material decision
and why.

## Quickstart (~30 seconds after data download, <15 min total)

```bash
python -m venv .venv && source .venv/Scripts/activate  # or .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
python scripts/run_all.py
```

This downloads a ~210MB public dataset mirror (no auth needed), builds the
golden set, trains the intent classifier, builds the retrieval index, and
runs the full evaluation + baselines. Results land in
`data/processed/eval_results.json`. No API key required for this path.

Optional: `export ANTHROPIC_API_KEY=...` before running to use the real LLM
reply-generation and judge paths instead of the extractive/heuristic
fallbacks (`src/reply_generation.py`, `src/judge.py`).

## Why SpotifyCares
Scored 10 candidate brands on volume, resolution rate, and vocabulary
diversity. SpotifyCares is a single product with a genuinely bounded issue
space (vs. Amazon/Uber's sprawl across unrelated domains), at a size
(27,910 conversations) that's processable end-to-end in this session.
Full comparison: `planning/03_BRAND_SELECTION.md`.

## Intent taxonomy
Derived by reading ~150 sampled `dev_pool` messages, not borrowed from an
unrelated domain taxonomy (e.g. Banking77 doesn't map to music streaming).
Full rationale: `planning/04_INTENT_TAXONOMY.md`.

| Intent | Definition |
|---|---|
| `account_access` | Login, signup, password reset, account takeover/security |
| `billing_subscription` | Payment, premium plans, family/student plans, cancellation, refunds, regional pricing |
| `playback_technical` | App bugs, crashes, device/platform compatibility, sync/download loss |
| `content_availability` | Missing/duplicate songs, albums, artists; regional catalog/licensing |
| `feature_request_feedback` | Suggestions, editorial requests, praise/complaints with no actionable fix |
| `other` | Doesn't fit cleanly -- ambiguous/fragmentary text (see limitations: this bucket is the system's single largest source of error) |

## Implemented vs. executed vs. not possible here
| Component | Status |
|---|---|
| Data pipeline, brand selection, taxonomy | **Executed** -- real data, real numbers |
| Intent classifier, retrieval, extractive generation, escalation | **Executed** |
| Baselines (trivial, simple) | **Executed** |
| Evaluation harness, leakage checks | **Executed** |
| LLM reply generation (`src/reply_generation.py::generate_llm`) | **Implemented**, verified correct via mocked API test (`tests/test_llm_paths_mocked.py`), **not executed live** -- no `ANTHROPIC_API_KEY` in this environment |
| LLM judge (`src/judge.py::llm_judge`) | **Implemented**, mock-verified, **not executed live** -- same reason |
| Heuristic judge (fallback) | **Executed** -- results disclosed as heuristic, not LLM |
| Human-vs-LLM validation | **Not possible in this environment** -- no annotators. Annotation template + workflow fully specified (`planning/12_HUMAN_VALIDATION.md`), zero fabricated ratings |

## How it works
```
customer message
      |
      v
intent classification  (TF-IDF + Logistic Regression, trained on dev_pool)
      |
      v
historical retrieval    (TF-IDF cosine over 36,004 historical
                          (customer_msg, support_reply) pairs)
      |
      v
grounded reply generation  (extractive by default; LLM path via Claude
                             when ANTHROPIC_API_KEY is set)
      |
      v
escalation decision   (rules: account security, billing disputes, explicit
                        human requests, high distress, low retrieval
                        confidence, unclassified intent)
      |
      v
{intent, confidence, evidence, draft_reply, grounded, escalate, escalation_reason}
```
Code: `src/taxonomy.py`, `src/intent_classifier.py`, `src/retrieval.py`,
`src/reply_generation.py`, `src/pipeline.py`.

## Results (250-example frozen golden set, `data/processed/golden_set.csv`)

| | intent accuracy | intent macro-F1 | escalation P / R / F1 | false auto-handle rate | false escalation rate |
|---|---|---|---|---|---|
| Baseline 1 (trivial) | 0.472 | 0.107 | 0.00 / 0.00 / 0.00 | **1.000** (127/127) | 0.000 |
| Baseline 2 (simple keyword) | 0.704 | 0.522 | 1.00 / 0.055 / 0.105 | **0.945** (120/127) | 0.000 |
| **Full system** | **0.736** | **0.591** | 0.714 / 0.945 / 0.814 | **0.055** (7/127) | 0.390 (48/123) |

Reproduce: `python scripts/run_all.py` (retrains from scratch, no cached
model reused).

## Headline result
**False auto-handle rate drops from 100% (trivial) / 94.5% (simple keyword
baseline) to 5.5% with the full system** -- i.e., when a message actually
needed a human, the naive baselines almost always let it through as
auto-handled; the full system catches ~95% of those. This is the metric the
assignment prioritizes explicitly (auto-handling something that should have
escalated is the worst failure mode), so it's the headline, not raw
accuracy.

The cost of that gain: false-escalation rate is 39% and intent macro-F1
(0.591) is actually a bit lower than an earlier version of this classifier
(0.620) -- see "Char n-gram tradeoff" below. The system is deliberately
conservative, escalating some cases a human wouldn't have needed to see.
That's a real, disclosed tradeoff, not hidden in the summary table.

## Char n-gram tradeoff (this session's one kept experiment)
The intent classifier was changed from word-only TF-IDF to word+char
n-gram features (`src/intent_classifier.py`), tuned on a held-out slice of
`dev_pool` and checked once against the golden set. Effect on the frozen
golden set, char n-grams vs. word-only:

| | word-only (previous) | word+char n-grams (current) |
|---|---|---|
| intent accuracy | 0.728 | 0.736 |
| intent macro-F1 | 0.620 | 0.591 |
| escalation F1 | 0.804 | 0.814 |
| false-auto-handle rate | 0.095 (12/127) | **0.055 (7/127)** |
| false-escalation rate | 0.358 (44/123) | 0.390 (48/123) |

This is a genuine, explainable tradeoff, not a straight improvement: char
n-grams make the classifier more conservative (predicts `other` more
often), which cuts the headline false-auto-handle count nearly in half but
costs minority-class intent precision (macro-F1 down) and produces more
false escalations. Kept because the assignment explicitly prioritizes
minimizing false-auto-handle over minimizing false-escalation. A separate,
earlier experiment (a raw confidence-override threshold, see
`planning/18_DECISION_LOG.md` #15) was rejected because it made *every*
metric worse, including the headline one -- this one only trades secondary
metrics for the primary one, which is a defensible call, not a rejected
regression.

## What's misleading about the headline number
1. **The 250-example golden set's escalation ground truth is rule-derived,
   not independently human-labeled.** Both baseline 2 and the golden
   labeler share family resemblance in their keyword logic (see
   `planning/18_DECISION_LOG.md` #11 for how baseline 2 was deliberately
   made different to avoid literal circularity), but the fundamental
   labeling method (regex over English keywords) is the same across golden
   labels and parts of the system. A truly independent human-labeled set
   might show a smaller gap.
2. **100% of ALL escalation errors (55/55: both false-auto-handle and
   false-escalation) involve one root cause**: the intent classifier or
   the golden labeler collapsing ambiguous text to the catch-all `other`
   class (see `planning/13_FAILURE_ANALYSIS.md`, failure mode #1 -- verified
   precisely across two sessions, revising an earlier ~85% estimate). This
   means the headline number is really measuring "how well does the system
   handle the `other` bucket," not a broad, even spread of failure types.
   One fix (probability-override threshold) was tried and **made every
   metric worse**, and was reverted; a second fix (char n-grams) improved
   the headline metric specifically but at a real cost to macro-F1 and
   false-escalation rate (see "Char n-gram tradeoff" above) -- neither is a
   clean solution, both are disclosed with numbers
   (`planning/18_DECISION_LOG.md` #15, #18).
3. **Reply groundedness is reported as 100%, but that's close to
   tautological**: the extractive generator (used because no API key was
   available) literally copies the retrieved historical reply, so
   word-overlap-based grounding checks are mechanically satisfied. It does
   not mean the reply is contextually correct for the new customer's
   specific situation.
4. **Single brand, single language (English), 2017-era tweets.** No claim
   of generalization to other brands, languages, or current-day Spotify UX
   is made or implied.

## Baselines
- **Trivial**: majority-class intent (`other`), one canned reply, never
  escalates. `src/baselines.py::baseline_trivial`.
- **Simple**: single-keyword-per-intent lookup (deliberately blunter than
  the taxonomy regex, to avoid circularity with the golden labeler), one
  canned reply per intent, escalates only on explicit security/billing/
  human-request keywords. `src/baselines.py::baseline_simple`.

## Top failure modes (real examples, not generic AI weaknesses)
See `planning/13_FAILURE_ANALYSIS.md` for full detail with quoted examples:
1. Intent collapsing to `other` drives 100% of all escalation errors
   (55/55), in both directions. Root cause: minority intent classes have
   very few training examples (`feature_request_feedback` n=11,
   `content_availability` n=19 in `dev_pool` weak labels) -- likely a data
   scarcity problem, not a model/threshold problem. Two fixes were tried:
   a confidence-threshold override (regressed everything, reverted) and
   char n-gram features (improved the headline metric, cost macro-F1 --
   kept, disclosed as a tradeoff).
2. Regex word-form brittleness missed a real account-takeover case
   ("hacking" vs. "hacked") until caught by manual spot-check and fixed.
3. Extractive-reply groundedness metric is near-tautological (see above).
4. Retrieval on very short/generic messages (<5 words) can clear the
   similarity threshold on superficial term overlap.
5. The golden set's `other` bucket conflates genuinely ambiguous text with
   rule-coverage gaps that undercount real classes.

## Limitations
- **Golden-set labels are rule-derived + single-rater spot-checked, not
  independently multi-annotator human-labeled.** See
  `planning/05_GOLDEN_SET.md`.
- **No live LLM judge or human validation was executed** -- both implemented
  and mock/contract-tested, neither run for real (no API key, no
  annotators). See the table above and `planning/12_HUMAN_VALIDATION.md`.
- **Reply-groundedness metric (100%) is near-tautological** under the
  current extractive generator, since the reply is literally copied from
  the retrieved evidence. The added unsupported-claims check
  (`src/judge.py`) is honest about returning 0/250 for the same reason --
  it isn't a real signal until the LLM paraphrasing path runs.
- **Single brand, single language, 2017-era tweets** -- no generalization
  claim beyond this dataset.
- **The `other`-class confusion (100% of escalation errors) remains only
  partly mitigated.** One fix (confidence-threshold override) was tried and
  rejected; a second (char n-grams) is kept because it improves the
  headline metric, but it trades away macro-F1 and false-escalation rate
  to do it -- it does not resolve the underlying data-scarcity problem in
  minority intent classes.

## What was intentionally NOT built
- No production infrastructure, auth, queues, or orchestration frameworks
  -- single-process Python pipeline.
- No embeddings/vector DB -- TF-IDF is sufficient at 36K documents and
  keeps the quickstart path fully offline.
- No fine-tuned/custom LLM -- generation uses either extractive templating
  or a single Claude call per message, no training loop.
- No multi-turn conversation state -- each message is scored independently.
- No live human-vs-LLM judge study (no annotators/API key available; the
  annotation template exists at `planning/12_HUMAN_VALIDATION.md` and the
  judge code is ready to run given credentials).

## Reproducibility
- `scripts/fetch_data.py` -- downloads the dataset (public HF mirror, no auth)
- `scripts/process_data.py` -- parses, dedupes, splits into eval/dev/retrieval pools (conversation-level split, no leakage -- see tests)
- `scripts/build_golden_set.py` -- labels the frozen golden set
- `src/intent_classifier.py` -- trains the classifier
- `src/retrieval.py` -- builds the TF-IDF retrieval index
- `scripts/evaluate.py` -- runs system + both baselines, writes metrics
- `scripts/judge_replies.py` -- runs the reply-quality judge (heuristic fallback without an API key)
- `scripts/run_all.py` -- runs all of the above in order
- `tests/` -- data-leakage checks, taxonomy unit tests, reply-generation unit tests: `pytest tests/`

No secrets or credentials are committed. No raw/processed data over a few
MB is committed (`data/raw/` is gitignored; regenerate with
`scripts/fetch_data.py`).

**Two reproducibility bugs found and fixed across sessions**:
1. `requirements.txt` originally left `scikit-learn` unpinned, so a stale
   committed model pickle could silently produce different metrics under a
   newer installed sklearn. Fixed by pinning `scikit-learn==1.9.1`.
2. **More serious**: `scripts/run_all.py` originally shelled out to a bare
   `"python"` rather than `sys.executable`. In this environment that bare
   `python` resolved to an *entirely different, unrelated system Python
   install* with an old, unpinned scikit-learn (1.3.2) -- meaning the
   documented one-command reproduction path silently ignored the pinned
   `.venv` dependencies every time it ran. Fixed by using `sys.executable`
   for every subprocess step, so `run_all.py` always runs under whichever
   interpreter launched it. Re-verified after the fix: no version warnings,
   fully self-consistent results. See `planning/10_EVALUATION.md`.

**A safety-relevant escalation bug found and fixed this pass**: the
`account_security` escalation rule matched "hacked" but not "hacking" (a
real account-takeover message using the latter word-form would not have
triggered security escalation). Fixed in `src/taxonomy.py`. Since this rule
is shared with the golden-set labeler, the golden set was regenerated and
diffed against the previous frozen version before accepting the change:
**zero `should_escalate` labels changed** (still 127/250) -- only one row's
recorded escalation *reason* was corrected. See
`planning/18_DECISION_LOG.md` #20 for the full diff.

## Citations
- Dataset: Twitter customer-support conversations, originally published on
  Kaggle as `thoughtvector/customer-support-on-twitter`. Accessed here via
  the public HuggingFace mirror
  [`TNE-AI/customer-support-on-twitter-conversation`](https://huggingface.co/datasets/TNE-AI/customer-support-on-twitter-conversation)
  (no Kaggle auth was available in this environment; see
  `planning/18_DECISION_LOG.md` #1).
- Libraries: pandas, pyarrow, scikit-learn, numpy, anthropic (Python SDK),
  pytest -- see `requirements.txt`. No other third-party code, prompts, or
  methodology was copied from an external source.

## One additional week
1. Fix the `other`-class collapse (failure mode #1) -- highest-value single
   fix, would move both false-auto-handle and false-escalation rates
   together instead of trading one for the other. Two attempts so far
   (confidence-threshold override: rejected; char n-grams: kept as a
   partial, lopsided improvement) suggest the real fix is more/better
   labeled training data for minority classes
   (`feature_request_feedback` n=11, `content_availability` n=19), not
   further feature/threshold engineering on the same tiny label set.
2. Run the real LLM judge and a genuine 2+ rater human validation pass with
   an API key and actual annotators (`planning/12_HUMAN_VALIDATION.md`).
3. Swap regex escalation triggers for stemmed/lemmatized or embedding-based
   matching to close the "hacking" vs. "hacked" class of bug.
4. Replace TF-IDF retrieval with embeddings and evaluate whether retrieval
   quality (not just intent) is a bottleneck on short messages.
5. Independent (non-rule-derived) re-labeling of a subset of the golden set
   to quantify how much of the headline gap is measurement artifact vs. real.
