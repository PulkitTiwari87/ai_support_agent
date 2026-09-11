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
| **Full system** | **0.924** | **0.894** | 0.907 / 1.000 / 0.951 | **0.000** (0/127) | 0.106 (13/123) |

(weighted-F1 0.921; per-class F1: account_access 0.811, billing_subscription
0.949, content_availability 0.909, feature_request_feedback 0.857, other
0.944, playback_technical 0.893 -- `data/processed/eval_results.json`.)

Reproduce: `python scripts/run_all.py` (retrains from scratch, no cached
model reused).

## Headline result
**Intent accuracy 73.6%->92.4%, macro-F1 59.1%->89.4%, and false
auto-handle rate 5.5%->0.0% (0/127), all improved together** -- not a
tradeoff. This came from a classifier improvement pass (see "Classifier
improvement pass" below), not from touching the golden set, the taxonomy,
or the escalation rules.

Unlike the previous session's char-n-gram change (which traded macro-F1
for false-auto-handle), every metric moved in the same direction this time:
false-escalation rate also fell (39.0%->10.6%). The reason this was
possible is diagnostic, not architectural: the earlier classifier was
trained on only 529 examples and was badly overfit (99.8% train accuracy vs
60.6% held-out-dev macro-F1); the fix was mostly about training data
volume, not a cleverer model.

## Classifier improvement pass (this session)
**Diagnosis first.** Measured the previous model on TRAIN (in-sample),
a held-out 20% slice of `dev_pool` never seen during fit, and GOLDEN:

| | TRAIN | held-out DEV | GOLDEN |
|---|---|---|---|
| accuracy | 0.998 | 0.830 | 0.712 |
| macro-F1 | 0.997 | 0.606 | 0.558 |

A ~40-point macro-F1 gap between TRAIN and held-out DEV is textbook
overfitting, not underfitting. Per-class F1 fell in lockstep with per-class
training-example count in `dev_pool` (billing_subscription, 71 examples:
dev F1 0.92; feature_request_feedback, 9 examples: dev F1 0.00, i.e. 0/2
correct on held-out data despite perfect training-set recall). This is
quantified evidence of data scarcity, not just an architecture problem --
see `planning/10_EVALUATION.md` for the full diagnosis
(`scripts/diagnose_classifier.py`, `scripts/diagnose_labels.py`).

**Data expansion, justified by a learning curve, not assumed.**
`retrieval_corpus.csv` (36,004 real historical SpotifyCares messages, same
dataset/brand, already conversation-level *and* exact-text disjoint from
the golden set -- re-verified this session) had never been used to train
the classifier, only for retrieval grounding. A learning curve
(`scripts/learning_curve.py`) trained on increasing fractions of
`dev_train + retrieval_corpus`, evaluated each time on the *same* fixed
held-out DEV slice: macro-F1 rose from 0.606 (0% expansion) to 0.787 at
just 10% of the expanded pool, continuing to ~0.84 by 50%, then plateauing
-- clear evidence more real data helps, per the assignment's own "if DEV
keeps improving substantially, data scarcity is likely" test.

**Controlled feature/model search on the expanded pool only** (never
golden), sequential small grids, not exhaustive (`scripts/experiment_grid.py`,
`scripts/experiment_svc_tuning.py`):
- TF-IDF features: word(1,2)+char_wb(3,7) beat word-only and the previous
  char(3,5) config on held-out DEV macro-F1 (0.857 vs 0.821 vs 0.834).
- Logistic Regression C/class_weight: heavier regularization
  (`class_weight="balanced", C=2.0`, the previous config) was actively
  hurting -- with 60x more training data, `class_weight=None, C=10` scored
  0.888 vs 0.857 for the old regularization on the same features.
- Classifier family: LinearSVC (0.919 macro-F1 on held-out DEV) beat both
  the best Logistic Regression config (0.888) and MultinomialNB (0.460,
  rejected outright). LinearSVC's own C/class_weight barely mattered here
  (all 6 tested combinations scored identically, 0.9188) -- the
  high-dimensional sparse feature space is close to linearly separable
  regardless of margin width, so `class_weight="balanced", C=1.0` (a
  principled default given class imbalance) was kept rather than
  cherry-picking among ties.

**Golden set touched exactly once**, after model selection was already
done on TRAIN/DEV, per the assignment's protocol. `golden_set.csv` itself
was never read, modified, or trained on by any part of this pass (verified:
`git diff` shows zero changes to the file).

**Sanity-checked the jump, not just accepted it**: an 18-point accuracy
jump is exactly the kind of result the assignment says to be suspicious of.
Checked: (1) conversation-level AND exact-text leakage between
`retrieval_corpus`/`dev_pool` and `golden_set` -- zero overlap, checked
before training; (2) the golden-set confusion matrix is not degenerate --
every class still has some residual, explainable error (e.g. 6/31
`playback_technical` examples still fall to `other`), not a suspicious
uniform 100% that would suggest memorization; (3) the improvement direction
matches the diagnosis exactly (biggest per-class gains are in the classes
that had the fewest training examples before). No leakage found.

Final model config: TF-IDF word(1,2) + char_wb(3,7), `min_df=2`,
`sublinear_tf=True`, `LinearSVC(class_weight="balanced", C=1.0)`, trained
on `dev_pool` (529) + `retrieval_corpus` (36,004) = 36,533 real,
weakly-labeled SpotifyCares messages. See `src/intent_classifier.py` and
`planning/18_DECISION_LOG.md` #22-24.

## What's misleading about the headline number
1. **The 250-example golden set's escalation ground truth is rule-derived,
   not independently human-labeled.** The training labels for the expanded
   pool come from the same regex taxonomy. A classifier that gets very good
   at reproducing that regex will score well on any evaluation set labeled
   by the same regex family, including the golden set -- this is a
   genuinely different, disjoint *split* of data (real conversations the
   model never saw), but not a genuinely independent *labeling method*. A
   truly independently human-labeled golden set could show a smaller gap
   than 92.4%.
2. **`other`'s near-perfect recall (118/118 on golden) reflects `other`
   being the easiest class to learn with abundant data (23,498/36,533
   training examples), not necessarily deep semantic understanding of
   ambiguous text.** The classifier now closely mirrors the rule-based
   labeler's own boundary for `other`, for the same reason as point 1.
3. **Reply groundedness is still reported as 100%, and that's still close
   to tautological**: the extractive generator (used because no API key was
   available) literally copies the retrieved historical reply. This
   classifier improvement pass did not touch reply generation at all.
4. **Single brand, single language (English), 2017-era tweets.** No claim
   of generalization to other brands, languages, or current-day Spotify UX
   is made or implied.
5. **The 0.0% false-auto-handle rate is not a validated real-world safety
   rate, and "zero" should be read with extra caution, not extra
   confidence.** It's a measurement against a 250-example, rule-derived,
   single-rater-spot-checked golden set (points 1-2 above) -- a genuinely
   different production system, different data era, or an independently
   human-labeled test set could all show a materially different number.
   Treat it as "how this system compares to two baselines on this specific
   test set," not as a calibrated estimate of how often a deployed version
   of this agent would fail to escalate a real customer issue.

## Baselines
- **Trivial**: majority-class intent (`other`), one canned reply, never
  escalates. `src/baselines.py::baseline_trivial`.
- **Simple**: single-keyword-per-intent lookup (deliberately blunter than
  the taxonomy regex, to avoid circularity with the golden labeler), one
  canned reply per intent, escalates only on explicit security/billing/
  human-request keywords. `src/baselines.py::baseline_simple`.

## Top failure modes (real examples, not generic AI weaknesses)
See `planning/13_FAILURE_ANALYSIS.md` for full detail with quoted examples.
The classifier improvement pass resolved what was previously the dominant
failure (100% of escalation errors tracing to `other`-class confusion,
now mostly gone -- 0 false-auto-handle, `other` recall 118/118 on golden).
Remaining failure modes:
1. **`playback_technical` still has the weakest per-class F1 (0.893) among
   well-represented classes** -- 6/31 golden examples still fall to
   `other`. Smaller residual than before, but not eliminated by more data
   alone.
2. Regex word-form brittleness missed a real account-takeover case
   ("hacking" vs. "hacked") until caught by manual spot-check and fixed
   (prior session).
3. Extractive-reply groundedness metric is near-tautological (see above) --
   unaffected by this session's classifier work, since reply generation
   wasn't touched.
4. Retrieval on very short/generic messages (<5 words) can clear the
   similarity threshold on superficial term overlap -- also unaffected by
   this session's changes, since retrieval wasn't touched.
5. **The training-label ceiling**: since the expanded training pool is
   weakly labeled by the same regex family as the golden set, the
   classifier's very high golden-set agreement partly reflects convergence
   toward the *labeling function*, not necessarily deeper semantic
   understanding beyond what regex-derived labels can teach it. See "What's
   misleading about the headline number" above.

## Limitations
- **Golden-set labels are rule-derived + single-rater spot-checked, not
  independently multi-annotator human-labeled.** See
  `planning/05_GOLDEN_SET.md`. This matters more now than before: the
  classifier is trained on 36,533 examples labeled by the same rule family,
  so high golden-set agreement is partly measuring rule-self-consistency at
  scale, not independent ground truth.
- **No live LLM judge or human validation was executed** -- both implemented
  and mock/contract-tested, neither run for real (no API key, no
  annotators). See the table above and `planning/12_HUMAN_VALIDATION.md`.
- **Reply-groundedness metric (100%) is near-tautological** under the
  current extractive generator, since the reply is literally copied from
  the retrieved evidence. Unaffected by this session's classifier changes.
- **Single brand, single language, 2017-era tweets** -- no generalization
  claim beyond this dataset.
- **`playback_technical` remains the weakest well-represented class**
  (F1 0.893 on golden) even after data expansion -- some residual overlap
  with `other` persists.

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
- `src/intent_classifier.py` -- trains the classifier on `dev_pool` + `retrieval_corpus` (36,533 examples)
- `src/retrieval.py` -- builds the TF-IDF retrieval index
- `scripts/diagnose_classifier.py`, `scripts/diagnose_labels.py` -- train/dev/golden diagnosis and label-quality checks (read-only, no training)
- `scripts/learning_curve.py` -- evidence that data expansion was justified
- `scripts/experiment_grid.py`, `scripts/experiment_svc_tuning.py` -- the controlled feature/hyperparameter/classifier search behind the current model
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
1. **Independent (non-rule-derived) re-labeling of a sample of the golden
   set** -- now the single highest-value item. With training data expanded
   to 36,533 regex-labeled examples, the biggest remaining question is how
   much of the 92.4% golden accuracy reflects real language understanding
   versus convergence toward the same regex family that built both the
   training labels and the evaluation labels. A human-labeled sample would
   directly answer this.
2. Run the real LLM judge and a genuine 2+ rater human validation pass with
   an API key and actual annotators (`planning/12_HUMAN_VALIDATION.md`).
3. Apply the same "expand training data from the untapped retrieval
   corpus" idea to reply generation and retrieval quality -- both were
   left untouched this session and still show the limitations in
   `planning/13_FAILURE_ANALYSIS.md`.
4. Reduce `playback_technical`'s residual overlap with `other` (still the
   weakest well-represented class, F1 0.893) -- likely needs targeted
   error analysis on the specific golden examples that still misclassify,
   not more of the same undifferentiated data.
5. Swap regex escalation triggers for stemmed/lemmatized or embedding-based
   matching, closing classes of word-form bugs like the "hacking" vs.
   "hacked" one found and fixed in a prior session.
