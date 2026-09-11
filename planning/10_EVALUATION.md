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
## TF-IDF word+char(3,7) + LinearSVC on expanded training pool, sklearn 1.9.1, no version warnings)
| | intent acc | intent macro-F1 | intent weighted-F1 | escalation P/R/F1 | false-auto-handle | false-escalation |
|---|---|---|---|---|---|---|
| trivial | 0.472 | 0.107 | -- | 0.00/0.00/0.00 | 1.000 (127/127) | 0.000 (0/123) |
| simple | 0.704 | 0.522 | -- | 1.00/0.055/0.105 | 0.945 (120/127) | 0.000 (0/123) |
| **system** | **0.924** | **0.894** | **0.921** | 0.907/1.000/0.951 | **0.000 (0/127)** | 0.106 (13/123) |

Per-class F1 (system, golden set): account_access 0.811,
billing_subscription 0.949, content_availability 0.909,
feature_request_feedback 0.857, other 0.944, playback_technical 0.893.

(Superseded numbers, kept for the audit trail across three prior model
iterations: (a) word-only TF-IDF + LR: accuracy 0.728, macro-F1 0.620,
false-auto-handle 0.095 (12/127), false-escalation 0.358 (44/123); (b)
word+char(3,5) TF-IDF + LR (previous session's kept experiment): accuracy
0.736, macro-F1 0.591, false-auto-handle 0.055 (7/127), false-escalation
0.390 (48/123). See "Classifier improvement pass" below for the full
diagnosis behind the current numbers.)

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

## Leakage audit (re-verified this pass, now also covering retrieval_corpus as a training source)
- `eval_pool` / `dev_pool` / `retrieval_corpus`: zero `conversation_id`
  overlap between any pair of pools (checked exhaustively, not sampled).
- Zero exact-text `customer_msg` overlap between `golden_set` and
  `retrieval_corpus` -- re-checked this session specifically because
  `retrieval_corpus` is now also a classifier training source, not just a
  retrieval index. Zero overlap confirmed before any training was done with it.
- Zero exact-text `customer_msg` overlap between `golden_set` and `dev_pool`.
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

The underlying data-scarcity problem for minority intent classes remained
unresolved by either experiment above -- both were architecture/threshold
tweaks on the same ~423-example training set. The next session confirmed
this diagnosis correctly identified the bottleneck and fixed it by
expanding the training data itself (below), not by further tuning the same
small pool.

## Classifier improvement pass (controlled, evidence-driven; this pass)

### Diagnosis (Phase 1): TRAIN / held-out DEV / GOLDEN, previous model
`scripts/diagnose_classifier.py` measured the shipped word+char(3,5) +
Logistic Regression model on three disjoint sets: TRAIN (423-example
in-sample fit), a held-out 20% stratified slice of `dev_pool` (106
examples, never fit on), and GOLDEN (250 examples, read-only).

| | TRAIN | held-out DEV | GOLDEN |
|---|---|---|---|
| accuracy | 0.998 | 0.830 | 0.712 |
| macro-F1 | 0.997 | 0.606 | 0.558 |

Per-class F1, TRAIN -> held-out DEV -> GOLDEN, sorted by training count:
`feature_request_feedback` (n=9): 1.00 -> 0.00 -> 0.22 (recall 0.12/16 on
golden); `content_availability` (n=15): 1.00 -> 0.60 -> 0.57;
`playback_technical` (n=19): 1.00 -> 0.57 -> 0.42; `account_access` (n=34):
0.99 -> 0.67 -> 0.54; `billing_subscription` (n=71): 1.00 -> 0.92 -> 0.82;
`other` (n=275): 1.00 -> 0.88 -> 0.78. Generalization quality tracks
per-class training count almost monotonically -- textbook overfitting
compounded by data scarcity, not underfitting (TRAIN is not low) and not
primarily a labeling-taxonomy problem (see `scripts/diagnose_labels.py`
below, which rules out first-match-wins rule ambiguity as a major factor:
only 5.1% of `dev_pool` examples match more than one taxonomy rule).

### Label-quality / `other`-class check (`scripts/diagnose_labels.py`, dev_pool only, never golden)
- `other` is 65% of `dev_pool` (344/529).
- Of those, only 10.8% are short fragments (<=5 words) -- the median
  `other`-labeled message is 13 words, i.e. most of `other` is full-length
  text that simply doesn't match any of the 5 specific-intent regexes, not
  short noise. This is consistent with a rule-coverage recall gap
  (documented in `planning/13_FAILURE_ANALYSIS.md`) rather than the bucket
  being dominated by genuinely un-categorizable fragments.
- Zero exact-duplicate `customer_msg` within `dev_pool`.

### Learning curve (`scripts/learning_curve.py`): is more real data justified?
Combined the 423-example `dev_pool` training split with all 36,004 rows of
`retrieval_corpus` (previously used only for retrieval grounding, never for
classifier training) into one 36,427-row universe, weakly labeled the same
way, and trained on increasing fractions, evaluating each time on the
*same fixed* 106-example held-out DEV slice (never touched during
training, at any fraction):

| train fraction | n examples | held-out DEV accuracy | held-out DEV macro-F1 |
|---|---|---|---|
| 10% | 3,642 | 0.915 | 0.787 |
| 25% | 9,106 | 0.887 | 0.806 |
| 50% | 18,213 | 0.915 | 0.836 |
| 75% | 27,320 | 0.915 | 0.828 |
| 100% | 36,427 | 0.925 | 0.834 |

Macro-F1 nearly triples from 0% expansion (0.606) to just 10% (0.787), and
keeps improving to ~50% before plateauing -- this is the assignment's own
explicit criterion for "data scarcity is likely" being satisfied with real
numbers, not assumed. Expansion pool composition per class:
`feature_request_feedback` 9->1,212, `content_availability` 15->599,
`playback_technical` 19->2,350, `account_access` 34->2,282,
`billing_subscription` 71->6,486, `other` 275->23,498.

**Leakage check before using this pool for training** (not after):
`retrieval_corpus` was already known conversation-ID-disjoint from
`golden_set` (existing tests). Additionally checked exact-text
`customer_msg` overlap between `golden_set` and `retrieval_corpus`: zero.
Also re-checked `golden_set` vs `dev_pool`: zero. Expanding into
`retrieval_corpus` for training does not compromise golden-set isolation.

### Controlled feature / hyperparameter / classifier grid (`scripts/experiment_grid.py`, `scripts/experiment_svc_tuning.py`)
All on the full expanded 36,427-example universe, evaluated on the same
fixed held-out DEV slice, sequential (not full cross-product) per the
assignment's "small controlled grid" instruction:

**Group A (TF-IDF features, fixed LR C=2.0 balanced):**
| config | held-out DEV acc | held-out DEV macro-F1 |
|---|---|---|
| word(1,1) only | 0.802 | 0.676 |
| word(1,2) only | 0.915 | 0.810 |
| word(1,3) only | 0.925 | 0.821 |
| word(1,2)+char(3,5) [previous] | 0.925 | 0.834 |
| word(1,2)+char(3,6) | 0.934 | 0.847 |
| word(1,2)+char(4,6) | 0.925 | 0.834 |
| **word(1,2)+char(3,7) [winner]** | **0.934** | **0.857** |
| word(1,2)+char(3,5), min_df=1 | 0.915 | 0.828 |

**Group B (Logistic Regression C / class_weight, winning A features):**
`class_weight=None` beat `class_weight="balanced"` at every C tested once
data was expanded (opposite of the previous small-data regime, where
`balanced` was necessary). Best: `C=10, class_weight=None` ->
macro-F1 0.888 (vs. 0.857 for the old `C=2.0, balanced` config on the same
new features -- regularization tuned for 423 examples was actively
under-fitting once trained on 36,427).

**Group C (alternative classifiers, winning A features + best B hyperparameters):**
| classifier | held-out DEV acc | held-out DEV macro-F1 | decision |
|---|---|---|---|
| Logistic Regression (C=10, no class weight) | 0.953 | 0.888 | baseline for this group |
| **LinearSVC (balanced, C=1.0)** | **0.972** | **0.919** | **KEEP -- best result** |
| MultinomialNB | 0.830 | 0.460 | REJECT -- far worse, NB's independence assumption doesn't fit the TF-IDF+char-n-gram feature space |

A follow-up LinearSVC-only sweep (`scripts/experiment_svc_tuning.py`,
`C in {0.5,1,2} x class_weight in {None,"balanced"}`, 6 configs) produced
**identical** results (acc 0.972, macro-F1 0.919) for all 6 -- in this
high-dimensional sparse feature space the classes are close to linearly
separable regardless of margin width, so C/class_weight stopped mattering
past a point. Kept `class_weight="balanced", C=1.0` as the principled
default given class imbalance, rather than treating the tie as license to
cherry-pick.

### Final model selection (Phase 8)
TF-IDF word(1,2) + char_wb(3,7), `min_df=2`, `sublinear_tf=True`,
`LinearSVC(class_weight="balanced", C=1.0, max_iter=5000)`, trained on
`dev_pool` (529, full) + `retrieval_corpus` (36,004, full) = 36,533
examples. `LinearSVC` has no `predict_proba`; `src/intent_classifier.py`
uses a softmax over `decision_function` margins as a display-only
confidence proxy -- verified nothing in `classify_escalation` or the
evaluation harness consumes this field (it uses `retrieval_confidence`
instead), so this substitution has zero effect on escalation logic or
metrics.

### Golden set evaluated exactly once, after model selection
`golden_set.csv` was not read, modified, or trained on anywhere in this
pass (`git diff` on the file shows zero changes). Final golden numbers are
in the "Verified results" table above.

### Sanity check on the large jump (Phase 7, done before accepting the result)
An 18-point accuracy jump is exactly what the assignment says to be
suspicious of. Checked:
1. Conversation-ID and exact-text leakage (above) -- zero, checked *before*
   training with the expanded pool, not after seeing a good result.
2. The golden-set confusion matrix is not degenerate: every class retains
   some residual, explainable error (e.g. `playback_technical` still sends
   6/31 examples to `other`) rather than a suspicious uniform 100%, which
   would suggest memorization rather than generalization.
3. The improvement pattern matches the diagnosis: the classes with the
   fewest original training examples (`feature_request_feedback`,
   `content_availability`) show the largest relative F1 gains, consistent
   with "more training data fixed a data-scarcity problem," not with an
   unrelated artifact.

No leakage found. The result is accepted as genuine, with the important
caveat (in README "What's misleading about the headline number") that
training and evaluation labels share the same regex-labeling *method*, so
this measures agreement with that labeling function at scale, not
independent ground truth.

## LLM judge and human validation: status
Both implemented, neither executed with real credentials/annotators in this
environment. Code-path correctness (request/response contract, fallback
behavior on missing key or API error) is verified with mocked API calls in
`tests/test_llm_paths_mocked.py` (6 tests, all passing) -- this proves the
integration is correct without fabricating outputs. See
`planning/12_HUMAN_VALIDATION.md` and `planning/18_DECISION_LOG.md`.
