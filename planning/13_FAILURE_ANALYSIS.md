# Failure Analysis

Based on `data/processed/eval_results.json` and
`data/processed/system_predictions.csv` (system run on the 250-example
golden set). Regenerate with `python scripts/evaluate.py`.

## Top failure modes (real, inspected examples)

**Update (3rd pass -- classifier improvement pass, largely resolved)**:
a diagnosis-first classifier improvement pass (see
`planning/10_EVALUATION.md` "Classifier improvement pass") found the true
root cause was training-data scarcity for minority classes, expanded the
training pool 69x using the previously-untapped `retrieval_corpus.csv`
(same brand, leakage-safe), and re-tuned features/classifier on that
expanded pool. Result: **false-auto-handle eliminated entirely (0/127, was
7/127)**, and **false-escalation cut more than in half (13/123, was
48/123)** -- both directions of the same failure mode improved together,
unlike either of the two earlier fix attempts (below), which each only
moved one metric at the expense of another.

Earlier fix attempts, kept for the record:
- Probability-override threshold (tuned on held-out `dev_pool`, same 423
  examples): **rejected** -- made every golden-set metric worse, including
  false-auto-handle rate (0.095->0.134).
- Word+char(3,5) n-gram features (tuned on held-out `dev_pool`, same 423
  examples): **kept at the time** -- cut false-auto-handle from 0.095 to
  0.055 (12->7 cases), but macro-F1 dropped 0.620->0.591 and
  false-escalation rose 0.358->0.390. A tradeoff, not a fix -- superseded
  by the data-expansion pass below.

See `planning/18_DECISION_LOG.md` #15, #18, #22-24 and
`planning/10_EVALUATION.md`.

### 1. Intent collapsing to `other` drove most escalation errors -- now mostly resolved by training-data expansion, not by touching the taxonomy or golden labels
- **Root cause, confirmed with per-class evidence**: a diagnostic
  train/held-out-dev/golden comparison showed the previous classifier's
  per-class F1 tracked training-example count almost exactly
  (`feature_request_feedback`, 9 training examples: dev F1 0.00;
  `billing_subscription`, 71 examples: dev F1 0.92) -- textbook overfitting
  from too little data, not a labeling or architecture problem per se.
- **Fix**: expanded the training pool from 529 to 36,533 real,
  weakly-labeled SpotifyCares messages by reusing `retrieval_corpus.csv`
  (previously used only for retrieval grounding). A learning curve
  confirmed this was justified (held-out DEV macro-F1 0.606 -> 0.787 at
  just 10% of the expansion) before committing to it. Re-tuned TF-IDF
  features (word(1,2)+char(3,7)) and classifier (LinearSVC) on the
  expanded pool.
- **Result on golden**: false-auto-handle 7->0, false-escalation 48->13,
  intent macro-F1 0.591->0.894, `other`-class recall on golden 107/118 (old
  word-only model) -> 118/118.
- **Not fully eliminated**: the 13 remaining false-escalation cases are
  ALL still `pred_intent == other` while gold is a real class (mostly
  `playback_technical`, 6/13; `feature_request_feedback`, 4/13) --
  see example `sp_034363` (playback) and `sp_027899` (feature request).
  Same failure mode, smaller residual. More data plus a genuinely
  independent (non-regex) label source is the most likely next lever
  (see README "One additional week" #1).
- **Important caveat, not new but sharper now**: training labels for the
  expanded pool come from the same regex family that built the golden
  labels. High agreement partly reflects the classifier converging on the
  labeling function itself, not necessarily independent ground truth. See
  `planning/10_EVALUATION.md` and README "What's misleading about the
  headline number."

### 2. Regex word-form brittleness on security-relevant language (found, then fixed this pass)
`sp_024683`: "Somebody appears to have been **hacking** my spotify..." was
NOT caught by the `account_security` escalation rule, which matched
"hacked" but not "hacking". Before this pass's fix, this example still
escalated correctly by coincidence (both the gold labeler and the
classifier landed on `other` for it, independently triggering escalation
via `unclassified_intent`) -- but the underlying bug was real: if this
message were ever classified into a real intent (e.g. `account_access`),
it would NOT have escalated via `account_security`. **Fixed this pass**:
broadened `hacked` to `hack\w*` (and `stolen` to `stole\w*|stolen`) in
`src/taxonomy.py::_ESCALATE_RULES`. Verified impact: zero change to any
`should_escalate` label in the golden set (still 127/250, since this
example already escalated via the coincidental path) -- only its
`escalation_reason` corrected from `unclassified_intent` to
`account_security` (the true cause). `baseline_simple`, which reuses the
first 3 `_ESCALATE_RULES` entries, gained one true-positive escalation from
this fix (recall 0.047->0.055). This was the single highest-risk latent bug
in the system (a real account-security complaint could have silently
failed to escalate under different classifier behavior) and was found by
manual spot-check, not by automated metrics -- regex-only escalation logic
remains fragile to other morphological variants not yet found.

### 3. Extractive reply generation's "groundedness" check is close to
tautological
`heuristic_judge` reports 250/250 (100%) grounded replies. This number is
close to meaningless: the extractive generator (used because no
`ANTHROPIC_API_KEY` was configured, see decision log item 8) *literally
copies* the top-retrieved historical reply, so word-overlap-with-evidence is
mechanically high by construction. It says nothing about whether the copied
historical reply actually fits the new customer's specific situation
(different account details, different song title, different error code).
This is disclosed rather than presented as a real quality signal -- see
`planning/11_LLM_JUDGE.md` for what an LLM judge would need to actually
assess.

### 4. Retrieval quality on very short/generic messages
Messages under ~5 words ("Need your help with something", "Backstage???")
retrieve historical pairs by superficial TF-IDF term overlap rather than
topical similarity, since there's not enough signal to disambiguate intent.
The `LOW_EVIDENCE_SIMILARITY = 0.12` threshold in
`src/reply_generation.py` catches the most extreme cases but a 3-8 word
message can still clear 0.12 cosine similarity on a common word like
"account" or "help" while matching a topically unrelated historical pair.

### 5. Golden-set `other` bucket conflates "truly ambiguous" with "rules
incomplete"
Manually reading the `other`-labeled examples (`planning/04_INTENT_TAXONOMY.md`)
shows a mix of: genuinely fragmentary/off-topic text (defensible as
`other`), and messages a human would clearly categorize (undercounted
`playback_technical`/`content_availability`/`billing_subscription` --
recall problem in the golden-label rules themselves, not just the system
under test). This inflates the apparent size of the `other` class and, per
failure mode #1, is the single largest driver of the reported escalation
error rates in both directions.

## What was attempted and what remains
Failure mode #1's highest-value fix (expanding training data rather than
further tuning the same 423-example pool) was attempted this session and
worked: false-auto-handle eliminated, false-escalation cut by more than
half. What's NOT attempted: independent (non-regex-derived) re-labeling to
determine how much of the remaining agreement is genuine language
understanding vs. convergence on the same labeling function used to build
both the training and evaluation labels -- flagged as the top item in "one
additional week" (see README).
