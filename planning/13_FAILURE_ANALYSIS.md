# Failure Analysis

Based on `data/processed/eval_results.json` and
`data/processed/system_predictions.csv` (system run on the 250-example
golden set). Regenerate with `python scripts/evaluate.py`.

## Top failure modes (real, inspected examples)

**Update (verification pass)**: re-measured precisely -- **100% of escalation
errors (56/56: 12 false-auto-handle + 44 false-escalation) involve `other`
on the gold-label side, the predicted side, or both** (more precise than
the earlier ~85% estimate). Escalation rate by gold intent is 1.0 for
`other` and near-0 for every real class, so this is close to a clean,
single-cause failure mode. An experiment to fix it (probability-override
threshold, tuned on held-out `dev_pool`) was tried and **rejected** after
it made every golden-set metric worse, including the headline
false-auto-handle rate (0.095->0.134) -- see
`planning/18_DECISION_LOG.md` #15 and `planning/10_EVALUATION.md`. The root
cause (65% `other` share in weak training labels) remains unresolved.

### 1. Nearly all escalation errors (both directions) trace to one root cause: intent collapsing to `other`
- **13 false-auto-handle cases** (system says handle, gold says escalate):
  every single one has `pred_intent` != `other` while `gold_intent == other`
  (escalation_reason=`unclassified_intent`). Example: `sp_019350`, "I can't
  log into my account. It says I use Facebook, but let me do it" -- the
  learned classifier correctly recognized this as `account_access`, but the
  golden-label rules couldn't match it, defaulted to `other`, and the
  `unclassified_intent -> escalate` fallback fired. This is *golden-label*
  brittleness showing up as an apparent system failure, not the system
  misunderstanding the message.
- **42 false-escalation cases** (system escalates, gold says don't): all 42
  have `pred_intent == other` while `gold_intent` is a real class (mostly
  `playback_technical` and `billing_subscription`). Example: `sp_035655`,
  "Every month, my app dumps 15 GB of downloaded playlists and forces me to
  download all of it again" -- clearly `playback_technical` to a human
  reader, but the TF-IDF+LogReg classifier's confidence collapsed to
  `other` on this noisy, profanity-adjacent, run-on sentence.
- **Root cause**: `other` is simultaneously (a) the taxonomy's genuine
  catch-all for ambiguous text and (b) the golden labeler's fallback when
  its regex coverage is incomplete and (c) the classifier's fallback when
  training signal is weak. All three collapse onto the same label, so
  errors compound in both directions around this one class.
- **Fix attempted**: broadened taxonomy regex coverage (see decision log
  item 13, `planning/18_DECISION_LOG.md`), which reduced but did not
  eliminate this. A real fix would need either more training data per class
  or merging `other` handling into confidence-based escalation rather than
  label-based (already partially done: `retrieval_confidence < 0.15` also
  triggers escalation independent of intent).

### 2. Regex word-form brittleness on security-relevant language
`sp_024683`: "Somebody appears to have been **hacking** my spotify..." was
NOT caught by the `account_security` escalation rule, which matched
"hacked" but not "hacking". This is the single highest-risk category of bug
in this system (a real account-security complaint that could go
un-escalated) and was found by manual spot-check, not by the automated
metrics. Regex-only escalation logic is fragile to morphological variants;
a production version should not rely solely on literal keyword matching for
safety-critical categories.

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

## What was NOT attempted
Given the size of failure mode #1, the highest-value next fix would be
retraining the intent classifier with the `other` class either downweighted
or split into sub-categories, or replacing golden-label fallback logic so
`other` isn't simultaneously an intent class and an escalation trigger. Not
attempted in this pass due to time -- flagged as the top item in "one
additional week" (see README).
