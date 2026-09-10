# Golden Evaluation Set

## Construction
- 250 examples, one `(customer_msg, support_reply)` pair per unique
  `conversation_id`, sampled from the deduped SpotifyCares pair pool
  (`scripts/process_data.py`).
- **Leakage control**: split is at `conversation_id` level, not pair level,
  across three disjoint pools -- `eval_pool` (250 conversations),
  `dev_pool` (529 conversations, used only to tune taxonomy regex and train
  the intent classifier), and `retrieval_corpus` (36,004 pairs, the
  grounding index). No conversation appears in more than one pool.
- Frozen at `data/processed/golden_set.csv`. Do not re-label based on how
  the system performs.

## Labeling method
Intent and escalation ground truth come from `src/taxonomy.py` rules (same
module, disjoint data from what trains the intent classifier). This is
**not** independent multi-rater human annotation -- there was no annotation
tooling/budget available in this environment. To catch rule errors:

**Spot-check**: read 40 randomly sampled golden-set rows by hand
(`random_state=7`). Found and fixed one real bug: messages describing
account takeover ("Someone has just accessed my account and changed the
account email to one I do not recognize") were not triggering the
`account_security` escalation rule because it only matched the literal word
"hacked"/"stolen". Broadened the pattern
(`src/taxonomy.py::_ESCALATE_RULES`) and regenerated the golden set. Also
found the punctuation-based distress rule (`!!`/`??`) over-triggered on
ordinary Twitter typing style; tightened to 3+ repeated marks.

After the fix, 39/40 spot-checked labels were judged correct by re-reading
them against the taxonomy definitions; the one remaining disagreement
(`sp_027687`, "stop playing me... put Reputation up" mislabeled
`playback_technical` instead of `content_availability` due to a false
keyword match on "stop playing") is left as a known, documented rule
limitation rather than hand-patched, since the golden set must stay rule-
generated + spot-checked, not manually edited case-by-case.

## Honest limitation
This is a single-rater, rule-derived golden set, not a human-annotated one.
Treat golden-set accuracy numbers as measuring agreement with a documented,
inspectable rule system -- not ground truth in an absolute sense. See
`planning/12_HUMAN_VALIDATION.md` for what a real human validation pass
would add, and why it wasn't run here (no annotators available).

## Class balance (n=250)
```
other                       119
billing_subscription         58
playback_technical           30
account_access               21
feature_request_feedback     16
content_availability          6
```
`should_escalate=True`: 128/250 (51.2%).
