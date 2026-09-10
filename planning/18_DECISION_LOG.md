# Decision Log

Real decisions made while building this system, in order. Each links to the
evidence that drove it.

1. **Dataset source: HuggingFace mirror, not Kaggle API.** The Kaggle
   `thoughtvector/customer-support-on-twitter` dataset requires an
   authenticated `kaggle.json`, which wasn't available in this environment.
   Used the public, unauthenticated mirror
   `TNE-AI/customer-support-on-twitter-conversation` on HuggingFace instead
   (794,335 conversations, same underlying Twitter support data, pre-grouped
   by company). No account/token needed, fully reproducible via
   `scripts/fetch_data.py`.

2. **Brand: SpotifyCares.** Scored 10 candidate brands on volume, resolution
   rate, and vocabulary diversity (`scripts/select_brand.py`,
   `data/processed/brand_candidates.csv`). Rejected AmazonHelp/Uber_Support
   despite higher volume+diversity scores because they span too many
   unrelated sub-domains (orders, devices, drivers, Prime Video, rides,
   Eats...) for a *small, defensible* taxonomy. SpotifyCares is a single
   product with natural, bounded issue categories. See
   `planning/03_BRAND_SELECTION.md`.

3. **Small taxonomy (6 intents), derived from reading real data, not
   Banking77.** Banking77 is for retail banking and doesn't map to a music
   streaming product. Read ~150 sampled dev_pool messages and converged on:
   account_access, billing_subscription, playback_technical,
   content_availability, feature_request_feedback, other. See
   `planning/04_INTENT_TAXONOMY.md`.

4. **Golden set labeled by rules + engineer spot-check, not multi-rater
   human annotation.** No crowd-annotation budget/tooling in this
   environment. Labels come from `src/taxonomy.py` regex rules, spot-checked
   by hand against 40 random examples (found and fixed one real bug: account
   takeover messages weren't triggering escalation). This is a genuine
   limitation, disclosed in the README/report, not hidden. See
   `planning/05_GOLDEN_SET.md`.

5. **Train/eval split at conversation_id, not pair, level.** Initially split
   individual (customer_msg, support_reply) pairs randomly across
   eval/dev/retrieval pools; caught during review that multiple pairs from
   the same conversation could land in different pools, leaking context.
   Fixed to split on conversation_id first.

6. **Intent classifier: TF-IDF + Logistic Regression trained on dev_pool,
   not the same rules used to label golden set.** Reusing the taxonomy
   regex directly as "the system" would make intent accuracy on the
   golden set trivially ~100% (the rules would be grading themselves). A
   separately-trained model on a disjoint split gives a real number.

7. **Retrieval: TF-IDF cosine similarity, not embeddings.** No API key
   configured for embedding calls; sentence-transformers would add a
   heavyweight dependency for a 36K-document corpus where TF-IDF is
   sufficient and fully offline. Documented as a "next week" upgrade path.

8. **Reply generation: extractive by default, LLM path implemented but
   unused.** `src/reply_generation.py` calls Claude when
   `ANTHROPIC_API_KEY` is set; none was configured in this environment, so
   all reported results use the extractive fallback (adapts the top
   retrieved historical reply). This is disclosed, not disguised as LLM
   generation.

9. **Escalation: threshold + keyword rules, not a trained classifier.**
   Escalation policy is naturally rule/threshold-based in real support
   systems (security keywords, billing disputes, low retrieval confidence,
   unclassified intent) and this phase doesn't need a learned model for it.

10. **LLM-as-judge and human-vs-LLM validation: implemented but not run.**
    `src/judge.py` has a working LLM judge gated on `ANTHROPIC_API_KEY`.
    Without a key, results use `heuristic_judge` (word-overlap + length
    checks) which is reported as a heuristic proxy, not an LLM or human
    judge. No human ratings were fabricated; see `planning/12_HUMAN_VALIDATION.md`
    for the annotation template that would be used.

11. **Baseline 2 ("simple") uses a different keyword rule set than the
    golden-set labeler.** Reusing the labeling regex as a baseline would
    make that baseline circularly ~100% accurate on intent. Baseline 2 uses
    a deliberately blunter single-keyword-per-intent lookup instead.

12. **Headline metric: false auto-handle rate, not raw accuracy.** Per the
    assignment's stated priority (auto-handling something that should have
    escalated is the worst failure), the headline result is
    false-auto-handle-rate reduction (100% trivial -> 95% simple baseline ->
    10% full system), with escalation F1 and intent macro-F1 as supporting
    numbers, not the primary claim.

13. **Punctuation-based distress rule tightened from 2+ to 3+ repeated
    `!`/`?`.** Spot-check showed 2+ triggered too often on ordinary Twitter
    typing style (false escalations); 3+ is closer to genuine emphasis
    while still erring toward escalation over silent failure.

14. **Pinned `scikit-learn==1.9.1` in requirements.txt after finding a real
    reproducibility bug.** Re-running `scripts/evaluate.py` against the
    previously-committed model pickle (trained under sklearn 1.3.2) with
    sklearn 1.9.1 installed produced silently different metrics (intent
    accuracy 0.68 vs. the originally reported 0.728) with an
    `InconsistentVersionWarning`. Retraining fresh under 1.9.1 reproduced
    the original 0.728 exactly, confirming the drift was a stale-pickle
    artifact, not a real regression -- but an unpinned dependency let it
    happen silently. See `planning/10_EVALUATION.md`.

15. **Rejected an "other-class override threshold" fix for the intent
    classifier after measuring it on the golden set.** `other` is 65% of
    `dev_pool` weak training labels, causing the classifier to over-predict
    it. Tuned a probability-override rule on a held-out 80/20 split of
    `dev_pool` (macro-F1 0.595->0.622 there,
    `scripts/tune_other_threshold.py`), but checking it once against the
    frozen golden set showed it made every headline metric worse,
    including false-auto-handle rate (0.095->0.134). Reverted per the
    regression policy rather than keeping a change that only helped a
    proxy metric. Root cause of the `other`-class confusion remains
    unresolved -- flagged as the top "one more week" item.

16. **Added an unsupported-claims heuristic check to `src/judge.py`,
    disclosed as near-vacuous under the current extractive generator.**
    The existing word-overlap grounding score is close to tautological for
    extractive replies (the reply IS the evidence). Added a check for
    numbers/amounts/durations in the reply not present in the evidence
    text as a more specific (if still heuristic) signal -- it returns 0/250
    on the current extractive-only results by construction, and is
    documented as only becoming meaningful once the LLM paraphrasing path
    actually runs.

17. **Verified LLM-backed code paths (reply generation, judge) with mocked
    API calls instead of skipping testing them.** No `ANTHROPIC_API_KEY`
    was available to make real calls, but `tests/test_llm_paths_mocked.py`
    proves the request/response contract and fallback-on-error behavior are
    correct by substituting a fake `anthropic` module -- so supplying a
    real key is the only remaining step to make these paths live, and that
    claim is now tested rather than assumed.

18. **Kept a word+char n-gram classifier change despite a lower macro-F1,
    because it improves the assignment's stated priority metric.** Tuned
    on a held-out `dev_pool` split (macro-F1 0.595->0.606 there,
    `scripts/experiment_char_ngrams.py`), then checked once against the
    frozen golden set: accuracy 0.728->0.736, macro-F1 0.620->0.591,
    false-auto-handle rate 0.095->0.055 (12->7 cases), false-escalation
    rate 0.358->0.390 (44->48 cases). Unlike decision #15 (which regressed
    every metric and was rejected), this trades secondary metrics for an
    improvement in exactly the metric the assignment weighs highest
    (auto-handling something that needed escalation is the worst failure).
    Kept, with the tradeoff stated in numbers everywhere it's reported, not
    presented as an unqualified win.

19. **Found and fixed a second, more serious reproducibility bug:
    `scripts/run_all.py` used a bare `"python"` instead of
    `sys.executable`.** In this environment, bare `python` on PATH resolves
    to a completely different, unrelated system Python installation with an
    old, unpinned scikit-learn (1.3.2) -- not the project's `.venv` that
    `requirements.txt` pins to 1.9.1. This meant the documented one-command
    reproduction path silently ignored the pinned dependencies on every
    run, regardless of decision #14's version pin. Fixed by using
    `sys.executable` for every subprocess step. This also means the
    previous session's "byte-identical reproduction" verification was
    comparing the wrong things (manual `.venv` invocations against each
    other, not the actual documented `run_all.py` command) -- corrected
    and re-verified this pass with zero version warnings.

20. **Fixed the "hacking" vs. "hacked" escalation regex gap found in
    failure analysis, and verified it doesn't silently change the frozen
    golden set's grading.** Broadened `hacked` to `hack\w*` in
    `src/taxonomy.py::_ESCALATE_RULES` (a genuine safety-relevant bug: a
    real account-security complaint using "hacking" instead of "hacked"
    wasn't triggering the security escalation rule). Since this rule is
    shared between the golden labeler and the system, regenerated the
    golden set and diffed it against the previous frozen version before
    accepting the change: zero `should_escalate` labels changed (still
    127/250) -- only one row's `escalation_reason` corrected from a
    coincidental `unclassified_intent` fallback to the true
    `account_security` cause. `baseline_simple`'s escalation recall moved
    slightly (0.047->0.055) since it reuses the same rule list. This is the
    kind of fix the "don't cheat the golden set" rule requires disclosing
    explicitly, which is why it's logged here with the before/after diff
    rather than silently applied.

21. **Did not attempt further `other`-class fixes after two experiments.**
    Confirmed via per-class training counts that `feature_request_feedback`
    (11 examples) and `content_availability` (19 examples) in `dev_pool`
    are too small for any TF-IDF-based classifier to learn robustly
    regardless of features/thresholds -- verified by held-out validation
    where `feature_request_feedback` recall stayed at 0 under both the
    word-only and word+char-n-gram classifiers (2 validation examples in
    that class, held-out). Concluded this is a data-scarcity problem
    requiring more labeled examples, not further architecture tuning on the
    same ~11-19-example classes -- a valid engineering conclusion per the
    assignment's own framing (section 34: "the bottleneck is data labeling,
    not model sophistication" is an acceptable answer).
