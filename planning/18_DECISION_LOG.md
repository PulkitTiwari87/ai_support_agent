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
