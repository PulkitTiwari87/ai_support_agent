# Project Map

The full Graphify skill (knowledge-graph extraction with community
detection) was not run in this session -- it spins up a separate agent pass
and the marginal value over a direct, hand-maintained map was low relative
to its cost given the session's time budget. Below is the actual, current
component map (kept in sync manually; last updated after the taxonomy/
signoff bug fixes and re-evaluation).

```
data/raw/customer_support_twitter.parquet   (scripts/fetch_data.py)
        |
        v
scripts/process_data.py  -->  data/processed/{eval,dev}_pool.csv, retrieval_corpus.csv
        |                               |
        v                               v
scripts/select_brand.py         src/taxonomy.py (regex rules)
(brand choice, one-time)                |
        |                               v
        |                     scripts/build_golden_set.py --> data/processed/golden_set.csv (FROZEN)
        |                               |
        v                               v
src/retrieval.py              src/intent_classifier.py (trained on dev_pool)
(TF-IDF index over                      |
 retrieval_corpus)                      |
        |                               |
        +---------------+---------------+
                         v
                 src/pipeline.py  (intent -> retrieval -> generation -> escalation)
                         |
          +--------------+---------------+
          v                               v
  src/reply_generation.py          src/taxonomy.py::classify_escalation
  (extractive / LLM via
   ANTHROPIC_API_KEY)
                         |
                         v
              scripts/evaluate.py  --> data/processed/eval_results.json
              scripts/judge_replies.py --> data/processed/judge_results.json
                         |
                         v
              README.md (headline results, failure analysis, decision log)
```

Baselines (`src/baselines.py`) run against the same `golden_set.csv`
independently of the pipeline above, not through it.
