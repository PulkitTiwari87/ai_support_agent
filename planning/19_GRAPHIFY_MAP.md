# Project Map

The full Graphify skill (knowledge-graph extraction with community
detection) was not run in this session -- it spins up a separate agent pass
and the marginal value over a direct, hand-maintained map was low relative
to its cost given the session's time budget. This is disclosed, not
presented as if the tool ran, across all three sessions. Below is the
actual, current component map (kept in sync manually; last updated after
the run_all.py interpreter bugfix, the kept char-n-gram experiment, and the
account-security regex fix).

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
src/retrieval.py              src/intent_classifier.py (word+char n-gram
(TF-IDF index over               TF-IDF, trained on dev_pool)
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

```
scripts/tune_other_threshold.py   (experiment: rejected, see decision log #15)
scripts/experiment_char_ngrams.py (experiment: kept, see decision log #18)
tests/test_llm_paths_mocked.py    (proves LLM code paths correct w/o a real API key)
tests/test_data_pipeline.py       (leakage checks)
planning/10_EVALUATION.md         (verified metrics, denominators, both reproducibility bugs found+fixed)
planning/FINAL_SUBMISSION_AUDIT.md (requirement-by-requirement status)
```

`scripts/run_all.py` orchestrates the full left column above via
`sys.executable` subprocess calls (fixed this session -- previously used a
bare `"python"` that silently resolved to an unrelated system interpreter
with mismatched dependencies, see decision log #19).
