# Human vs. LLM Judge Validation

## Status: NOT RUN (no annotators, no LLM API credentials in this environment)

This assignment asks for a human-vs-LLM-judge agreement study. Both legs
require resources not available in this session:
- **LLM judge**: `src/judge.py::llm_judge` requires `ANTHROPIC_API_KEY`,
  which is not configured. All reported reply-quality numbers instead use
  `heuristic_judge` (word-overlap + length), explicitly labeled as a
  heuristic proxy, not an LLM judge, in every artifact that reports it
  (`data/processed/judge_results.json`, `planning/13_FAILURE_ANALYSIS.md`).
- **Human judge**: no annotator pool or annotation UI available. Producing
  "human ratings" here would mean a single engineer (not independent,
  not blind, not multi-rater) rating outputs of a system they built --
  which is not what "human validation" means and would misrepresent the
  result if reported as one.

Per the no-fake-results rule: **no synthetic or single-rater results are
presented as human evaluation anywhere in this repo.**

## What IS ready to run
`data/processed/system_predictions.csv` has 250 system replies with
customer messages. To run the real study with credentials/annotators:

1. `export ANTHROPIC_API_KEY=...` then `python scripts/judge_replies.py`
   -- runs the real LLM judge (rubric: correctness, relevance, grounding,
   completeness, tone, hallucination -- `src/judge.py::RUBRIC_DIMENSIONS`)
   over all 250 replies and writes per-dimension scores.
2. Sample ~40-50 of those replies into a spreadsheet with columns
   `[pair_id, customer_msg, draft_reply, evidence, rubric dimensions]`
   for 2+ independent human raters, blind to the LLM judge's scores.
3. Compute agreement with e.g. Cohen's/weighted kappa or Spearman
   correlation per dimension between human and LLM scores.

This is left as a template, not executed, to avoid fabricating agreement
numbers that don't exist.
