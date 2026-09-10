# Final Submission Audit

Status values are `COMPLETE` (executed, verified this session), `PARTIALLY
COMPLETE` (implemented but not fully executed, with a stated reason), or
`BLOCKED` (genuinely not possible with available resources).

| Requirement | Status | Evidence |
|---|---|---|
| Runnable repo | COMPLETE | `python scripts/run_all.py` runs clean, ~28s after data download; 24/24 tests pass (`pytest tests/`) |
| README <15 min reproduction | COMPLETE | Timed full `run_all.py` at 27.5s post-download; documented in README Quickstart |
| Golden set 150-250 | COMPLETE | 250 examples, `data/processed/golden_set.csv`, conversation-level split verified zero-overlap with dev/retrieval pools this session |
| Automated metrics | COMPLETE | `scripts/evaluate.py` -> `data/processed/eval_results.json`; denominators and class distributions verified in `planning/10_EVALUATION.md` |
| LLM judge | PARTIALLY COMPLETE | Implemented and mock/contract-tested (`tests/test_llm_paths_mocked.py`, 2 tests for judge specifically); not executed with a real API call -- no `ANTHROPIC_API_KEY` in this environment |
| Human agreement | BLOCKED | No annotators available. Annotation schema, rubric, sample-selection, and agreement-calculation workflow fully specified in `planning/12_HUMAN_VALIDATION.md`; zero fabricated ratings |
| Trivial baseline | COMPLETE | `src/baselines.py::baseline_trivial`, run on identical golden set, F1=0.00 escalation as expected (never escalates) |
| Simple baseline | COMPLETE | `src/baselines.py::baseline_simple`, deliberately distinct keyword logic from the golden-set labeler to avoid circularity |
| Failure analysis | COMPLETE | `planning/13_FAILURE_ANALYSIS.md`, 5 modes with quoted real examples; root-cause number re-verified precisely this pass (100% of escalation errors involve `other`, revising an earlier ~85% estimate) |
| Misleading headline number | COMPLETE | README "What's misleading about the headline number" section, 4 concrete caveats with evidence |
| One-week plan | COMPLETE | README "One additional week", 5 concrete items ranked by value |
| Decision log | COMPLETE | `planning/18_DECISION_LOG.md`, 17 real decisions including this session's bugfix and rejected experiment |
| Citations | COMPLETE | README "Citations" section: dataset mirror + all libraries used |

## What this session specifically changed
1. Found and fixed a real reproducibility bug: unpinned `scikit-learn`
   caused a committed model pickle to produce different metrics when
   loaded under a newer installed version. Pinned the version, retrained,
   and confirmed the originally-reported numbers were correct all along.
2. Ran an aggressive leakage audit (exact + normalized near-duplicate
   checks across all pools) -- found zero leakage, one benign generic-phrase
   near-duplicate across the golden set and retrieval corpus (not a leak:
   different conversations).
3. Precisely quantified the dominant failure mode: 100% (not ~85%) of all
   escalation errors involve the `other` intent class on at least one side.
4. Attempted a targeted fix (classifier confidence-override threshold),
   tuned it honestly on held-out dev data, checked it once against the
   frozen golden set, found it regressed the headline metric, and reverted
   it -- documented as a rejected experiment, not silently dropped.
5. Added a mocked-API test suite (`tests/test_llm_paths_mocked.py`, 4
   tests) proving the LLM reply-generation and LLM-judge code paths are
   wired correctly, without fabricating any live API results.
6. Added an unsupported-claims check to the reply-quality judge, disclosed
   as currently near-vacuous under extractive generation.
7. Re-verified baselines share the exact same evaluation set and metric
   functions as the full system (no hidden advantage).

## Genuine remaining blockers
- No `ANTHROPIC_API_KEY` in this environment -- LLM judge and LLM reply
  generation remain implemented-but-unexecuted.
- No human annotators available -- human-vs-LLM agreement remains
  not-possible-here, template-only.
- The `other`-class intent confusion (100% of escalation errors) remains
  unresolved after one rejected fix attempt; the next fix likely needs more
  labeled minority-class training data, which requires either human
  labeling effort or a different weak-supervision source, neither available
  here.
