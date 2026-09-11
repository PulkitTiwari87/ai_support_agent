# Final Submission Audit

Status values are `COMPLETE` (executed, verified this session), `PARTIALLY
COMPLETE` (implemented but not fully executed, with a stated reason), or
`BLOCKED` (genuinely not possible with available resources).

| Requirement | Status | Evidence |
|---|---|---|
| Runnable repo | COMPLETE | `python scripts/run_all.py` runs clean, ~21-30s after data download; 24/24 tests pass (`pytest tests/`) |
| README <15 min reproduction | COMPLETE | Timed at ~21-30s post-download across four sessions |
| Golden set 150-250 | COMPLETE | 250 examples, `data/processed/golden_set.csv`, conversation-level split, zero leakage re-verified this session (including exact-text overlap check against the now-training-source `retrieval_corpus`); file itself unmodified this session (`git diff` clean) |
| Automated metrics | COMPLETE | `scripts/evaluate.py` -> `data/processed/eval_results.json`; now includes weighted-F1 and per-class F1, added this session |
| LLM judge | PARTIALLY COMPLETE | Implemented and mock/contract-tested (`tests/test_llm_paths_mocked.py`); not executed with a real API call -- no `ANTHROPIC_API_KEY` in this environment (re-checked this session, still absent) |
| Human agreement | BLOCKED | No annotators available. Annotation schema, rubric, sample-selection, and agreement-calculation workflow fully specified in `planning/12_HUMAN_VALIDATION.md`; zero fabricated ratings |
| Trivial baseline | COMPLETE | `src/baselines.py::baseline_trivial`, run on identical golden set |
| Simple baseline | COMPLETE | `src/baselines.py::baseline_simple`, deliberately distinct keyword logic from the golden-set labeler to avoid circularity |
| Classifier diagnosis before changing anything | COMPLETE | `scripts/diagnose_classifier.py` (TRAIN/held-out-DEV/GOLDEN split), `scripts/diagnose_labels.py` (label-quality/`other`-class check) -- both run and documented before any model change this session |
| Data-expansion justification | COMPLETE | `scripts/learning_curve.py` shows held-out DEV macro-F1 0.606->0.787 at 10% of the expanded pool, continuing to plateau by ~50% -- satisfies the assignment's explicit "prove it before expanding" requirement |
| Controlled experiments (not golden-tuned) | COMPLETE | `scripts/experiment_grid.py`, `scripts/experiment_svc_tuning.py` -- all tuning on held-out DEV; golden touched exactly once, after model selection |
| Failure analysis | COMPLETE | `planning/13_FAILURE_ANALYSIS.md`, updated to reflect the largely-resolved `other`-class failure mode (false-auto-handle 7->0, false-escalation 48->13) |
| Misleading headline number | COMPLETE | README "What's misleading about the headline number" section, 5 concrete caveats, updated for the new numbers -- including a new caveat specific to training/eval label-method overlap |
| One-week plan | COMPLETE | README "One additional week", re-ranked now that the `other`-class fix succeeded |
| Decision log | COMPLETE | `planning/18_DECISION_LOG.md`, 24 real decisions across four sessions |
| Citations | COMPLETE | README "Citations" section: dataset mirror + all libraries used; no new dependencies added this session |

## Final verified headline numbers (this session, `scripts/run_all.py`, expanded training pool)
| | intent acc | intent macro-F1 | intent weighted-F1 | escalation P/R/F1 | false-auto-handle | false-escalation |
|---|---|---|---|---|---|---|
| trivial | 0.472 | 0.107 | -- | 0.00/0.00/0.00 | 1.000 (127/127) | 0.000 (0/123) |
| simple | 0.704 | 0.522 | -- | 1.00/0.055/0.105 | 0.945 (120/127) | 0.000 (0/123) |
| **system** | **0.924** | **0.894** | **0.921** | 0.907/1.000/0.951 | **0.000 (0/127)** | 0.106 (13/123) |

## What this session specifically changed
1. **Diagnosed before changing anything** (per explicit instruction):
   built a real TRAIN/held-out-DEV/GOLDEN split (previous model had never
   been evaluated this way) and found textbook overfitting -- 99.8% train
   accuracy vs. 60.6% held-out-dev macro-F1 -- with per-class F1 tracking
   training-example count almost exactly.
2. **Justified data expansion with a learning curve before doing it**:
   `retrieval_corpus.csv` (36,004 real SpotifyCares messages, previously
   used only for retrieval) showed held-out DEV macro-F1 rising sharply
   with more of it, satisfying the assignment's own "prove scarcity before
   expanding" bar. Re-verified leakage-safety (conversation-ID and
   exact-text) against `golden_set` before using it for training.
3. **Ran a small, sequential, controlled experiment grid on TRAIN/DEV
   only** (never golden): feature config, then LR hyperparameters, then
   classifier family. Selected TF-IDF word(1,2)+char_wb(3,7) + LinearSVC.
4. **Evaluated golden exactly once**, after model selection was already
   locked in. Every metric improved together (accuracy, macro-F1,
   escalation P/R/F1, false-auto-handle, false-escalation) -- unlike the
   previous session's char-n-gram change, which traded macro-F1 for
   false-auto-handle.
5. **Treated the large jump as suspicious until checked, not as a win to
   report immediately**: re-verified zero leakage (conversation-ID and
   exact-text) before and after training, confirmed the golden-set
   confusion matrix is non-degenerate (real residual errors persist), and
   confirmed the improvement pattern matches the diagnosis (biggest gains
   in the previously-smallest classes).
6. Added weighted-F1 and per-class F1 to `scripts/evaluate.py`'s output
   (a reporting completeness fix on already-generated predictions, not a
   re-tuning pass).
7. Re-verified all leakage checks, baseline fairness, and existing tests
   still hold after all changes (24/24 passing throughout).
8. `golden_set.csv` was never read for tuning, never modified, and never
   trained on -- confirmed via `git diff` showing zero changes to the file
   across the entire session.

## Genuine remaining blockers
- No `ANTHROPIC_API_KEY` in this environment -- LLM judge and LLM reply
  generation remain implemented, mock-tested, and unexecuted live.
- No human annotators available -- human-vs-LLM agreement remains
  not-possible-here, template-only (`planning/12_HUMAN_VALIDATION.md`).
- **The training/evaluation label-provenance overlap is now the primary
  remaining validity question**, not a data-scarcity problem. The
  classifier is trained on 36,533 examples labeled by the same regex
  family that built the golden set's labels. The very high golden
  agreement (92.4% accuracy) is real (no leakage, no memorization,
  genuine held-out generalization on real conversations the model never
  saw), but it partly measures "how well does the classifier reproduce the
  regex labeler at scale," not independent ground truth. Resolving this
  needs an independently human-labeled sample, which needs annotators --
  the same resource blocker as human validation above.
- `playback_technical` remains the weakest well-represented class (golden
  F1 0.893) even after data expansion -- not a resource blocker, just an
  unresolved residual worth targeted error analysis in a future pass.
