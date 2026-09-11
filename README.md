# SpotifyCares Support Agent

A grounded AI support agent that classifies customer intent, retrieves historically similar resolutions, drafts a reply, and decides whether a case can be auto-handled or needs a human. Built and evaluated end-to-end on the real [Customer Support on Twitter](https://huggingface.co/datasets/TNE-AI/customer-support-on-twitter-conversation) dataset for `@SpotifyCares`.

Take-home for the Hiver SDE Intern assignment. Every number below is measured and reproducible with one command — where a component (LLM reply generation, LLM judge, human validation) is implemented but wasn't executed, that's stated explicitly rather than faked.

## At a Glance

| | |
|---|---|
| **Intent accuracy** | **92.4%** (frozen 250-example eval set) |
| **Macro-F1** | **89.4%** |
| **Escalation F1** | **95.1%** |
| **False-auto-handle rate** | **0.0%** (0/127) |
| **Stack** | Python, scikit-learn (TF-IDF + LinearSVC), pandas |
| **Tests** | 24/24 passing (`pytest tests/`) |
| **Setup** | `pip install -r requirements.txt && python scripts/run_all.py` — no API key required |

These are evaluation-set numbers, not a claim of production accuracy — see [Evaluation](#evaluation) for why.

## How It Works

```
customer message
       │
       ▼
intent classification      TF-IDF (word 1-2gram + char 3-7gram) + LinearSVC
       │
       ▼
historical retrieval       TF-IDF cosine similarity over 36,004 real
       │                   (customer_msg, support_reply) pairs
       ▼
grounded reply generation  extractive by default; optional Claude call
       │                   when ANTHROPIC_API_KEY is set
       ▼
escalation decision        rule-based: account security, billing disputes,
                            explicit human request, high distress, weak
                            retrieval evidence, unclassified intent
       │
       ▼
{intent, confidence, evidence, draft_reply, grounded, escalate, reason}
```

Code: [`src/taxonomy.py`](src/taxonomy.py), [`src/intent_classifier.py`](src/intent_classifier.py), [`src/retrieval.py`](src/retrieval.py), [`src/reply_generation.py`](src/reply_generation.py), [`src/pipeline.py`](src/pipeline.py).

**Intent taxonomy** (derived from reading ~150 sampled real support messages, not borrowed from an unrelated domain):

| Intent | Covers |
|---|---|
| `account_access` | Login, signup, password reset, account takeover/security |
| `billing_subscription` | Payment, plans, cancellation, refunds, regional pricing |
| `playback_technical` | App bugs, crashes, device compatibility, sync/download loss |
| `content_availability` | Missing/duplicate content, regional catalog/licensing |
| `feature_request_feedback` | Suggestions, praise/complaints with no actionable fix |
| `other` | Doesn't fit cleanly — the largest remaining source of error |

## Why These Design Choices

- **TF-IDF, not embeddings.** Simple, fast, interpretable, and a strong baseline for short, informal support messages at a 36K-document corpus size — an embedding index would add a dependency without addressing the actual bottleneck (see below).
- **Word + character n-grams.** Character n-grams (3-7) pick up spelling variation and informal word forms that word n-grams alone miss in support text.
- **LinearSVC over Logistic Regression.** On a controlled feature/hyperparameter grid on held-out data, LinearSVC scored 0.919 macro-F1 vs. 0.888 for the best tuned Logistic Regression config and 0.460 for MultinomialNB (rejected outright).
- **Historical retrieval for grounding.** A reply should reflect how the brand actually responded to similar issues before, not a paraphrase with no known resolution — so retrieval carries both the past customer message and the brand's real reply.
- **Escalation defaults to caution.** The rules are written so `other`-intent and low-retrieval-confidence cases escalate by default, since auto-handling a case that should reach a human is the worse failure mode.

## The Engineering Story

The interesting part of this project isn't the final config — it's how the classifier got there.

**Initial approach:** TF-IDF + Logistic Regression, trained on a 529-example dev pool.

**Diagnosis, not guessing.** Comparing TRAIN vs. held-out-DEV vs. GOLDEN showed a ~40-point macro-F1 gap between TRAIN (99.8%) and held-out DEV (60.6%) — textbook overfitting. Per-class F1 tracked per-class training-example count almost exactly (`feature_request_feedback`: 9 training examples → 0.00 held-out F1; `billing_subscription`: 71 examples → 0.92).

**Evidence before acting.** A learning curve (`scripts/learning_curve.py`) tested whether more real, same-dataset training data would help before committing to the change: held-out macro-F1 rose from 0.606 to 0.787 at just 10% of an expanded pool, continuing to ~0.84 by 50%. The expansion source was `retrieval_corpus.csv` — 36,004 real SpotifyCares messages already in the repo for retrieval grounding, never used for training, and verified conversation-level *and* exact-text disjoint from the golden set.

**Controlled search, on dev data only.** A small grid on the expanded pool (never touching golden) selected word(1,2)+char_wb(3,7) TF-IDF features and LinearSVC over Logistic Regression and Naive Bayes.

**Result:** intent accuracy 73.6% → 92.4%, macro-F1 59.1% → 89.4%, false-auto-handle 5.5% → 0.0% — all improved together, from a data and diagnosis fix, not a golden-set or taxonomy change. Final config: TF-IDF word(1,2)+char_wb(3,7), `min_df=2`, `sublinear_tf=True`, `LinearSVC(class_weight="balanced", C=1.0)`, trained on 36,533 real examples. Full trace: [`planning/18_DECISION_LOG.md`](planning/18_DECISION_LOG.md) #22–24, [`planning/10_EVALUATION.md`](planning/10_EVALUATION.md).

## Evaluation

All numbers below come from `python scripts/evaluate.py` against the frozen 250-example golden set (`data/processed/golden_set.csv`), never touched during model selection.

| System | Intent accuracy | Macro-F1 | Escalation P/R/F1 | False-auto-handle | False-escalation |
|---|---:|---:|---|---:|---:|
| Baseline: trivial (majority class) | 0.472 | 0.107 | 0.00 / 0.00 / 0.00 | 1.000 (127/127) | 0.000 |
| Baseline: simple keyword | 0.704 | 0.522 | 1.00 / 0.06 / 0.10 | 0.945 (120/127) | 0.000 |
| **Full system** | **0.924** | **0.894** | **0.91 / 1.00 / 0.95** | **0.000 (0/127)** | 0.106 (13/123) |

Weighted-F1: 0.921. Per-class F1: `billing_subscription` 0.949, `other` 0.944, `content_availability` 0.909, `playback_technical` 0.893, `feature_request_feedback` 0.857, `account_access` 0.811. Full breakdown: `data/processed/eval_results.json`.

### What's misleading about the headline number

1. **The golden labels are rule-derived, not independently human-labeled.** They come from the same regex taxonomy used to weakly-label the expanded training pool. A classifier that gets good at reproducing that regex will score well on any evaluation set built by the same regex family — this is a genuinely disjoint *data split* (the model never saw these conversations), but not an independent *labeling method*. An independently human-labeled golden set could show a smaller gap than 92.4%.
2. **`other`'s near-perfect recall (118/118)** partly reflects it being the largest, easiest-to-learn class (23,498 of 36,533 training examples) and the classifier converging on the same rule boundary that labels it — not necessarily deeper semantic understanding.
3. **Reply groundedness (100%) is close to tautological.** The extractive generator literally copies the top-retrieved historical reply, so word-overlap-with-evidence is mechanically high by construction.
4. **Single brand, single language, 2017-era tweets.** No claim of generalization to other brands, languages, or current Spotify UX.
5. **0.0% false-auto-handle is a measurement, not a validated safety rate.** It's a result on a 250-example, rule-derived, single-rater-spot-checked test set — a different production system, data era, or independently labeled test set could show a materially different number.

## Failure Analysis

| Failure mode | Evidence | Likely cause |
|---|---|---|
| `playback_technical` weakest well-represented class (F1 0.893) | 6/31 golden examples still predicted as `other` | Residual overlap with `other` even after data expansion |
| Escalation regex missed a word form | `sp_024683`: "hacking" wasn't matched by a rule that only caught "hacked" — found by manual spot-check, fixed this pass | Regex-only escalation logic is brittle to morphological variants not yet found |
| Reply groundedness metric near-tautological | `heuristic_judge` reports 250/250 grounded, but the extractive generator literally copies the retrieved reply | No independent judge (no LLM judge run) to check whether a copied reply actually fits the new case |
| Short/generic messages retrieve on superficial overlap | Messages under ~5 words ("Need your help with something") can clear the 0.12 similarity threshold on a common word | Not enough signal in very short queries to disambiguate topic |
| Remaining false-escalations all trace to `other` | All 13 remaining false-escalation cases have `pred_intent == other` against a real gold class (mostly `playback_technical`, `feature_request_feedback`) | Same root cause as row 1 — more data plus a non-regex label source is the likely next lever |

Full detail with quoted examples: [`planning/13_FAILURE_ANALYSIS.md`](planning/13_FAILURE_ANALYSIS.md).

## Quickstart

```bash
python -m venv .venv && source .venv/Scripts/activate   # .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
python scripts/run_all.py
```

`run_all.py` downloads a ~210MB public dataset mirror (no auth needed), builds the golden set, trains the classifier, builds the retrieval index, and runs the full evaluation plus both baselines. Metrics land in `data/processed/eval_results.json`. No API key required.

**Optional LLM path:** set `export ANTHROPIC_API_KEY=...` before running to use Claude for reply generation and the LLM judge instead of the extractive/heuristic fallbacks (`src/reply_generation.py`, `src/judge.py`). This path is implemented and covered by mocked-API tests (`tests/test_llm_paths_mocked.py`) but was **not executed live** in this submission environment — no key was available. Reported reply-quality results use the heuristic judge, labeled as such.

Run tests: `pytest tests/` (24 tests: taxonomy, reply generation, data-leakage checks, mocked LLM contracts).

## Project Structure

```text
src/                  Core pipeline: taxonomy, classifier, retrieval, reply generation, judge, orchestration
scripts/              Data fetch/process, golden-set build, training/diagnosis experiments, evaluation, run_all
tests/                Unit + leakage tests (pytest)
planning/             Decision log, taxonomy/brand rationale, evaluation and failure-analysis writeups
data/                 Raw (gitignored) and processed artifacts (golden set, model, eval results)
```

## Limitations

- Golden-set labels are rule-derived and single-rater spot-checked, not independently multi-annotator human-labeled.
- No live LLM judge or human validation was executed (implemented and mock-tested, blocked on missing API key / annotators).
- Reply-groundedness metric is near-tautological under the current extractive generator.
- Single brand, single language, 2017-era Twitter data — no generalization claim beyond this dataset.
- `playback_technical` remains the weakest well-represented class (F1 0.893) even after data expansion.

## What I Did Not Build

- No production infrastructure, auth, queues, or orchestration — a single-process Python pipeline.
- No embeddings/vector DB — TF-IDF is sufficient at this corpus size and keeps the quickstart path fully offline.
- No fine-tuned/custom LLM — generation is either extractive templating or a single Claude call per message.
- No multi-turn conversation state — each message is scored independently.
- No executed human-vs-LLM validation study — the annotation template and workflow exist (`planning/12_HUMAN_VALIDATION.md`) but no annotators or API key were available in this environment.

These were scope decisions for a take-home, not oversights.

## What I Would Build Next

1. **Independently (non-rule-derived) re-label a sample of the golden set** — the highest-value next step, to measure how much of the 92.4% reflects real language understanding vs. convergence on the same labeling function used for training.
2. Run the real LLM judge and a genuine multi-rater human validation pass with a configured API key and actual annotators.
3. Apply the same data-expansion approach to retrieval and reply-quality evaluation, which weren't touched this pass.
4. Reduce `playback_technical`'s residual overlap with `other` via targeted error analysis rather than more undifferentiated data.
5. Replace regex escalation triggers with stemmed/lemmatized or embedding-based matching to close classes of word-form bugs like the "hacking" vs. "hacked" one found and fixed here.

## Tech Stack

Python · pandas · scikit-learn (`TfidfVectorizer`, `LinearSVC`) · numpy · Anthropic Python SDK (optional path) · pytest

## Reproducibility

- `scripts/fetch_data.py` — downloads the dataset (public HF mirror, no auth)
- `scripts/process_data.py` — parses, dedupes, splits into eval/dev/retrieval pools (conversation-level, leakage-tested)
- `scripts/build_golden_set.py` — labels the frozen 250-example golden set
- `src/intent_classifier.py` — trains the classifier on the expanded pool (36,533 examples)
- `src/retrieval.py` — builds the TF-IDF retrieval index
- `scripts/diagnose_classifier.py`, `scripts/diagnose_labels.py`, `scripts/learning_curve.py` — the diagnosis and evidence behind the classifier improvement pass
- `scripts/experiment_grid.py`, `scripts/experiment_svc_tuning.py` — the controlled feature/hyperparameter search
- `scripts/evaluate.py` — runs the system and both baselines, writes metrics
- `scripts/judge_replies.py` — reply-quality judge (heuristic fallback without an API key)
- `scripts/run_all.py` — runs all of the above in order under `sys.executable`, so it always uses the launching interpreter's pinned dependencies rather than whatever `python` happens to resolve to on `PATH` (a real bug found and fixed during this project)

No secrets are committed. `data/raw/` is gitignored and regenerated by `scripts/fetch_data.py`.

## Assignment Coverage

| Requirement | Status |
|---|---|
| Intent classification | Implemented and evaluated — `src/intent_classifier.py` |
| Historically grounded reply | Implemented — `src/retrieval.py` + `src/reply_generation.py` |
| Auto-handle vs. escalation decision | Implemented — `src/taxonomy.py::classify_escalation` |
| Golden evaluation set (150-250 examples) | Implemented — 250 examples, `data/processed/golden_set.csv` |
| Automated evaluation + baselines | Implemented — `scripts/evaluate.py`, two baselines |
| LLM-as-judge | Implemented, mock-tested, **not executed live** (no API key) |
| Failure analysis | Implemented — see above and `planning/13_FAILURE_ANALYSIS.md` |
| Reproducibility (one command) | Implemented — `python scripts/run_all.py` |

Full requirement-by-requirement trace with evidence: [`REQUIREMENT_TRACEABILITY.md`](REQUIREMENT_TRACEABILITY.md).

## Citations

- Dataset: Twitter customer-support conversations, originally published on Kaggle as `thoughtvector/customer-support-on-twitter`, accessed via the public HuggingFace mirror [`TNE-AI/customer-support-on-twitter-conversation`](https://huggingface.co/datasets/TNE-AI/customer-support-on-twitter-conversation) (no Kaggle auth was available in this environment).
- Libraries: pandas, pyarrow, scikit-learn, numpy, anthropic (Python SDK), pytest — see `requirements.txt`.
