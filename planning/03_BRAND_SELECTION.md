# Brand Selection

## Method
Scored 10 candidate brands present in the dataset (`scripts/select_brand.py`)
on: conversation volume, % of conversations with an actual support reply
(resolution proxy), average turns, and a vocabulary-diversity ratio
(unique/total words over a 2000-conversation sample). Results:
`data/processed/brand_candidates.csv`.

| brand | n_conversations | % w/ reply | avg turns | vocab diversity |
|---|---|---|---|---|
| AmazonHelp | 81,092 | 99.8% | 4.39 | 0.132 |
| AppleSupport | 76,639 | 99.7% | 2.84 | 0.087 |
| Uber_Support | 41,185 | 99.7% | 2.95 | 0.090 |
| **SpotifyCares** | **27,910** | **99.7%** | **3.18** | **0.090** |
| comcastcares | 23,442 | 99.5% | 2.86 | 0.088 |
| TMobileHelp | 22,322 | 99.5% | 3.24 | 0.102 |
| XboxSupport | 12,841 | 98.8% | 3.71 | 0.079 |
| AskPlayStation | 12,237 | 99.7% | 3.22 | 0.089 |
| AskPayPal | 9,238 | 99.4% | 2.69 | 0.112 |
| VerizonSupport | 8,126 | 99.6% | 4.99 | 0.081 |

## Decision: SpotifyCares

AmazonHelp scores highest on raw volume and diversity, but that diversity is
a liability here: Amazon support spans orders, devices, Prime Video, Whole
Foods, drivers, returns -- unrelated domains that would force either a large
taxonomy (violates "small, defensible") or a taxonomy so coarse it loses
meaning. Uber_Support has the same problem (rider issues vs. driver issues
vs. Eats).

SpotifyCares is a single digital product. Reading ~150 sample messages
(`planning/04_INTENT_TAXONOMY.md`) showed the issue space is genuinely
bounded: account access, billing/subscriptions, playback bugs, content
availability, and feature feedback cover the overwhelming majority of
traffic. 27,910 conversations is enough for a 36K-pair retrieval corpus plus
a 250-example frozen golden set with no overlap, while staying processable
within this session's time budget (AmazonHelp's 81K conversations would
roughly triple parsing/retrieval-index build time for no taxonomy benefit).

## What was NOT selected and why
- **AppleSupport, Uber_Support** (highest volume after Amazon): too broad
  (hardware + software + services for Apple; rides + Eats for Uber).
- **comcastcares, TMobileHelp, VerizonSupport** (telecom): plausible
  alternates, similar profile to Spotify but with more regulatory/billing
  complexity (contracts, equipment) that would inflate scope.
- **XboxSupport, AskPlayStation**: smaller, gaming-specific, good candidates
  but more device-troubleshooting-heavy with less clean "billing" signal.
- **AskPayPal**: financial services -- explicitly avoided giving investment/
  financial advice is a hard constraint on this agent; a payments brand
  raises exactly those issues.
