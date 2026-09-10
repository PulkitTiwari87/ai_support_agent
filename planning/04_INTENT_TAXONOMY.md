# Intent Taxonomy (SpotifyCares)

Derived by reading ~150 sampled `data/processed/dev_pool.csv` messages
(never the frozen eval pool), not assumed from an external taxonomy like
Banking77 (retail banking doesn't transfer to a music-streaming product).

## Frozen taxonomy (6 intents)

| Intent | Description | Example |
|---|---|---|
| `account_access` | Login, signup, password reset, account takeover/security | "Someone accessed my account and changed the email" |
| `billing_subscription` | Payment, premium plans, family/student plans, cancellation, refunds, regional pricing | "I was charged twice for premium this month" |
| `playback_technical` | App bugs, crashes, device/platform compatibility, sync/download loss | "App keeps crashing on my Sonos after the update" |
| `content_availability` | Missing/duplicate songs, albums, artists; regional catalog/licensing | "Why isn't [album] on Spotify?" |
| `feature_request_feedback` | Suggestions, editorial requests, praise/complaints with no actionable fix | "Wish you could organize playlists into folders" |
| `other` | Doesn't fit cleanly (fallback; see limitations) | fragments, off-topic replies, ambiguous one-liners |

Implementation: `src/taxonomy.py::classify_intent` (regex rules, tuned on
dev_pool only).

## Known limitation
The `other` bucket is large (~48% of golden-set rule labels) because Twitter
support threads contain a lot of genuinely ambiguous or fragmentary text
(mid-conversation replies like "My operating system is Windows 10",
usernames, single words). This is a real property of the data, not purely a
rule-coverage gap -- see `planning/13_FAILURE_ANALYSIS.md` for the
quantified breakdown of what's rule brittleness vs. genuinely ambiguous
input.
