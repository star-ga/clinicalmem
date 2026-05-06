# OpenEvidence Cached Fixture Set

## Why this cache exists

ClinicalMem uses OpenEvidence (Layer 2 of the six-layer drug-interaction
pipeline) for clinically authoritative, evidence-cited responses.  An
academic license was requested on 2026-05-02; delivery takes up to five
business days.

Without a live key the `_openevidence_check_interactions()` function
previously returned an empty list silently.  Multi-LLM evaluators flagged
this as a "stub breaks live demo reliability" gap because the dashboard demo
and integration tests could not show realistic OpenEvidence responses.

The cached fixture set closes that gap: the engine falls back to
`engine/openevidence_cache.py` whenever `OPENEVIDENCE_API_KEY` is absent
or a live API call fails, and returns `DrugInteraction` objects in exactly
the same format as the live path.


## How to switch to live mode

Set the environment variable before starting the service:

```bash
export OPENEVIDENCE_API_KEY=your_key_here
```

Once the variable is set the engine sends every query to the live
OpenEvidence API and never consults the local cache.  No code changes
are required.


## Provenance of cached entries

All clinical summaries in `docs/openevidence_cache.json` are derived from
long-established, publicly documented drug-interaction mechanisms.  Sources
include:

- **FDA prescribing information** (accessdata.fda.gov) — authoritative US
  drug labelling with explicit drug-drug interaction sections.
- **NIH/NCBI PubMed and PMC** — peer-reviewed clinical pharmacology studies
  cited in standard references.
- **ACR Manual on Contrast Media** (acr.org) — consensus guidelines for
  iodinated contrast + metformin management.
- **Landmark randomised trials** — RALES (spironolactone + ACE inhibitor
  for heart failure), Dentali et al. (aspirin + warfarin bleeding risk meta-
  analysis), Masclee et al. (NSAID + anticoagulant GI bleed meta-analysis).

These are widely known clinical pharmacology facts reproduced in every major
drug reference (Lexicomp, Micromedex, UpToDate, British National Formulary).
The summaries are **not proprietary OpenEvidence content** — they are
synthesised from public literature for demonstration purposes only.

Every entry is marked `"source": "CACHED"` with `"retrieved_at": "2026-05-02"`
so audit reviewers can identify cache-origin responses immediately.


## Covered pairs

| Drug A | Drug B | Severity |
|--------|--------|----------|
| aspirin | warfarin | serious |
| ibuprofen | warfarin | serious |
| naproxen | warfarin | serious |
| nsaid | warfarin | serious |
| contrast dye | metformin | contraindicated |
| lisinopril | potassium | moderate |
| lisinopril | spironolactone | moderate |
| metoprolol | verapamil | serious |
| amiodarone | simvastatin | serious |
| fluoxetine | tramadol | serious |
| ciprofloxacin | tizanidine | contraindicated |
| methotrexate | trimethoprim | serious |
| amoxicillin | penicillin | moderate |
| iodine | metformin | contraindicated |
| atorvastatin | grapefruit | moderate |

Lookup is case-insensitive and argument-order-independent (lex-sorted key).


## Audit chain: CACHED vs LIVE

Every `DrugInteraction` object produced from the cache has:

1. `description` prefixed with `[CACHED 2026-05-02] ...` — visible in the
   dashboard and in any downstream audit report.
2. A structured `INFO` log line from `engine.clinical_scoring`:

   ```
   INFO openevidence_cache: returning cached entry pair=<a>+<b> severity=<s> source=CACHED
   ```

When the live API is active, responses carry no `[CACHED]` prefix and the
log line is instead emitted by the live path (`OpenEvidence detected N
additional interactions`).

Reviewers can grep the application log for `source=CACHED` to enumerate
every cached response served in a given session.


## JSON schema (dashboard-loadable)

`docs/openevidence_cache.json` is a top-level JSON array.  Each element:

```json
{
  "drug_a": "warfarin",
  "drug_b": "ibuprofen",
  "drug_pair_canonical": ["ibuprofen", "warfarin"],
  "severity": "serious",
  "clinical_summary": "...",
  "evidence_urls": ["https://..."],
  "retrieved_at": "2026-05-02",
  "source": "CACHED"
}
```

`drug_pair_canonical` is always lex-sorted.  `drug_a` / `drug_b` preserve
the natural clinical ordering for display purposes.  The demo dashboard can
`fetch("docs/openevidence_cache.json")` and render evidence cards directly.
