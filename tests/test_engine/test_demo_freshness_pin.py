"""Pin: demo provenance footer's verified-date tracks the manifest.

The dashboard's footer carries a single-line provenance ribbon:

    tests 996/996 · PCCP cohort 114 pairs · FHIR cohort 23 patients
    · flows 7/7 ship · verified <YYYY-MM-DD>

The `verified <YYYY-MM-DD>` cell is the only freshness signal a
hackathon judge sees at a glance. If it drifts by more than a day
behind `docs/reproducibility_manifest.json::dateCreated`, the
"every dashboard claim traces to a regenerable manifest" pitch is
silently broken: a judge could verify with `--check` and see the
manifest was regenerated yesterday while the demo claims the same
freshness today.

Iter 95 (T2 round 19) catches this drift class. Iter 94 manifest
was regenerated at 2026-05-04T14:00 UTC; the demo footer still said
`verified 2026-05-03`. Fix + pin so future iters that regenerate the
manifest also rotate the footer.

Discipline:
  - Demo's verified-date must equal the manifest's dateCreated date
    portion (YYYY-MM-DD).
  - The footer carries a stable id="provenance-verified" so future
    automation can rotate it programmatically.
  - The whole manifest cannot be older than 7 days; if it is, the
    "verified" claim is itself stale and the dashboard is no longer a
    trust signal.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEMO = _REPO_ROOT / "docs" / "demo.html"
_MANIFEST = _REPO_ROOT / "docs" / "reproducibility_manifest.json"

_VERIFIED_PATTERN = re.compile(
    r'id="provenance-verified"[^>]*>(\d{4}-\d{2}-\d{2})</span>'
)


def _manifest_date() -> str:
    """Return the YYYY-MM-DD of the manifest's dateCreated."""
    m = json.loads(_MANIFEST.read_text())
    iso = m["dateCreated"]
    return iso.split("T")[0]


def test_demo_provenance_footer_has_verified_id():
    """The provenance ribbon must expose a stable id for the
    `verified <date>` cell so the freshness check can find it."""
    html = _DEMO.read_text()
    m = _VERIFIED_PATTERN.search(html)
    assert m is not None, (
        "demo.html provenance footer must expose "
        "`<span id=\"provenance-verified\">YYYY-MM-DD</span>`. "
        "If you renamed it, update this test."
    )


def test_demo_verified_date_matches_manifest_date_created():
    """Footer's `verified <date>` must equal the manifest's
    dateCreated YYYY-MM-DD prefix."""
    html = _DEMO.read_text()
    m = _VERIFIED_PATTERN.search(html)
    assert m is not None
    displayed = m.group(1)
    expected = _manifest_date()
    assert displayed == expected, (
        f"demo.html footer says `verified {displayed}` but manifest "
        f"dateCreated is {expected}. Re-run "
        f"`python3 scripts/build_reproducibility_manifest.py` and "
        f"update the footer span (or use the same value if you just did)."
    )


def test_manifest_is_no_more_than_seven_days_old():
    """If the manifest hasn't been regenerated in a week, the entire
    audit-trail story is stale. Fail loudly."""
    m = json.loads(_MANIFEST.read_text())
    iso = m["dateCreated"]
    when = dt.datetime.fromisoformat(iso)
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    age = now - when
    assert age <= dt.timedelta(days=7), (
        f"docs/reproducibility_manifest.json is {age.days} days old "
        f"(generated {iso}). Re-run "
        f"`python3 scripts/build_reproducibility_manifest.py`."
    )


def test_no_stale_verified_dates_lurking():
    """A previous `verified <date>` value left in plain text without an
    id (or as a stripped duplicate) would confuse a judge reading the
    page. Block the iter-94 historical 2026-05-03 unless the manifest
    legitimately rolls back to it (which would be a regression).
    """
    html = _DEMO.read_text()
    expected_today = _manifest_date()
    historicals = ("verified</span> <span class=\"text-trust-700\">2026-05-03",)
    for stale in historicals:
        if expected_today == "2026-05-03":
            # If we ever genuinely roll back to this date (e.g. CI
            # restoring an earlier manifest), the historical stops being
            # historical.
            return
        assert stale not in html, (
            f"Stale verified-date snippet {stale!r} still in "
            f"docs/demo.html. Manifest is at {expected_today}; rotate."
        )
