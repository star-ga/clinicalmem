"""Pin: `bitnet_classified` DEBUG log record from
`engine.bitnet_classifier.classify()` is PHI-safe.

**Iter-311 T1 round-66 — runtime regression test for the iter-309 PHI fix.**

Iter-309 caught a real PHI leak in the LIVE classifier's
`bitnet_classified` DEBUG event:

  ```
  logger.debug(
      "bitnet_classified",
      extra={
          "drug_a": a_canonical,    # ← raw drug name, fires on EVERY classify()
          "drug_b": b_canonical,    # ← raw drug name
          ...
      },
  )
  ```

Live since the iter-72-era classifier landing (~280 cron iters). The
iter-240 PHI-discipline pin's var-name list at line 65 (`_PHI_RISKY_VAR_NAMES`)
DID include `drug_a` / `drug_b` for positional `%s` patterns, but the
iter-240 forbidden-extras-keys list at line 289 was missing them — so
the source-scan pin would have passed even with the live leak.

Iter-309 atomic fix:
  1. Replaced raw drug_a/drug_b extras with module-inline 16-char
     SHA-256 pair_hash_prefix (canonical lex-sorted form, mirroring
     iter-291 / iter-284 / iter-279 PHI discipline pattern).
  2. Extended iter-240 forbidden-extras-keys list with drug_a, drug_b,
     drug, drug_pair, medication, med — closes the gap so source
     scans catch this regression class.

This pin is the **runtime regression test** — same shape as iter-296
which is the runtime regression test for the iter-291 fix. Where the
iter-240 source-scan pin (post iter-309 extension) catches the leaking
PATTERN at compile time, this pin invokes the LIVE classify() code
path with sentinel drug names and asserts the actual log record's
extras shape end-to-end.

Without this pin a future refactor could:
  - Revert pair_hash_prefix to drug_a / drug_b (caught by iter-240
    source scan AND by this runtime test)
  - Add a NEW field that DOES carry raw drug names (caught only by
    this runtime test — source scan is structurally unaware of new
    field names not in the forbidden list)
  - Change the canonical-pair hash semantics so swap(a,b) produces
    different hashes (caught only by this runtime test)
  - Stop emitting the bitnet_classified event entirely (caught only
    by this runtime test)

The iter-309 fix runs on EVERY classify() call (which the live
production engine does ~50× per patient handoff) so any regression
would leak ~50 PHI events per request.

28th cross-pin family in the discipline lineage.
"""
from __future__ import annotations

import hashlib
import logging
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _live_weights():
    """Load the live v8 engine weights for use in classify()."""
    from engine.bitnet_classifier import load_weights  # noqa: PLC0415
    return load_weights()


def test_classify_log_emits_pair_hash_prefix_not_raw_names(caplog):
    """A `classify()` call emits exactly one DEBUG event named
    'bitnet_classified', whose extras carry pair_hash_prefix (16-char
    SHA-256 lowercase hex) and NO raw drug names anywhere.
    """
    from engine.bitnet_classifier import classify  # noqa: PLC0415

    # Use sentinel drug names so we can scrub for PHI leakage in the
    # record. These won't be in the v8 ATC flag table, so the encoder
    # falls back to BLAKE2b hash trits with all-0 flag bits — fine for
    # the classify() forward pass to run deterministically.
    sentinel_a = "secret-classify-sentinel-aaa-marker"
    sentinel_b = "secret-classify-sentinel-bbb-marker"

    with caplog.at_level(logging.DEBUG, logger="engine.bitnet_classifier"):
        result = classify(sentinel_a, sentinel_b, _live_weights())

    assert result is not None, "classify() must return a BitNetResult"
    matched = [
        r for r in caplog.records
        if r.name == "engine.bitnet_classifier"
        and r.message == "bitnet_classified"
    ]
    assert matched, (
        "classify() must emit a DEBUG 'bitnet_classified' event. "
        "iter-309 added pair_hash_prefix to the extras; if this event "
        "stops firing the runtime regression test for iter-309 fails."
    )
    rec = matched[0]

    # Required PHI-safe fields (post iter-309).
    assert rec.levelno == logging.DEBUG
    assert hasattr(rec, "pair_hash_prefix"), (
        "iter-309 fix: bitnet_classified extras MUST carry "
        "pair_hash_prefix (16-char SHA-256 lowercase hex). If absent, "
        "either the iter-309 fix was reverted or the field was renamed."
    )
    assert re.fullmatch(r"[0-9a-f]{16}", rec.pair_hash_prefix), (
        f"pair_hash_prefix must be 16-char lowercase hex; got "
        f"{rec.pair_hash_prefix!r}"
    )

    # Forbidden PHI fields — these MUST NOT appear in the record extras.
    forbidden_attrs = ("drug_a", "drug_b", "drug", "drug_pair",
                       "medication", "med", "drug_name")
    for attr in forbidden_attrs:
        assert not hasattr(rec, attr) or getattr(rec, attr, None) is None, (
            f"PHI LEAK: bitnet_classified extras carry {attr!r}. "
            f"iter-309 fix reverted? Use pair_hash_prefix instead."
        )

    # Source-of-truth scalar fields (kept post iter-309).
    assert hasattr(rec, "severity")
    assert hasattr(rec, "severity_name")
    assert hasattr(rec, "repro_hash")
    assert hasattr(rec, "weights_id")
    assert hasattr(rec, "deterministic_match")

    # PHI scrub: raw sentinels must NEVER appear in the record's
    # message, args, or any string-valued extra.
    full_text = rec.getMessage() + " " + " ".join(
        f"{k}={v}" for k, v in vars(rec).items()
        if not k.startswith("_") and isinstance(v, (str, int, bool, float))
    )
    assert sentinel_a not in full_text, (
        f"PHI LEAK: raw drug name {sentinel_a!r} found in "
        f"bitnet_classified log record. iter-309 fix reverted?\n"
        f"Record: {full_text!r}"
    )
    assert sentinel_b not in full_text, (
        f"PHI LEAK: raw drug name {sentinel_b!r} found in "
        f"bitnet_classified log record. iter-309 fix reverted?\n"
        f"Record: {full_text!r}"
    )

    # The record's positional args MUST be empty (no positional %s
    # interpolation that would carry drug names).
    assert not rec.args, (
        f"bitnet_classified record carries positional args {rec.args!r}. "
        f"This is the iter-291 / iter-299 leak pattern. Use "
        f"extra={{}} only."
    )


def test_classify_log_pair_hash_is_canonical_lex_sorted(caplog):
    """The pair_hash_prefix MUST be computed from the canonical
    lex-sorted pair so swap(a, b) produces the same hash. iter-291
    / iter-284 / iter-279 same discipline."""
    from engine.bitnet_classifier import classify  # noqa: PLC0415

    drug_x = "zzz-classify-pair-late"
    drug_y = "aaa-classify-pair-early"

    # Compute expected hash from canonical lex-sorted form. The
    # iter-309 inline form is `f"{a_canonical}+{b_canonical}"`
    # where (a_canonical, b_canonical) = sorted((drug_a, drug_b)).
    a_canon, b_canon = sorted((drug_x, drug_y))
    expected_hash = hashlib.sha256(
        f"{a_canon}+{b_canon}".encode("utf-8")
    ).hexdigest()[:16]

    weights = _live_weights()
    hashes_seen = set()
    for a, b in [(drug_x, drug_y), (drug_y, drug_x)]:
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="engine.bitnet_classifier"):
            classify(a, b, weights)
        recs = [
            r for r in caplog.records
            if r.name == "engine.bitnet_classifier"
            and r.message == "bitnet_classified"
        ]
        assert recs, f"classify({a!r}, {b!r}) emitted no bitnet_classified event"
        rec = recs[-1]
        hashes_seen.add(rec.pair_hash_prefix)

    assert len(hashes_seen) == 1, (
        f"swap(a, b) produced {len(hashes_seen)} distinct pair_hash_prefix "
        f"values: {hashes_seen!r}. The iter-309 fix MUST canonicalise the "
        f"pair (lex sort) before hashing so identity is symmetric."
    )
    actual_hash = hashes_seen.pop()
    assert actual_hash == expected_hash, (
        f"pair_hash_prefix mismatch: live={actual_hash!r} "
        f"expected={expected_hash!r}. The iter-309 inline hash form is "
        f"`hashlib.sha256(f'{{a_canonical}}+{{b_canonical}}'.encode())."
        f"hexdigest()[:16]` over the lex-sorted pair."
    )


def test_classify_source_carries_iter309_pair_hash_prefix_token():
    """Source-level guard: `engine/bitnet_classifier.py` MUST contain
    the `pair_hash_prefix` token in the bitnet_classified extras
    block. Catches accidental refactor that drops the iter-309 fix
    even if the runtime tests above somehow pass under different
    inputs."""
    src = (_REPO_ROOT / "engine" / "bitnet_classifier.py").read_text()
    assert '"pair_hash_prefix"' in src, (
        "engine/bitnet_classifier.py must reference 'pair_hash_prefix' "
        "in the bitnet_classified extras (iter-309 PHI fix). If the "
        "field was renamed, update this pin to track the new name; "
        "but DO NOT replace it with raw drug_a / drug_b."
    )
    # And must NOT carry "drug_a" / "drug_b" as extras keys.
    # (Variables `a_canonical` / `b_canonical` are still allowed —
    # they're internal canonical form, not log payload.)
    bitnet_classified_block = re.search(
        r'logger\.debug\(\s*\n\s*"bitnet_classified",\s*\n\s*extra=\{(.*?)\}',
        src,
        re.DOTALL,
    )
    assert bitnet_classified_block is not None, (
        "Could not find logger.debug('bitnet_classified', extra={...}) "
        "block in engine/bitnet_classifier.py. iter-309 fix removed?"
    )
    extras_block = bitnet_classified_block.group(1)
    forbidden_in_extras = ('"drug_a"', '"drug_b"', '"drug"',
                            '"drug_pair"', '"medication"', '"med"')
    leaked = [k for k in forbidden_in_extras if k in extras_block]
    assert not leaked, (
        f"bitnet_classified extras contain forbidden PHI keys: {leaked!r}. "
        f"iter-309 fix reverted? Use pair_hash_prefix instead."
    )
