"""Pin: pharmacology flag table covers every cache drug + URL hygiene.

Iter 100 (T1 round 20 substituted, eval-rigor): the new
`docs/pharmacology_flags.json` is shipped + manifest-tracked
(iter 96 + iter 97), but no test enforces:

  1. **Coverage**: every drug name appearing in
     `docs/openevidence_cache.json` (drug_a or drug_b) must have a
     matching entry in pharmacology_flags. Future cache growth that
     adds a new drug pair without flagging the new drug would slip
     past every existing test, breaking the curated-table claim
     ("every drug in our cohort traces to FDA pharmacology classes").

  2. **URL hygiene**: every flagged drug must have ≥ 1 evidence URL,
     all HTTPS. Same pattern as `test_cache_evidence_urls.py` but
     against the flag table.

  3. **Schema-version pin**: schema_version present + flag_keys is
     a non-empty list (iter-96 baseline schema).

This is the same drift class the iter-94 cache shape pins catch for
the cache itself, applied to its companion artifact.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"
_FLAGS = _REPO_ROOT / "docs" / "pharmacology_flags.json"


def _cache_drug_set():
    cache = json.loads(_CACHE.read_text())
    drugs = set()
    for it in cache:
        drugs.add(it["drug_a"].lower().strip())
        drugs.add(it["drug_b"].lower().strip())
    return drugs


def _flags_doc():
    return json.loads(_FLAGS.read_text())


def test_every_cache_drug_has_pharmacology_flag_entry():
    """Every drug name in the cache must have an entry in
    `pharmacology_flags.json::drugs`. Empty-flag entries are fine
    (some drugs genuinely have no pharmacology-class flags), but the
    drug must be CATALOGUED so a future flag addition has a home."""
    cache_drugs = _cache_drug_set()
    flagged = set(_flags_doc()["drugs"].keys())
    missing = cache_drugs - flagged
    assert not missing, (
        f"{len(missing)} cache drugs missing from pharmacology_flags: "
        f"{sorted(missing)[:8]}{'...' if len(missing) > 8 else ''}. "
        f"Add each one to docs/pharmacology_flags.json with at least "
        f"`flags: []` and an FDA-label evidence URL."
    )


# iter-270 ratchet — bound the orphan-flag-drug set (drugs flagged
# but not yet referenced by any cache entry). These are intentional
# pre-staging entries for future cohort growth (e.g., tranylcypromine
# passed iter-264 v8 pre-flight at +42.95 margin and is reserved for
# a future T5 growth iter). The pin lets a small bounded set live but
# fails the gate if orphans drift past the documented allowlist —
# preventing dead-code accumulation.
_ALLOWED_ORPHAN_DRUGS = frozenset({
    # NSAID class — pre-staged for future ketorolac+lithium or
    # ketorolac+ACE-inhibitor cohort growth. FDA Toradol § 4 flags
    # multiple high-risk combinations.
    "ketorolac",
    # MAOI class — pre-staged for tranylcypromine+tramadol, +sertraline,
    # +meperidine etc. cohort growth (tranylcypromine+tramadol passed
    # iter-264 v8 pre-flight at +42.95 Q16.16 margin, reserved for a
    # future T5 iter). FDA Parnate § 4 explicit serotonergic contras.
    "tranylcypromine",
})


def test_orphan_flag_drugs_bounded_to_allowlist():
    """Drugs in `pharmacology_flags.json` that have no cache
    reference must be in the explicit `_ALLOWED_ORPHAN_DRUGS`
    allowlist. Intentional pre-staging entries are documented;
    accidental dead-code orphans fail the gate.

    To add a new orphan (e.g., post a v8 pre-flight that selected
    a candidate but didn't yet land in cache), extend
    `_ALLOWED_ORPHAN_DRUGS` deliberately with the rationale.
    """
    cache_drugs = _cache_drug_set()
    flagged = set(_flags_doc()["drugs"].keys())
    orphans = flagged - cache_drugs
    unauthorized = orphans - _ALLOWED_ORPHAN_DRUGS
    assert not unauthorized, (
        f"{len(unauthorized)} flag-drug entries have no cache reference "
        f"and are NOT in the documented allowlist: {sorted(unauthorized)}. "
        f"Either remove the orphan from pharmacology_flags.json, OR "
        f"add it to `_ALLOWED_ORPHAN_DRUGS` in this pin file with a "
        f"comment explaining the pre-staging rationale (typically a "
        f"v8 pre-flight pass)."
    )
    # Soft floor: don't let the allowlist itself grow without bound.
    # If 10+ orphans accumulate, that's a signal the cohort growth
    # cadence is lagging the flag-table additions.
    assert len(orphans) <= 5, (
        f"{len(orphans)} orphan flag-drugs exceeds the soft-floor of 5. "
        f"Either run a T5 cohort-growth iter to land the pre-staged "
        f"candidates, or trim the flag table to match live cohort scope."
    )


def test_every_flagged_drug_has_at_least_one_evidence_url():
    """The curated-table pitch is 'every flag traces to a published
    FDA label or peer-reviewed reference'. An entry with no URLs
    breaks that pitch."""
    flags = _flags_doc()["drugs"]
    no_url = [n for n, e in flags.items() if not e.get("evidence_urls")]
    assert not no_url, (
        f"{len(no_url)} flagged drugs have no evidence URL: "
        f"{no_url[:5]}{'...' if len(no_url) > 5 else ''}. "
        f"Each entry must cite at least one FDA label or peer-review reference."
    )


def test_every_evidence_url_is_https():
    """Plain-HTTP URLs in a regulated-domain audit chain are a
    silent integrity hole. Same pin as test_cache_evidence_urls.py
    applied to the flag table."""
    flags = _flags_doc()["drugs"]
    bad = []
    for name, e in flags.items():
        for u in e.get("evidence_urls", []):
            if not u.startswith("https://"):
                bad.append((name, u))
    assert not bad, (
        f"{len(bad)} pharmacology flag URLs are not HTTPS. "
        f"First: {bad[0]}"
    )


def test_schema_version_and_flag_keys_present():
    """Iter-96 baseline schema: schema_version set + flag_keys non-empty
    list. Iter-140 raised the floor 12 -> 25 (the 13 baseline flags +
    12 new flags added by the 100% explanation-coverage closure)."""
    doc = _flags_doc()
    assert doc.get("schema_version"), (
        "pharmacology_flags.json must carry a schema_version field"
    )
    flag_keys = doc.get("flag_keys", [])
    assert isinstance(flag_keys, list) and len(flag_keys) >= 25, (
        f"flag_keys must be a non-empty list of >= 25 entries; "
        f"got {len(flag_keys) if isinstance(flag_keys, list) else type(flag_keys).__name__}. "
        f"Iter-140 added 12 new flag classes for 100%-coverage closure; "
        f"removing any of them silently regresses the curated-table claim."
    )


def test_every_flag_key_is_canonical_snake_case_is_prefix():
    """Every flag in `flag_keys` MUST start with `is_` and be
    snake_case. Violating this breaks downstream encoders that pattern-
    match on the prefix (e.g. `_pair_derived_flags` checks for
    `has_pair("is_cyp3a4_strong_inhibitor", "is_cyp3a4_substrate")`)."""
    flag_keys = _flags_doc()["flag_keys"]
    snake = re.compile(r"^is_[a-z][a-z0-9_]*$")
    bad = [k for k in flag_keys if not snake.match(k)]
    assert not bad, (
        f"non-canonical flag key(s): {bad}. "
        f"All keys must match `is_[a-z][a-z0-9_]*`."
    )


def test_drug_names_in_flag_table_are_canonicalised_lowercase():
    """Drug-name keys must be lowercase + whitespace-collapsed so
    `_flag_bits(drug_name)` lookup is consistent with the cache's
    `drug_a` / `drug_b` canonicalisation."""
    flags = _flags_doc()["drugs"]
    bad = [n for n in flags.keys() if n != " ".join(n.lower().split())]
    assert not bad, (
        f"non-canonical drug-name keys: {bad[:5]}. "
        f"Each key must equal `' '.join(name.lower().split())`."
    )


# ─── iter-141 (T1 round 28) flag-key + rule canonical-name pins ─────────

# The 13 baseline flag classes (iter-96 era).
_BASELINE_FLAG_KEYS = frozenset({
    "is_cyp3a4_strong_inhibitor",
    "is_cyp3a4_substrate",
    "is_cyp2c9_inhibitor",
    "is_cyp2d6_inhibitor",
    "is_p_gp_inhibitor",
    "is_p_gp_substrate",
    "is_oatp1b1_inhibitor",
    "is_statin",
    "is_anticoagulant",
    "is_maoi",
    "is_serotonergic",
    "is_nsaid",
    "is_pde5_inhibitor",
})

# The 12 iter-140 additions (the 100%-coverage closure flags).
_ITER140_FLAG_KEYS = frozenset({
    "is_iodinated_contrast",
    "is_metformin",
    "is_renal_state",
    "is_cyp1a2_inhibitor",
    "is_cyp1a2_substrate",
    "is_xanthine_oxidase_inhibitor",
    "is_thiopurine",
    "is_folate_antagonist",
    "is_tetracycline",
    "is_retinoid",
    "is_ace_inhibitor",
    "is_neprilysin_inhibitor",
})


def test_baseline_flag_keys_present():
    """The 13 baseline flag classes (iter-96 era) must remain in
    `flag_keys`. Renaming any of them silently breaks
    `_pair_derived_flags` rule lookups (rule 0 calls
    `has_pair('is_cyp3a4_strong_inhibitor', 'is_cyp3a4_substrate')`)."""
    flag_keys = set(_flags_doc()["flag_keys"])
    missing = _BASELINE_FLAG_KEYS - flag_keys
    assert not missing, (
        f"{len(missing)} baseline flag class(es) missing from "
        f"pharmacology_flags.json::flag_keys: {sorted(missing)}. "
        f"These names are referenced by `_pair_derived_flags` rules 0-5 "
        f"and a rename would silently regress 6 of the 13 rules."
    )


def test_iter140_flag_keys_present():
    """The 12 iter-140 flag classes (added for 100%-coverage closure)
    must remain in `flag_keys`. Renaming any silently regresses the
    100%-coverage gate to 71.4% (the iter-105 baseline)."""
    flag_keys = set(_flags_doc()["flag_keys"])
    missing = _ITER140_FLAG_KEYS - flag_keys
    assert not missing, (
        f"{len(missing)} iter-140 flag class(es) missing from "
        f"pharmacology_flags.json::flag_keys: {sorted(missing)}. "
        f"These flags close the 8-mechanism documented-gap class for "
        f"100% explanation coverage (iter-140); removing any of them "
        f"silently regresses coverage to 71.4%."
    )


# Canonical example: each pair-derived rule index → (drug_a, drug_b)
# from the live cache that MUST fire that rule. Catches both
#   (a) a rule going dead (its named example stops firing it) and
#   (b) a flag-table edit that removes the example's flag.
_RULE_CANONICAL_EXAMPLES = (
    # (idx, drug_a,        drug_b,           rule_name)
    (0,  "clarithromycin", "simvastatin",    "cyp3a4_inhib_substrate"),
    (1,  "gemfibrozil",    "simvastatin",    "oatp1b1_inhib_statin"),
    (2,  "clarithromycin", "simvastatin",    "p_gp_inhib_substrate"),
    # rule 3 (cyp2c9_inhib_anticoag): not yet exercised by any contra
    # cache entry — placeholder for future warfarin+strong-CYP2C9
    # additions. Skipped from canonical-example pin until populated.
    (4,  "phenelzine",     "sertraline",     "maoi_serotonergic"),
    (5,  "isosorbide mononitrate", "sildenafil", "pde5_nitrate"),
    (6,  "iodine",         "metformin",      "iodinated_contrast_metformin"),
    (7,  "ciprofloxacin",  "tizanidine",     "cyp1a2_inhib_substrate"),
    (8,  "allopurinol",    "azathioprine",   "xo_thiopurine"),
    (9,  "methotrexate",   "trimethoprim-sulfamethoxazole", "folate_antagonist_pair"),
    (10, "doxycycline",    "isotretinoin",   "tetracycline_retinoid"),
    (11, "lisinopril",     "sacubitril",     "ace_neprilysin"),
    (12, "metformin",      "renal impairment", "metformin_renal"),
)


def test_each_rule_fires_on_canonical_example_pair():
    """Iter-141 (T1 round 28): for every pair-derived rule that has a
    cache contra example, assert the canonical example pair fires
    that rule. Catches the silent-regression class:
      (a) a rule disappears (the example pair stops firing it) — the
          100%-coverage gate could still pass if another rule covers
          the example, but the rule is dead;
      (b) a flag-table edit removes the example drug's flag — the rule
          still works in code but loses its only known cache example.

    Same canonical-example pin pattern as iter-84 arch-mind 9-rule
    canonical set + iter-94 cache shape.
    """
    import importlib
    import sys
    sys.path.insert(0, str(_REPO_ROOT / "retrain_runpod"))
    import train_bitnet_v3_atc
    importlib.reload(train_bitnet_v3_atc)
    pair_derived = train_bitnet_v3_atc._pair_derived_flags

    failures = []
    for idx, da, db, name in _RULE_CANONICAL_EXAMPLES:
        flags = pair_derived(da, db)
        assert len(flags) == 13, (
            f"_pair_derived_flags returned {len(flags)} bits "
            f"(iter-141 expects 13)."
        )
        if flags[idx] != 1:
            fired = [i for i, f in enumerate(flags) if f]
            failures.append(
                f"rule {idx} ({name}) — canonical example "
                f"{da!r} + {db!r} did NOT fire rule {idx}; "
                f"actually fired rules: {fired}"
            )
    assert not failures, (
        "Canonical-example regression(s):\n  "
        + "\n  ".join(failures)
        + "\nEither restore the missing flag(s) on the example drugs "
          "or update _RULE_CANONICAL_EXAMPLES with a new representative "
          "pair."
    )
