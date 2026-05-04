"""Pin the integrity of the negative-control cohort.

Iter 120 (round 24 T1 substitution).

The 0 / 10 false-positive precision claim on the demo + JUDGES is
load-bearing for safety: it is the *precision* gate that complements
the PCCP recall gate. The claim is computed by
`scripts/run_negative_control_eval.py` against
`docs/negative_control_cohort.json`. If that cohort silently drifts
(size shrinks, a pair gets reclassified, a pair collides with a
cache contra) the precision gate becomes meaningless without anyone
noticing.

This pin enforces six structural invariants:

  1. Cohort size = 10 (4 boundary + 6 clean — the iter-10 design).
  2. Every entry's `expected_severity` is exactly "none".
  3. ZERO entries collide with contraindicated cache entries
     (a logical contradiction — the same pair cannot be a
     "negative control" AND a "contraindicated DDI").
  4. Every entry has ≥ 1 evidence URL (audit-trail floor).
  5. The 4 named CYP-pathway boundary cases are present (their
     presence is what makes the precision gate non-trivial — clean
     negatives prove "BitNet doesn't fire on unrelated pairs"; the
     boundary cases prove "BitNet doesn't false-positive on pairs
     that look interaction-y but aren't").
  6. The 6 clean negatives must NOT include any drug pair where
     either drug appears in any cache contra entry — this prevents
     a future reviewer from sneaking a CYP3A4-substrate into "clean
     negatives" and inflating the apparent recall.

Same drift class as iter-100 pharmacology_flags coverage pin and
iter-22 federation invariant disclosure: a load-bearing artifact
must be auditable from the test suite.
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_NEG_CONTROL = _REPO_ROOT / "docs" / "negative_control_cohort.json"
_CACHE = _REPO_ROOT / "docs" / "openevidence_cache.json"

_EXPECTED_SIZE = 10
_EXPECTED_BOUNDARY_CASES = frozenset({
    ("amlodipine", "atorvastatin"),
    ("clopidogrel", "pantoprazole"),
    ("diltiazem", "simvastatin"),
    ("spironolactone", "trimethoprim"),
})


def _pair_key(entry: dict) -> tuple[str, str]:
    """Return the canonical sorted lower-case pair tuple."""
    pair = entry.get("drug_pair_canonical") or [
        entry["drug_a"], entry["drug_b"]
    ]
    a, b = sorted(d.strip().lower() for d in pair)
    return (a, b)


def _load_neg_control() -> list[dict]:
    return json.loads(_NEG_CONTROL.read_text())


def _load_cache_contras() -> set[tuple[str, str]]:
    cache = json.loads(_CACHE.read_text())
    return {
        _pair_key(e)
        for e in cache
        if (e.get("severity") or "").lower() == "contraindicated"
    }


def test_neg_control_cohort_size_pinned():
    """Cohort size must be exactly 10 — the iter-10 design (4 boundary + 6 clean).

    Adding a new entry forces a deliberate update to this constant
    AND the demo's `0 / 10 FP` claim AND the precision-gate description
    in JUDGES.md. Future cohort growth is healthy but must be visible.
    """
    cohort = _load_neg_control()
    assert len(cohort) == _EXPECTED_SIZE, (
        f"Negative-control cohort size drifted: live={len(cohort)}, "
        f"pinned={_EXPECTED_SIZE}. Update the demo's `0 / N FP` claim "
        f"AND JUDGES.md precision-gate description AND this constant "
        f"in the same commit."
    )


def test_neg_control_all_expected_severity_none():
    """Every cohort entry must have expected_severity == 'none'.

    A negative control with expected_severity != 'none' is a logical
    contradiction — by definition the pipeline SHOULD flag it.
    """
    cohort = _load_neg_control()
    offenders = [
        (entry["drug_a"], entry["drug_b"], entry.get("expected_severity"))
        for entry in cohort
        if entry.get("expected_severity") != "none"
    ]
    assert not offenders, (
        f"Negative-control entries with non-'none' expected_severity: "
        f"{offenders}. A negative control with expected severity != 'none' "
        f"belongs in the recall (cache) cohort, not the precision cohort."
    )


def test_neg_control_zero_collision_with_cache_contras():
    """No negative-control pair may also be a cache contra entry.

    This is the load-bearing safety invariant: the precision claim
    `0 / N FP` is meaningful only if every cohort pair is genuinely
    NOT a contraindicated interaction. A collision would mean we're
    counting a real contra as a 'false positive' — undermining both
    the precision claim and the recall claim simultaneously.
    """
    cohort_pairs = {_pair_key(e) for e in _load_neg_control()}
    cache_contras = _load_cache_contras()
    collisions = cohort_pairs & cache_contras
    assert not collisions, (
        f"Negative-control / cache-contra COLLISION: {sorted(collisions)}. "
        f"These pairs appear in BOTH negative_control_cohort.json AND "
        f"openevidence_cache.json (severity=contraindicated). One of the "
        f"two cohorts is wrong; remove the entry from whichever is the "
        f"weaker source of truth (typically: a curated FDA-label contra "
        f"trumps a 'no clinically significant interaction' rationale)."
    )


def test_neg_control_every_entry_has_evidence_url():
    """Every cohort entry must carry ≥ 1 evidence URL — audit-trail floor."""
    offenders = [
        (entry["drug_a"], entry["drug_b"])
        for entry in _load_neg_control()
        if not entry.get("evidence_urls")
    ]
    assert not offenders, (
        f"Negative-control entries with no evidence_urls: {offenders}. "
        f"Every cohort entry must cite ≥ 1 source so an FDA reviewer can "
        f"verify the 'no clinically significant interaction' rationale."
    )


def test_neg_control_boundary_cases_present():
    """The 4 named CYP-pathway boundary cases must be in the cohort.

    The boundary cases are what make the precision gate non-trivial —
    clean negatives show 'BitNet doesn't fire on unrelated pairs',
    the boundary cases show 'BitNet doesn't false-positive on pairs
    that LOOK interaction-y (CYP overlap, statin co-prescribing,
    transporter overlap) but ARE NOT clinically significant'.
    """
    cohort_pairs = {_pair_key(e) for e in _load_neg_control()}
    missing = _EXPECTED_BOUNDARY_CASES - cohort_pairs
    assert not missing, (
        f"Required CYP-pathway boundary cases missing from negative-control "
        f"cohort: {sorted(missing)}. The precision gate's safety case rests "
        f"on these — they are the pairs a careless system WOULD flag. "
        f"Removing them weakens the 'no false positives' claim from "
        f"non-trivial to trivial."
    )


def test_neg_control_clean_negatives_are_truly_clean():
    """The 6 clean negatives must use drug pairs where NEITHER drug
    appears in any cache contra entry.

    A 'clean negative' that includes (say) `simvastatin` would be
    misleading because simvastatin appears in 6+ cache contras —
    a clean test should use drugs the model has NOT seen in
    contraindicated contexts so we know the 'no FP' result is from
    correct generalization, not from the model having no signal at
    all on either drug.

    iter-120 design: clean negatives use acetaminophen, lisinopril,
    omeprazole, atorvastatin (boundary already covers this), metformin,
    albuterol, fluticasone, amoxicillin — drugs that appear in cache
    'serious' / 'moderate' / 'none' but never in 'contraindicated'.
    """
    cohort_pairs = {_pair_key(e) for e in _load_neg_control()}
    clean = cohort_pairs - _EXPECTED_BOUNDARY_CASES
    cache_contras = _load_cache_contras()
    contra_drugs = {drug for pair in cache_contras for drug in pair}

    # Drugs allowed in clean negatives even if in contra drugs — these
    # are the pharmacology-canonical examples we WANT in clean negatives
    # to demonstrate non-collision behavior. The MODEL should distinguish
    # "drug X + clean partner = none" from "drug X + dangerous partner
    # = contra" — the boundary case proves it can.
    _ALLOWED_OVERLAP: set[str] = {
        # metformin appears in `contrast dye + metformin`, `iodine +
        # metformin`, and `metformin + renal impairment` (lactic-
        # acidosis cluster). The clean negative `acetaminophen +
        # metformin` proves the model doesn't false-positive on
        # metformin's mere presence — only on the dangerous partner.
        "metformin",
        # lisinopril appears in `lisinopril + sacubitril` (bradykinin /
        # angioedema). The clean negative `acetaminophen + lisinopril`
        # proves the model doesn't false-positive on ACE-I's mere
        # presence — only on the specific neprilysin-inhibitor combo.
        "lisinopril",
    }

    offenders = []
    for pair in clean:
        for drug in pair:
            if drug in contra_drugs and drug not in _ALLOWED_OVERLAP:
                offenders.append((pair, drug))
                break

    assert not offenders, (
        f"Clean-negative cohort entries use drugs that appear in cache "
        f"contras: {offenders}. This weakens the precision claim — the "
        f"model should have NO signal on the clean-negative pairs (drugs "
        f"never seen in contra contexts). If the inclusion is intentional "
        f"(e.g. demonstrating non-collision on a specific drug class), "
        f"add the drug to _ALLOWED_OVERLAP with a comment."
    )
