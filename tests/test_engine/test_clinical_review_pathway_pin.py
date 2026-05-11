"""Pin the I-6 invariant: clinical-review pathway documented in docs/clinical_validation.md.

I-6 (`docs/clinicalmem_invariants.md`) requires that the clinical-review
pathway be documented — target reviewer profile, review scope, and the
CMS-NPPES verification mechanism (the `validate_npi` Luhn validator that
any reviewer attestation would land against).

The body of the invariant cites three load-bearing claims that must
appear in `docs/clinical_validation.md`:

  1. A "Clinical review pathway" section (or equivalent heading)
  2. A target reviewer profile (specialty, licensure, NPI-verifiability)
  3. A reference to `engine.npi_registry.validate_npi` as the CMS NPPES
     check the reviewer's NPI must pass

Same iter-228 / iter-244 cross-pin shape (every load-bearing claim in a
user-facing doc is tied to a concrete code surface): a future copy-edit
that strips the section or its CMS-NPPES verification reference must
fail at commit time, not at judge time.
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CLINICAL_VALIDATION = _REPO_ROOT / "docs" / "clinical_validation.md"
_INVARIANTS_DOC = _REPO_ROOT / "docs" / "clinicalmem_invariants.md"
_NPI_REGISTRY = _REPO_ROOT / "engine" / "npi_registry.py"


def test_clinical_validation_has_review_pathway_section():
    text = _CLINICAL_VALIDATION.read_text()
    assert "Clinical review pathway" in text, (
        "docs/clinical_validation.md must contain a "
        "'Clinical review pathway' section describing the target "
        "reviewer profile (specialty, licensure, NPI-verifiability). "
        "The I-6 invariant in docs/clinicalmem_invariants.md cites "
        "this section as its enforcement surface."
    )


def test_review_pathway_names_target_specialty():
    text = _CLINICAL_VALIDATION.read_text().lower()
    expected_specialty_tokens = ("family medicine", "family-medicine")
    found = any(tok in text for tok in expected_specialty_tokens)
    assert found, (
        "docs/clinical_validation.md must name the target reviewer "
        f"specialty using one of: {expected_specialty_tokens}. "
        "The I-6 invariant body lists 'US-licensed family medicine' "
        "as the target reviewer profile."
    )


def test_review_pathway_cites_cms_nppes_validator():
    text = _CLINICAL_VALIDATION.read_text()
    assert "validate_npi" in text, (
        "docs/clinical_validation.md must cite the "
        "`engine/npi_registry.py::validate_npi` CMS NPPES Luhn "
        "validator as the verification mechanism any reviewer "
        "attestation would have to pass. The I-6 invariant body "
        "names this function explicitly."
    )


def test_npi_registry_validate_npi_function_exists():
    src = _NPI_REGISTRY.read_text()
    assert "def validate_npi" in src, (
        "engine/npi_registry.py must define `validate_npi`. The I-6 "
        "invariant + the clinical-review pathway section both point "
        "to this function; if it disappears, the doc claims rot."
    )


def test_invariants_doc_status_table_lists_i6_pathway():
    text = _INVARIANTS_DOC.read_text()
    assert "I-6 Clinical-review pathway documented" in text, (
        "docs/clinicalmem_invariants.md status table must list "
        "'I-6 Clinical-review pathway documented' (not the old "
        "'Clinician attestation' framing). Body + status table "
        "must stay in lockstep — same iter-232 single-source-of-"
        "truth → derived-surface drift class."
    )
