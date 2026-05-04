"""Pin: hero contains a discoverable link to JUDGES.md.

Iter 52: hackathon judges decide in the first 5 seconds. The full
60-second audit guide lives in JUDGES.md (5-bullet pitch + single-command
gate + full audit trail) and is the load-bearing artifact for the
2026-05-11 DevPost evaluation. The hero must surface it visibly so a
judge sees it before scrolling.

This test prevents a future iteration from removing the link or
demoting it below the hero fold.
"""
from pathlib import Path

_DEMO = Path(__file__).resolve().parents[2] / "docs" / "demo.html"


def test_hero_links_to_judges_md():
    text = _DEMO.read_text()
    judges_url = "https://github.com/star-ga/clinicalmem/blob/main/JUDGES.md"
    assert judges_url in text, (
        f"Hero must link to {judges_url} so a hackathon judge can find "
        f"the 60-second audit guide before scrolling."
    )
    # Must reference JUDGES.md by name, with intent — not just a bare URL.
    assert "JUDGES.md" in text
    assert "Judges:" in text or "For judges" in text or "60-second audit" in text


def test_judges_link_appears_in_hero_section():
    """The link must be inside the hero, not buried in a later section."""
    text = _DEMO.read_text()
    # The hero ends at the first opening of `<section id="patient"` or
    # similar — find the JUDGES link's offset and ensure it lands before
    # the second top-level <section>.
    judges_offset = text.find("JUDGES.md")
    assert judges_offset != -1
    # The hero is the first <section>; the second <section> begins the
    # patient flow. Find the FIRST occurrence of `<section id="patient"`.
    patient_offset = text.find('<section id="patient"')
    assert patient_offset != -1, "patient section anchor not found"
    assert judges_offset < patient_offset, (
        "JUDGES.md link must be in the hero (above the patient section), "
        "not buried below it."
    )
