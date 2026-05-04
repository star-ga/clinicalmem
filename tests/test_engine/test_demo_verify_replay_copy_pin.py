"""Pin the Verify-Replay copy in docs/demo.html so it stays accurate.

External judge LLMs (round-4 multi-LLM eval, n=6) flagged the BitNet
demo block as the highest-friction wow-factor item: the original copy
implied the browser ran the Q16.16 ternary forward pass, when in reality
the browser does lookup against a 39-pair cache that the *server-side*
classifier produced. The cache + lookup + bit-equality verification IS
genuine, but the copy must say what the demo actually does.

This test prevents the misleading phrasing from sneaking back in.

If a future iteration ships an actual in-browser BitNet runtime (e.g.
WASM-compiled forward pass), this test must be deleted and replaced with
a test that asserts the WASM artifact ships and the JS calls into it.
"""
from pathlib import Path

_DEMO = Path(__file__).resolve().parents[2] / "docs" / "demo.html"


def test_verify_replay_does_not_claim_misleading_client_side_inference():
    """Iter-46 phrasing forbidden because it falsely promised in-browser
    inference when the demo only did a lookup. Iter-69 ACTUALLY ships
    in-browser inference via docs/bitnet_browser.js, so the iter-46
    phrasing is no longer false — but the specific FALSE phrases (which
    used inaccurate framing) are still blocked. The new iter-69 phrases
    that are honest are listed below as REQUIRED."""
    text = _DEMO.read_text()
    # These exact strings were the iter-46 misleading variants. Now
    # they're either (a) forbidden because they referenced a lookup
    # that no longer exists, or (b) replaced by the iter-69 in-browser
    # framing.
    forbidden_phrases = [
        # Iter-46 would say this when it was just doing a lookup.
        "hashes the deterministic Q16.16 forward pass client-side",  # imprecise
    ]
    for phrase in forbidden_phrases:
        assert phrase.lower() not in text.lower(), (
            f"deprecated iter-46 phrasing back in demo.html: {phrase!r}"
        )


def test_verify_replay_announces_in_browser_inference():
    """Iter 69 ships LIVE in-browser inference via bitnet_browser.js.
    The chip + body copy must reflect that, NOT the iter-46 lookup."""
    text = _DEMO.read_text()
    # The iter-69 chip is the visible signal.
    assert "live JS · in-browser Q16.16" in text
    # Body must reference the JS port file path so an auditor knows
    # exactly what runs.
    assert "docs/bitnet_browser.js" in text
    # And must point at the cross-language pin test.
    assert "test_browser_bitnet_pin.py" in text


def test_verify_replay_no_longer_claims_lookup_table():
    """Iter-46 explicitly framed Verify Replay as a 39-pair lookup
    fallback. Iter-69 removed the lookup entirely. The body MUST NOT
    promise a server-side fallback or a 39-pair cache anymore — that
    framing is stale and contradicts the live behavior."""
    text = _DEMO.read_text()
    # The lookup-related body text iter-46 shipped:
    deprecated_phrases = [
        "the live classifier runs server-side under",
        "Pairs outside the cache fall back to a canonical-input",
    ]
    for phrase in deprecated_phrases:
        assert phrase not in text, (
            f"iter-46 lookup-era phrasing still in demo.html: {phrase!r}"
        )
