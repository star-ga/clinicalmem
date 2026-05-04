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


def test_verify_replay_does_not_claim_client_side_forward_pass():
    text = _DEMO.read_text()
    forbidden_phrases = [
        # The exact prior wording.
        "hashes the deterministic Q16.16 forward pass client-side",
        # Common drift variants of the same false claim.
        "client-side Q16.16 forward pass",
        "in-browser ternary forward",
        "in-browser BitNet forward",
        "browser runs the ternary forward",
        "browser computes the Q16.16 forward",
    ]
    for phrase in forbidden_phrases:
        assert phrase.lower() not in text.lower(), (
            f"misleading client-side claim back in demo.html: {phrase!r}"
        )


def test_verify_replay_labels_demo_cache_explicitly():
    text = _DEMO.read_text()
    # The new copy makes the cache explicit so a judge skim catches it.
    assert "demo cache" in text.lower(), (
        "expected 'demo cache' label on the Verify Replay tile"
    )
    assert "39 pairs" in text, (
        "expected '39 pairs' label disclosing the cache size"
    )


def test_verify_replay_explains_server_side_forward_pass():
    """Positive assertion: the new copy must point to where the live
    forward pass actually runs."""
    text = _DEMO.read_text()
    assert "server-side" in text.lower()
    assert "engine.bitnet_classifier" in text or "engine/bitnet_classifier" in text
