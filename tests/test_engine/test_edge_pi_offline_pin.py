"""Pin the iter-73 Edge / Pi-offline build spec + its demo + JUDGES surfaces.

The Edge profile is the load-bearing GTM hook for hospital IT buyers:

  - 45 MB bundle on a $15 Pi Zero 2 W
  - Three plug-and-play deployment modes (USB OTG drop-in, office-router
    drop-in, EHR sidecar)
  - Same SHA-256 audit hash on Pi as on cloud

Without these pins, a future copy edit to the demo could quietly drop
the "$15 Pi Zero 2 W · USB plug-in" claim, the Box product profile
could drift out of `edge_pi_offline.md`, or the JUDGES.md audit-trail
row could fall out of step with what's actually documented.

These tests block any of those regressions.
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EDGE_DOC = _REPO_ROOT / "docs" / "edge_pi_offline.md"
_DEMO_HTML = _REPO_ROOT / "docs" / "demo.html"
_JUDGES_MD = _REPO_ROOT / "JUDGES.md"
_ARCH_MD = _REPO_ROOT / "docs" / "architecture.md"


# ─── docs/edge_pi_offline.md content ─────────────────────────────────────

def test_edge_doc_exists():
    assert _EDGE_DOC.exists(), (
        f"docs/edge_pi_offline.md must exist — it is the GTM spec for the "
        f"$15 Pi Zero 2 W / USB drop-in deployment story. Path: {_EDGE_DOC}"
    )


def test_edge_doc_pi_zero_price():
    body = _EDGE_DOC.read_text()
    assert "$15" in body, (
        "edge_pi_offline.md must quote the official $15 MSRP for the "
        "Pi Zero 2 W — the price is the headline GTM number"
    )
    assert "Pi Zero 2 W" in body, (
        "edge_pi_offline.md must name the Pi Zero 2 W explicitly (not a "
        "generic 'Raspberry Pi') — the Cortex-A53 / 512 MB / USB OTG "
        "trio is what makes the Box product viable"
    )


def test_edge_doc_clinicalmem_box_product_profile():
    body = _EDGE_DOC.read_text()
    assert "ClinicalMem Box" in body, (
        "edge_pi_offline.md must contain the 'ClinicalMem Box' section — "
        "the named hardware product is the user's GTM ask"
    )
    # The three deployment modes the box ships in.
    for mode in ("USB drop-in", "router drop-in", "EHR sidecar"):
        assert mode in body, (
            f"edge_pi_offline.md must document deployment mode: {mode!r}"
        )


def test_edge_doc_usb_otg_gadget_mode():
    body = _EDGE_DOC.read_text()
    assert "g_ether" in body, (
        "edge_pi_offline.md must reference the `g_ether` kernel module "
        "by name — it is the load-bearing technical claim for "
        "zero-driver USB plug-in"
    )
    assert "USB OTG" in body or "USB-Ethernet" in body, (
        "edge_pi_offline.md must explain the USB OTG / USB-Ethernet "
        "gadget-mode mechanism — that's what makes Mode 1 zero-IT"
    )


def test_edge_doc_pi_tier_matrix():
    body = _EDGE_DOC.read_text()
    # Each tier the spec promises latency numbers for must appear by name.
    for chip in ("Pi 5", "Pi 4", "Pi Zero 2 W", "ESP32"):
        assert chip in body, (
            f"edge_pi_offline.md must include the {chip!r} tier in the "
            f"latency matrix — judges will skim this"
        )


def test_edge_doc_offline_vs_online_table():
    body = _EDGE_DOC.read_text()
    # The table must call out which layer needs internet.
    assert "Layer 5" in body, (
        "edge_pi_offline.md must document that Layer 5 (LLM consensus) "
        "is the only online-required tier — the rest run offline"
    )
    assert "bit-identical" in body.lower() or "Bit-identical" in body, (
        "edge_pi_offline.md must promise bit-identical SHA-256 hashes "
        "across Pi/x86/CUDA — that's the FDA SaMD value prop"
    )


# ─── docs/demo.html surfacing ────────────────────────────────────────────

def test_demo_bitnet_chip_links_to_edge_doc():
    html = _DEMO_HTML.read_text()
    assert 'href="edge_pi_offline.md"' in html, (
        "docs/demo.html must contain at least one href pointing to "
        "edge_pi_offline.md — the chip + section pill should be "
        "discoverable from the hero in one click"
    )


def test_demo_offline_chip_copy():
    html = _DEMO_HTML.read_text()
    # The new chip subtitle (replaces stale 'runs on Pi Zero ($15)' text).
    assert "Pi Zero 2 W" in html, (
        "docs/demo.html must reference the Pi Zero 2 W in the BitNet "
        "weights chip — pricing the deployment in the hero is the "
        "5-second hook"
    )
    assert "offline" in html.lower(), (
        "docs/demo.html must mention 'offline' somewhere in the BitNet "
        "weights / Pi-Box messaging"
    )
    assert "USB plug-in" in html, (
        "docs/demo.html must mention 'USB plug-in' in the BitNet section "
        "pill — that's the zero-IT deployment claim"
    )


def test_demo_no_stale_pi_phrases():
    """Old iter-72 phrasing was 'runs on Pi Zero ($15)'. Iter-73 replaced
    it with the 2 W variant + offline + USB plug-in claims. Any future
    revert to the bare phrasing should fail the gate.
    """
    html = _DEMO_HTML.read_text()
    # The exact stale string the iter-73 edit replaced. We must never see
    # it again — its replacement is richer (offline + audit hash).
    assert "runs on Pi Zero ($15)</div>" not in html, (
        "docs/demo.html still contains the iter-72 stale chip subtitle "
        "'runs on Pi Zero ($15)'. The iter-73 version is "
        "'$15 Pi Zero 2 W · offline · same audit hash' — restore it."
    )


# ─── JUDGES.md audit-trail row ───────────────────────────────────────────

def test_judges_has_edge_audit_trail_row():
    body = _JUDGES_MD.read_text()
    assert "Edge / offline build" in body, (
        "JUDGES.md must list 'Edge / offline build' in the audit-trail "
        "table so judges can verify the deployment claim"
    )
    assert "edge_pi_offline.md" in body, (
        "JUDGES.md audit-trail row must point at docs/edge_pi_offline.md"
    )
    assert "ClinicalMem Box" in body, (
        "JUDGES.md must name the ClinicalMem Box product so the row "
        "isn't just an abstract Pi reference"
    )


# ─── docs/architecture.md cross-link ─────────────────────────────────────

def test_architecture_md_cross_links_edge_doc():
    body = _ARCH_MD.read_text()
    assert "edge_pi_offline.md" in body, (
        "docs/architecture.md must cross-link to edge_pi_offline.md — "
        "the architecture doc is the canonical entry point and judges "
        "following the breadcrumb should land on the Edge spec"
    )
