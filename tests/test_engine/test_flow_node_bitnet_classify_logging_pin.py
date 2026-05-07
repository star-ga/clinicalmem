"""Pin: `engine.flow_runner._bitnet_classify` emits a structured
flow-node DEBUG log on every flow execution.

Iter-314 T4 round-67 (29th cross-pin family). Pre-iter-314 the
per-flow-node `_bitnet_classify` helper inside `_dispatch_table()`
was silent — it called `classifier_layer()` for every drug pair (the
per-pair PHI-safe DEBUG event from engine.bitnet_classifier landed
at iter-309) but emitted no flow-node-level footprint. Without this
event operators couldn't tell from the audit chain alone:

  - WHICH flow execution triggered the per-pair events
  - HOW many pairs got classified per flow run
  - WHAT severity histogram the flow produced
  - WHICH weights bundle stamped them all

Same iter-289 / iter-304 silent-path observability class. iter-314
closes it with a single flow-node DEBUG event.

PHI safety
==========
The new event extras carry ONLY structural metadata:

  med_count           int — number of input medications
  pair_count          int — len(out) computed from cohort
  severity_histogram  dict[str, int] — categorical class counts only,
                                       NEVER per-pair correspondence
  weights_id_prefix   str — first 16 chars of the bundle SHA-256

Drug names never reach the log record. The severity histogram is a
categorical breakdown — given pair_count it does not reveal which
specific pairs were classified as which severity (information-
theoretically lossy under realistic cohorts).

Test gates
==========
1. Source-level: `flow_node_bitnet_classify` event_name appears in
   engine/flow_runner.py.
2. Source-level: `_bitnet_classify` body contains a logger.debug call.
3. Source-level: extras carry no drug_a/drug_b/medication/med/drug
   keys (mirrors the iter-309 forbidden-extras-keys discipline applied
   at the flow-node layer).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODULE_PATH = _REPO_ROOT / "engine" / "flow_runner.py"


def test_flow_node_bitnet_classify_event_name_in_source():
    """The flow-node event_name string MUST appear in the source."""
    src = _MODULE_PATH.read_text()
    assert '"flow_node_bitnet_classify"' in src, (
        "engine/flow_runner.py must emit 'flow_node_bitnet_classify' "
        "DEBUG inside _bitnet_classify(). iter-314 added this to close "
        "the silent flow-node observability gap. If renamed, update "
        "this pin."
    )


def test_bitnet_classify_helper_has_logger_call():
    """The `_bitnet_classify` closure body must contain a logger.X
    call. Source-level lower-bound guard against accidental deletion."""
    src = _MODULE_PATH.read_text()
    # Slice from "def _bitnet_classify" to the next "def " inside the
    # _dispatch_table closure body.
    m = re.search(
        r"(    def _bitnet_classify\(.*?)(?=\n    def |\n    table\[)",
        src,
        re.DOTALL,
    )
    assert m is not None, (
        "Could not find `_bitnet_classify` helper body in "
        "engine/flow_runner.py. If renamed, update this pin."
    )
    body = m.group(1)
    n = len(re.findall(r"\blogger\.(info|warning|error|debug)\b", body))
    assert n >= 1, (
        f"_bitnet_classify body has {n} logger.X call(s); expected ≥ 1. "
        f"iter-314 added flow-node observability — removing it "
        f"re-introduces the silent-path class."
    )


def test_bitnet_classify_extras_have_no_phi_keys():
    """The flow_node_bitnet_classify event extras MUST NOT carry raw
    drug-name keys. Mirrors the iter-309 forbidden-extras-keys
    discipline applied at the flow-node layer."""
    src = _MODULE_PATH.read_text()
    m = re.search(
        r'logger\.debug\(\s*\n\s*"flow_node_bitnet_classify"'
        r',\s*\n\s*extra=\{(.*?)\}',
        src,
        re.DOTALL,
    )
    assert m is not None, (
        "Could not find logger.debug('flow_node_bitnet_classify', "
        "extra={...}) in engine/flow_runner.py. If reformatted, update "
        "this pin to track the new shape."
    )
    extras_block = m.group(1)
    forbidden = (
        '"drug_a"', '"drug_b"', '"drug"', '"drug_pair"',
        '"medication"', '"med"', '"medications"', '"drug_name"',
    )
    leaked = [k for k in forbidden if k in extras_block]
    assert not leaked, (
        f"flow_node_bitnet_classify extras leak PHI-risky keys: "
        f"{leaked!r}. Use only structural counts (med_count, "
        f"pair_count, severity_histogram, weights_id_prefix)."
    )
