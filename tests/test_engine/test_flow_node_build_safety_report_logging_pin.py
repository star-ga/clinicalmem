"""Pin: `engine.flow_runner._build_safety_report` emits a structured
flow-node DEBUG log on every flow execution.

Iter-329 T4 round-70 (32nd cross-pin family). Closes the LAST silent
flow-node helper inside `_dispatch_table()`. After iter-329 the
dispatch-table observability sweep is **4/4 complete**:

  iter-314  _bitnet_classify       (Layer 4.5 dispatch)  29th cross-pin
  iter-319  _deterministic_check   (Layer 1 dispatch)    30th cross-pin
  iter-324  _rxnorm_normalize      (Layer 3 dispatch)    31st cross-pin
  iter-329  _build_safety_report   (final aggregator)    32nd cross-pin (this)

Every per-flow-execution dispatch-table closure now emits a
structured DEBUG event with PHI-safe extras. Operators reading the
audit chain can compute per-layer dispatch frequency, severity-
distribution drift, and cohort coverage trends end-to-end.

Surfaces locked
===============
1. `flow_node_build_safety_report` event_name in source.
2. `_build_safety_report` body has ≥ 1 logger.X call.
3. Extras carry no drug-name keys (mirrors iter-309 forbidden-extras-
   keys discipline at the flow-node layer).

PHI safety
==========
Extras carry only structural metadata:

  patient_id        str  - synthetic Synthea identifier (not real PHI)
  node_count        int  - flow node count
  interaction_count int  - len(interactions) from upstream tiers

Drug names never reach the log record.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODULE_PATH = _REPO_ROOT / "engine" / "flow_runner.py"


def test_flow_node_build_safety_report_event_name_in_source():
    """The flow-node event_name string MUST appear in the source."""
    src = _MODULE_PATH.read_text()
    assert '"flow_node_build_safety_report"' in src, (
        "engine/flow_runner.py must emit 'flow_node_build_safety_report' "
        "DEBUG inside _build_safety_report(). iter-329 added this to "
        "close the LAST silent flow-node helper. Completes the "
        "dispatch-table observability sweep (4/4)."
    )


def test_build_safety_report_helper_has_logger_call():
    """The `_build_safety_report` closure body must contain a logger.X
    call. Source-level lower-bound guard against accidental deletion."""
    src = _MODULE_PATH.read_text()
    m = re.search(
        r"(    def _build_safety_report\(.*?)(?=\n    def |\n    table\[)",
        src,
        re.DOTALL,
    )
    assert m is not None, (
        "Could not find `_build_safety_report` helper body in "
        "engine/flow_runner.py. If renamed, update this pin."
    )
    body = m.group(1)
    n = len(re.findall(r"\blogger\.(info|warning|error|debug)\b", body))
    assert n >= 1, (
        f"_build_safety_report body has {n} logger.X call(s); "
        f"expected ≥ 1. iter-329 added flow-node observability — "
        f"removing it re-introduces the silent-path class AND breaks "
        f"the dispatch-table observability-sweep claim (4/4)."
    )


def test_build_safety_report_extras_have_no_phi_keys():
    """The flow_node_build_safety_report event extras MUST NOT carry
    raw drug-name keys. Mirrors iter-309 forbidden-extras-keys
    discipline at the flow-node layer (iter-314 + iter-319 + iter-324
    same shape)."""
    src = _MODULE_PATH.read_text()
    m = re.search(
        r'logger\.debug\(\s*\n\s*"flow_node_build_safety_report"'
        r',\s*\n\s*extra=\{(.*?)\}',
        src,
        re.DOTALL,
    )
    assert m is not None, (
        "Could not find logger.debug('flow_node_build_safety_report', "
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
        f"flow_node_build_safety_report extras leak PHI-risky keys: "
        f"{leaked!r}. Use only structural counts (patient_id, "
        f"node_count, interaction_count)."
    )
