"""Pin: `engine.flow_runner._deterministic_check` emits a structured
flow-node DEBUG log on every flow execution.

Iter-319 T4 round-68 (30th cross-pin family). Closes the 2nd silent
flow-node helper inside `_dispatch_table()` (iter-314 closed the 1st
on `_bitnet_classify`; iter-314 noted `_deterministic_check`,
`_rxnorm_normalize`, and `_build_safety_report` as future T4
candidates — iter-319 closes `_deterministic_check`).

Pre-iter-319 the Layer 1 deterministic-table stamp helper called
`check_drug_interactions()` on every flow execution but emitted no
flow-node-level footprint. Same iter-289 / iter-304 / iter-314
silent-path observability class but at the Layer 1 dispatch entry.

Surfaces locked
===============
1. `flow_node_deterministic_check` event_name in source.
2. `_deterministic_check` body has ≥ 1 logger.X call.
3. Extras carry no drug-name keys (mirrors iter-309 forbidden-extras-
   keys discipline at the flow-node layer + iter-314 same shape on
   `_bitnet_classify`).

PHI safety
==========
Extras carry only structural metadata:

  med_count           int — input medication count
  interaction_count   int — len(results) from Layer 1 table
  severity_histogram  dict[str, int] — categorical class counts only,
                                       NEVER per-pair correspondence

Drug names never reach the log record. The severity histogram is a
categorical breakdown — given interaction_count it does not reveal
which specific pairs were classified as which severity (information-
theoretically lossy).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODULE_PATH = _REPO_ROOT / "engine" / "flow_runner.py"


def test_flow_node_deterministic_check_event_name_in_source():
    """The flow-node event_name string MUST appear in the source."""
    src = _MODULE_PATH.read_text()
    assert '"flow_node_deterministic_check"' in src, (
        "engine/flow_runner.py must emit 'flow_node_deterministic_check' "
        "DEBUG inside _deterministic_check(). iter-319 added this to "
        "close the silent flow-node observability gap on the Layer 1 "
        "table stamp. If renamed, update this pin."
    )


def test_deterministic_check_helper_has_logger_call():
    """The `_deterministic_check` closure body must contain a
    logger.X call. Source-level lower-bound guard against accidental
    deletion."""
    src = _MODULE_PATH.read_text()
    m = re.search(
        r"(    def _deterministic_check\(.*?)(?=\n    def |\n    table\[)",
        src,
        re.DOTALL,
    )
    assert m is not None, (
        "Could not find `_deterministic_check` helper body in "
        "engine/flow_runner.py. If renamed, update this pin."
    )
    body = m.group(1)
    n = len(re.findall(r"\blogger\.(info|warning|error|debug)\b", body))
    assert n >= 1, (
        f"_deterministic_check body has {n} logger.X call(s); "
        f"expected ≥ 1. iter-319 added flow-node observability — "
        f"removing it re-introduces the silent-path class."
    )


def test_deterministic_check_extras_have_no_phi_keys():
    """The flow_node_deterministic_check event extras MUST NOT carry
    raw drug-name keys. Mirrors iter-309 forbidden-extras-keys
    discipline at the flow-node layer (iter-314 same shape applied to
    `_bitnet_classify`)."""
    src = _MODULE_PATH.read_text()
    m = re.search(
        r'logger\.debug\(\s*\n\s*"flow_node_deterministic_check"'
        r',\s*\n\s*extra=\{(.*?)\}',
        src,
        re.DOTALL,
    )
    assert m is not None, (
        "Could not find logger.debug('flow_node_deterministic_check', "
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
        f"flow_node_deterministic_check extras leak PHI-risky keys: "
        f"{leaked!r}. Use only structural counts (med_count, "
        f"interaction_count, severity_histogram)."
    )
