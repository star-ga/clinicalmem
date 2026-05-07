"""Pin: `engine.flow_runner._rxnorm_normalize` emits a structured
flow-node DEBUG log on every flow execution.

Iter-324 T4 round-69 (31st cross-pin family). Closes the 3rd silent
flow-node helper inside `_dispatch_table()`. iter-314 closed the 1st
on `_bitnet_classify` (29th cross-pin, Layer 4.5 dispatch); iter-319
closed the 2nd on `_deterministic_check` (30th cross-pin, Layer 1
dispatch); iter-324 closes the 3rd on `_rxnorm_normalize` (Layer 3
dispatch). Same iter-289 / iter-304 silent-path observability class.

Surfaces locked
===============
1. `flow_node_rxnorm_normalize` event_name in source.
2. `_rxnorm_normalize` body has ≥ 1 logger.X call.
3. Extras carry no drug-name keys (mirrors iter-309 forbidden-extras-
   keys discipline at the flow-node layer).

PHI safety
==========
Extras carry only structural metadata:

  med_count          int   - input medication count
  resolved_count     int   - meds appearing in deterministic table
  coverage_ratio_q4  float - coverage / len(meds), rounded to 4
                             decimals. NEVER per-drug correspondence.

Drug names never reach the log record.

1 silent flow-node helper remains in _dispatch_table
(`_build_safety_report`) as the last future T4 candidate.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODULE_PATH = _REPO_ROOT / "engine" / "flow_runner.py"


def test_flow_node_rxnorm_normalize_event_name_in_source():
    """The flow-node event_name string MUST appear in the source."""
    src = _MODULE_PATH.read_text()
    assert '"flow_node_rxnorm_normalize"' in src, (
        "engine/flow_runner.py must emit 'flow_node_rxnorm_normalize' "
        "DEBUG inside _rxnorm_normalize(). iter-324 added this to "
        "close the silent flow-node observability gap on the Layer 3 "
        "RxNorm normalisation. If renamed, update this pin."
    )


def test_rxnorm_normalize_helper_has_logger_call():
    """The `_rxnorm_normalize` closure body must contain a logger.X
    call. Source-level lower-bound guard against accidental deletion."""
    src = _MODULE_PATH.read_text()
    m = re.search(
        r"(    def _rxnorm_normalize\(.*?)(?=\n    def |\n    table\[)",
        src,
        re.DOTALL,
    )
    assert m is not None, (
        "Could not find `_rxnorm_normalize` helper body in "
        "engine/flow_runner.py. If renamed, update this pin."
    )
    body = m.group(1)
    n = len(re.findall(r"\blogger\.(info|warning|error|debug)\b", body))
    assert n >= 1, (
        f"_rxnorm_normalize body has {n} logger.X call(s); "
        f"expected ≥ 1. iter-324 added flow-node observability — "
        f"removing it re-introduces the silent-path class."
    )


def test_rxnorm_normalize_extras_have_no_phi_keys():
    """The flow_node_rxnorm_normalize event extras MUST NOT carry
    raw drug-name keys. Mirrors iter-309 forbidden-extras-keys
    discipline at the flow-node layer (iter-314 + iter-319 same
    shape applied to `_bitnet_classify` + `_deterministic_check`)."""
    src = _MODULE_PATH.read_text()
    m = re.search(
        r'logger\.debug\(\s*\n\s*"flow_node_rxnorm_normalize"'
        r',\s*\n\s*extra=\{(.*?)\}',
        src,
        re.DOTALL,
    )
    assert m is not None, (
        "Could not find logger.debug('flow_node_rxnorm_normalize', "
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
        f"flow_node_rxnorm_normalize extras leak PHI-risky keys: "
        f"{leaked!r}. Use only structural counts (med_count, "
        f"resolved_count, coverage_ratio_q4)."
    )
