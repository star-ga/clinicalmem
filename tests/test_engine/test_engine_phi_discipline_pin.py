"""Pin: PHI-discipline across ALL engine modules.

Iter 240 (T1 round 50) — generalizes the iter-234 (rxnorm_client) and
iter-239 (llm_synthesizer) PHI-leak catches into a SINGLE forward-
looking pin that scans every `engine/*.py` source file. The same
old-style positional `%s` pattern was caught in TWO separate modules
where exception objects were passed as `record.args`:

  iter-234 rxnorm_client:    logger.warning("Could not resolve %s", med)
  iter-239 llm_synthesizer:  logger.info("OpenAI failed: %s, trying next", e)

Both leak PHI: rxnorm `med` is a drug name from clinical context,
llm_synthesizer `e` is an httpx exception that can carry the prompt
body in its `__str__()` representation. The fix in both cases was
identical: rewrite to structured `extra={}` form with categorical
metadata only (`error_type`, `name_length`, `provider`, etc.).

This pin closes the broader regression class: any engine module that
introduces a positional-%s logger call passing PHI-sensitive variable
names fires the gate at commit time.

13th cross-pin family in the discipline lineage (after iter-178/183/
188/193/198/203/223/228/232/234/236/239).

Surface scanned
================
All `engine/*.py` files (excluding `__init__.py` and pure-data
modules).

Forbidden patterns (PHI-risky positional %s args)
==================================================
  • `logger.X("...%s...", e)` — exception object (httpx errors carry
    URL/body in __str__)
  • `logger.X("...%s...", exc)` / `logger.X("...%s...", err)` —
    same shape with different name
  • `logger.X("...%s...", med)` / `logger.X("...%s...", drug)` /
    `logger.X("...%s...", drug_name)` — drug-name PHI risk
  • `logger.X("...%s...", patient_id)` (in non-test paths) — internal
    identifier may flow to logs
  • `logger.X("...%s...", clean)` — common trimmed-drug-name var
  • `logger.X("...%s...", message)` — full message body

Allowed
=======
  • `logger.X("static text", positional_int_or_status_code)` —
    structured numerics (e.g. `resp.status_code`, `len(items)`)
  • All `logger.X(event_name, extra={...})` form — the structured
    discipline.

Adding a new engine module that legitimately needs positional `%s`
args (e.g. for non-PHI debug breadcrumbs) requires extending the
allow-list in this pin file deliberately, which is the whole point.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ENGINE_DIR = _REPO_ROOT / "engine"

# Variable names that are PHI-risky if passed as positional %s args.
# These cover the iter-234 + iter-239 sites and the broader pattern.
_PHI_RISKY_VAR_NAMES = (
    "e",          # exception object
    "exc",        # exception object
    "err",        # exception object
    "error",      # exception object
    "med",        # medication name
    "medication", # medication name
    "drug",       # drug name
    "drug_name",  # drug name
    "drug_a",     # drug-pair members
    "drug_b",
    "clean",      # iter-234 var name (cleaned drug name)
    "name",       # generic name (often drug)
    "term",       # search term (often drug/condition)
    "message",    # exception message body
    "msg",        # exception message body
    "body",       # response body
    "response_body",
    "text",       # response text
    "resp_text",
    "content",    # response content
    # iter-243 extension: structured-data risks
    "item",       # iteration variable (dict / cache entry)
    "entry",      # cache/db entry
    "record",     # record dict
    "row",        # database row
    "data",       # parsed JSON / response data
    "payload",    # request/response payload
    "obj",        # generic object
    "resource",   # FHIR / API resource
    # iter-248 extension: HIPAA "designed identifier" risks
    # (§ 164.514(b)(2)(i)(R) — "any unique identifying number,
    # characteristic, or code" — patient_id qualifies as PHI;
    # provider NPI is PHI-adjacent but conservative discipline
    # treats it the same.)
    "pid",            # patient ID (the iter-248 leak in clinical_memory.py:434)
    "patient_id",     # patient ID (explicit form)
    "mrn",            # medical record number
    "npi",            # provider NPI (the iter-248 leak in npi_registry.py)
    "provider_npi",   # explicit provider NPI form
)


def _engine_modules() -> list[Path]:
    """All engine/*.py files except __init__.py and pure-data
    modules (no logger usage)."""
    out = []
    for p in sorted(_ENGINE_DIR.glob("*.py")):
        if p.name == "__init__.py":
            continue
        text = p.read_text()
        if "logger." not in text:
            continue  # pure-data modules don't need the discipline
        out.append(p)
    return out


def test_no_positional_phi_risky_logger_calls():
    """Scan every `engine/*.py` for `logger.X("...%s...", <phi_risky_var>)`
    pattern. Catches the iter-234 / iter-239 PHI-leak class across ALL
    engine modules in one pin."""
    violations: list[tuple[Path, int, str, str]] = []

    for module_path in _engine_modules():
        text = module_path.read_text()
        # Build a regex that matches logger.X("...%s...", <name>) where
        # <name> is one of the PHI-risky names (whole-word match).
        #
        # We also need to avoid false-positives where the same var name
        # appears as a structured `extra={"name_length": len(name)}`
        # (which is the SAFE form). We achieve that by only matching
        # when the var name follows the format-string comma directly,
        # not deep inside an `extra={}` dict.
        # iter-248 approach: find every logger.X(...) call; for each,
        # extract the call body (between '(' and the matching ')').
        # If the body contains a fmt string with %s|%d|%r AND a positional
        # arg matching a PHI-risky var name (after the fmt string but
        # before any `extra=` kwarg), it's a violation.
        #
        # The iter-240 single-positional-arg regex let multi-arg cases
        # through (e.g. `logger.info("X %s: %s", npi, exc)` — `exc` was
        # the 2nd arg, regex only checked the 1st).
        for call_match in re.finditer(r'logger\.\w+\(', text):
            start = call_match.end()
            depth = 1
            i = start
            in_str: str | None = None  # track ' " or """
            while i < len(text) and depth > 0:
                c = text[i]
                if in_str:
                    if c == "\\" and i + 1 < len(text):
                        i += 2
                        continue
                    if text[i:i + len(in_str)] == in_str:
                        i += len(in_str)
                        in_str = None
                        continue
                else:
                    if text[i:i + 3] in ('"""', "'''"):
                        in_str = text[i:i + 3]
                        i += 3
                        continue
                    if c in ('"', "'"):
                        in_str = c
                        i += 1
                        continue
                    if c == '(':
                        depth += 1
                    elif c == ')':
                        depth -= 1
                        if depth == 0:
                            break
                i += 1
            body = text[start:i]
            # Need a fmt string with %s|%d|%r and at least one comma after
            fmt_match = re.search(r'(?:[urb]?["\'][^"\']*%[sdr][^"\']*["\'])\s*,', body)
            if not fmt_match:
                continue
            after_fmt = body[fmt_match.end():]
            # Strip out `extra=...` regions from the positional-arg
            # search space — `extra=` is the SAFE-form keyword.
            extra_idx = after_fmt.find("extra=")
            positional = after_fmt[:extra_idx] if extra_idx >= 0 else after_fmt
            for var in _PHI_RISKY_VAR_NAMES:
                if re.search(r'\b' + re.escape(var) + r'\b', positional):
                    line_no = text[:call_match.start()].count("\n") + 1
                    snippet = (text[call_match.start():call_match.start() + 200]
                               .replace("\n", " "))
                    violations.append((module_path, line_no, var, snippet))

    if violations:
        lines = []
        for path, line_no, var, snippet in violations:
            rel = path.relative_to(_REPO_ROOT)
            lines.append(
                f"  • {rel}:{line_no}  (var={var!r})\n"
                f"      {snippet}"
            )
        raise AssertionError(
            "Engine modules contain `logger.X(\"...%s...\", <phi_risky_var>)` "
            "patterns that leak PHI-sensitive content via record.args. Same "
            "iter-234 (rxnorm_client) / iter-239 (llm_synthesizer) class. "
            "Rewrite to structured `extra={...}` form with categorical "
            "metadata only (error_type, name_length, provider, status_code).\n"
            + "\n".join(lines)
        )


def test_engine_logger_extras_have_no_phi_field_keys():
    """Source-level scrub: no `extra={}` dict in any engine module may
    include a key from the forbidden-PHI-keys set. These would directly
    leak prompt / patient context / drug names to logs.

    Mirror of iter-239's extras-PHI-scrub but applied to ALL engine
    modules.
    """
    forbidden_keys = (
        "prompt",
        "system_msg",
        "patient_narrative",
        "drug_name",
        "drug_text",
        "med_name",
        "allergen",
        "condition_text",
        "response_body",
        "raw_response",
        "exception_msg",
        "error_message",
    )
    violations: list[tuple[Path, str, str]] = []

    for module_path in _engine_modules():
        text = module_path.read_text()
        for match in re.finditer(r"extra=\{[^}]*\}", text):
            block = match.group()
            for key in forbidden_keys:
                # match either "key": ... or 'key': ...
                if f'"{key}"' in block or f"'{key}'" in block:
                    violations.append((module_path, key, block[:160]))

    if violations:
        lines = []
        for path, key, block in violations:
            rel = path.relative_to(_REPO_ROOT)
            lines.append(f"  • {rel}: forbidden key {key!r} in extra={{}}\n      {block}")
        raise AssertionError(
            "Engine modules contain `extra={{}}` dicts with PHI-leak field "
            "keys. The structured discipline allows categorical metadata "
            "ONLY (lengths, types, status codes, public IDs):\n"
            + "\n".join(lines)
        )


def test_engine_modules_with_loggers_floor():
    """Sanity floor: at least 6 engine modules emit logger calls. Drops
    below means a bulk refactor unintentionally removed structured
    logging from a load-bearing module. Floor pinned at iter-240 from
    the live count of >= 10 logger-emitting engine modules.
    """
    n = len(_engine_modules())
    assert n >= 6, (
        f"Only {n} engine modules emit logger calls; iter-240 floor is 6. "
        f"Either a bulk refactor removed structured logging or a major "
        f"engine-module relocation happened. Anchor this floor lower with "
        f"a deliberate refactor commit if the change is intentional."
    )
