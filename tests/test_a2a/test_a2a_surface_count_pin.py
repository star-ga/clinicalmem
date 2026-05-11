"""Pin the A2A surface counts.

The dashboard / README / JUDGES.md cite "A2A (13 tools)" or "A2A
(13 skills)" — sometimes interchangeably. The two numbers are
distinct in the agent design:

  - **13 ADK tools**: function-level callables registered into the
    `Agent(tools=[...])` array in `a2a_agent/agent.py`. These are
    the underlying execution units (4 FHIR + 2 memory + 4 safety +
    1 GenAI synthesis + 4 v4.0 expansion).
  - **5 AgentSkills**: high-level user-facing skill identifiers
    published in the AgentCard via `AgentSkill(...)` in
    `a2a_agent/app.py`. These compose multiple tools into named
    workflows for clients.

This test pins both counts so future tool/skill churn can't
silently drift the dashboard claims.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT = _REPO_ROOT / "a2a_agent" / "agent.py"
_APP = _REPO_ROOT / "a2a_agent" / "app.py"

_EXPECTED_ADK_TOOLS = 13
_EXPECTED_AGENT_SKILLS = 5


def _count_adk_tools() -> int:
    """Count entries inside the `tools=[...]` array in agent.py.

    Walks the file and counts top-level identifier-or-call entries
    between `tools=[` and the matching `]`. Comment-only lines and
    blank lines don't count.
    """
    text = _AGENT.read_text()
    m = re.search(r"tools=\[(.*?)\]\s*,", text, re.DOTALL)
    assert m is not None, "agent.py must declare `tools=[...]` on Agent"
    body = m.group(1)
    items = 0
    for line in body.splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith("#"):
            continue
        items += 1
    return items


def _count_agent_skills() -> int:
    """Count `AgentSkill(...)` registrations in app.py."""
    return sum(1 for _ in re.finditer(r"\bAgentSkill\(", _APP.read_text()))


def test_adk_tool_count_pinned():
    n = _count_adk_tools()
    assert n == _EXPECTED_ADK_TOOLS, (
        f"ADK tool count drifted: live={n}, pinned={_EXPECTED_ADK_TOOLS}. "
        f"Update _EXPECTED_ADK_TOOLS + every doc that cites 'A2A 13 tools' "
        f"in the same commit."
    )


def test_agent_skill_count_pinned():
    n = _count_agent_skills()
    assert n == _EXPECTED_AGENT_SKILLS, (
        f"AgentSkill count drifted: live={n}, pinned={_EXPECTED_AGENT_SKILLS}. "
        f"Either AgentCard.skills was edited or the AgentSkill regex no "
        f"longer matches. Update _EXPECTED_AGENT_SKILLS or the test."
    )


def test_dashboard_judges_cite_pinned_a2a_count():
    """The user-facing 13-tools figure must appear in the audit-trail map.

    Accepts any of these canonical phrasings (in priority order):
      - "A2A (5 skills · 13 tools)" — current canonical form, both counts
      - "A2A (13 skills)"            — older shorthand
      - "A2A (13 tools)"             — older shorthand
      - "13 tools"                   — bare count anywhere
    """
    judges = (_REPO_ROOT / "JUDGES.md").read_text()
    accepted = (
        f"A2A ({_EXPECTED_AGENT_SKILLS} skills · {_EXPECTED_ADK_TOOLS} tools)" in judges
        or f"A2A ({_EXPECTED_ADK_TOOLS} skills)" in judges
        or f"A2A ({_EXPECTED_ADK_TOOLS} tools)" in judges
        or f"{_EXPECTED_ADK_TOOLS} tools" in judges
    )
    assert accepted, (
        f"JUDGES.md must cite the live A2A count "
        f"({_EXPECTED_ADK_TOOLS} tools / {_EXPECTED_AGENT_SKILLS} skills)"
    )
