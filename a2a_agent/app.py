"""
ClinicalMem Agent — A2A application entry point for Submission 2.

Start the server with:
    uvicorn a2a_agent.app:a2a_app --host 0.0.0.0 --port 8001

The agent card is served publicly at:
    GET http://localhost:8001/.well-known/agent-card.json
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentSkill,
    APIKeySecurityScheme,
    In,
    SecurityScheme,
)
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from a2a_agent.agent import root_agent

logger = logging.getLogger(__name__)

# ── API Key Middleware ────────────────────────────────────────────────────────

_raw_keys = os.getenv("API_KEYS", "")
if not _raw_keys:
    logger.warning("API_KEYS not set — rejecting all requests until configured")
VALID_API_KEYS = set(filter(None, _raw_keys.split(",")))


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/.well-known/agent-card.json":
            return await call_next(request)
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "detail": "X-API-Key header required"},
            )
        if api_key not in VALID_API_KEYS:
            return JSONResponse(
                status_code=403,
                content={"error": "Forbidden", "detail": "Invalid API key"},
            )
        return await call_next(request)


# ── Agent Card ────────────────────────────────────────────────────────────────

BASE_URL = os.getenv("BASE_URL", "http://localhost:8001")
PO_PLATFORM_URL = os.getenv("PO_PLATFORM_BASE_URL", "http://localhost:5139")
PORT = int(os.getenv("PORT", "8001"))

agent_card = AgentCard(
    name="clinicalmem_agent",
    description=(
        "ClinicalMem — Persistent, auditable, contradiction-safe clinical memory. "
        "Provides medication safety analysis, clinical context recall with confidence "
        "gating, and tamper-proof audit trails. Powered by mind-mem engine with "
        "MIND Lang scoring kernels. By STARGA Inc."
    ),
    url=BASE_URL,
    version="0.1.0",
    defaultInputModes=["text/plain"],
    defaultOutputModes=["text/plain"],
    capabilities=AgentCapabilities(
        streaming=True,
        pushNotifications=False,
        stateTransitionHistory=True,
        extensions=[
            AgentExtension(
                uri=f"{PO_PLATFORM_URL}/schemas/a2a/v1/fhir-context",
                description="FHIR R4 context — provides patient identity and FHIR server credentials.",
                required=True,
            )
        ],
    ),
    skills=[
        AgentSkill(
            id="medication-safety-review",
            name="medication-safety-review",
            description=(
                "Comprehensive medication reconciliation: drug-drug interaction detection, "
                "allergy cross-reference, severity scoring. Uses MIND Lang clinical kernels."
            ),
            tags=["medications", "safety", "interactions", "allergies"],
        ),
        AgentSkill(
            id="clinical-context-recall",
            name="clinical-context-recall",
            description=(
                "Answer questions about patient history using persistent memory with "
                "importance-scored retrieval and confidence gating (abstention when "
                "evidence is insufficient)."
            ),
            tags=["memory", "recall", "context", "history"],
        ),
        AgentSkill(
            id="contradiction-assessment",
            name="contradiction-assessment",
            description=(
                "Scan patient record for conflicting information across providers, "
                "dates, and data sources. Detects allergy-medication conflicts and "
                "dangerous drug interactions."
            ),
            tags=["contradictions", "safety", "audit"],
        ),
        AgentSkill(
            id="care-transition-summary",
            name="care-transition-summary",
            description=(
                "Generate structured handoff summary for care transitions. "
                "Highlights active issues, medication conflicts, and pending actions."
            ),
            tags=["transitions", "handoff", "summary"],
        ),
        AgentSkill(
            id="explain-conflict",
            name="explain-conflict",
            description=(
                "GenAI-powered clinical conflict explanation. Uses LLM to generate "
                "patient-specific rationale for a detected safety conflict, with evidence "
                "citations and hard abstention when evidence is insufficient."
            ),
            tags=["genai", "explanation", "safety", "citations"],
        ),
    ],
    securitySchemes={
        "apiKey": SecurityScheme(
            root=APIKeySecurityScheme(
                type="apiKey",
                name="X-API-Key",
                in_=In.header,
                description="API key required to access ClinicalMem agent.",
            )
        )
    },
    security=[{"apiKey": []}],
)

# ── Build A2A app ─────────────────────────────────────────────────────────────

a2a_app = to_a2a(root_agent, port=PORT, agent_card=agent_card)
a2a_app.add_middleware(ApiKeyMiddleware)
