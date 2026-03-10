"""Tests for a2a_agent/app.py — RateLimitMiddleware and ApiKeyMiddleware dispatch."""
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

a2a_types = pytest.importorskip("a2a.types")
google_adk = pytest.importorskip("google.adk")

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient


@pytest.fixture
def rate_limit_mw():
    from a2a_agent.app import RateLimitMiddleware
    mock_app = MagicMock()
    mw = RateLimitMiddleware(mock_app)
    return mw


class TestRateLimitMiddlewareInit:
    def test_defaults(self, rate_limit_mw):
        assert rate_limit_mw.MAX_REQUESTS == 60
        assert rate_limit_mw.WINDOW_SECONDS == 60
        assert hasattr(rate_limit_mw, '_requests')


class TestApiKeyMiddleware:
    def test_middleware_exists(self):
        from a2a_agent.app import ApiKeyMiddleware
        mock_app = MagicMock()
        mw = ApiKeyMiddleware(mock_app)
        assert mw is not None


class TestAgentCardComplete:
    def test_all_skills_present(self):
        from a2a_agent.app import agent_card
        assert len(agent_card.skills) == 5
        skill_names = [s.name for s in agent_card.skills]
        assert "medication-safety-review" in skill_names
        assert "clinical-context-recall" in skill_names
        assert "contradiction-assessment" in skill_names
        assert "care-transition-summary" in skill_names
        assert "explain-conflict" in skill_names

    def test_url_and_version(self):
        from a2a_agent.app import agent_card
        assert agent_card.url
        assert agent_card.version == "0.1.0"

    def test_extension_fhir_context(self):
        from a2a_agent.app import agent_card
        ext = agent_card.capabilities.extensions[0]
        assert "fhir-context" in ext.uri
        assert ext.required is True


class TestA2aApp:
    def test_app_created(self):
        from a2a_agent.app import a2a_app
        assert a2a_app is not None


class TestValidApiKeys:
    def test_valid_keys_set(self):
        from a2a_agent.app import VALID_API_KEYS
        # Should be a set (may be empty if API_KEYS env not set)
        assert isinstance(VALID_API_KEYS, set)


# ── ASGI dispatch tests for RateLimitMiddleware (lines 56-69) ──────────────


def _make_starlette_app_with_rate_limit():
    """Create a minimal Starlette app with RateLimitMiddleware only."""
    from a2a_agent.app import RateLimitMiddleware

    async def home(request):
        return JSONResponse({"ok": True})

    async def agent_card_endpoint(request):
        return JSONResponse({"name": "test-agent"})

    app = Starlette(routes=[
        Route("/test", home),
        Route("/.well-known/agent-card.json", agent_card_endpoint),
    ])
    app.add_middleware(RateLimitMiddleware)
    return app


class TestRateLimitMiddlewareDispatch:
    """Cover lines 56-69: RateLimitMiddleware.dispatch() logic."""

    def test_normal_request_passes(self):
        """A single request below the rate limit passes through."""
        app = _make_starlette_app_with_rate_limit()
        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_agent_card_bypasses_rate_limit(self):
        """Line 56-57: /.well-known/agent-card.json is exempt from rate limiting."""
        app = _make_starlette_app_with_rate_limit()
        client = TestClient(app)
        # Hit the agent card endpoint many times — should never be rate limited
        for _ in range(100):
            resp = client.get("/.well-known/agent-card.json")
            assert resp.status_code == 200

    def test_rate_limit_exceeded_returns_429(self):
        """Lines 62-66: After MAX_REQUESTS in a window, returns 429."""
        from a2a_agent.app import RateLimitMiddleware

        async def home(request):
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/test", home)])
        app.add_middleware(RateLimitMiddleware)
        client = TestClient(app)

        # Send MAX_REQUESTS requests — they should all pass
        for _ in range(RateLimitMiddleware.MAX_REQUESTS):
            resp = client.get("/test")
            assert resp.status_code == 200

        # The next request should be rate limited
        resp = client.get("/test")
        assert resp.status_code == 429
        body = resp.json()
        assert body["error"] == "Too Many Requests"
        assert "60/min" in body["detail"]

    def test_rate_limit_uses_api_key_header(self):
        """Line 58: Rate limit key is X-API-Key when present."""
        app = _make_starlette_app_with_rate_limit()
        client = TestClient(app)

        # Two different API keys should have independent rate limits
        resp1 = client.get("/test", headers={"X-API-Key": "key-a"})
        assert resp1.status_code == 200

        resp2 = client.get("/test", headers={"X-API-Key": "key-b"})
        assert resp2.status_code == 200


# ── ASGI dispatch tests for ApiKeyMiddleware (lines 74-87) ─────────────────


def _make_starlette_app_with_api_key(valid_keys=None):
    """Create a minimal Starlette app with ApiKeyMiddleware only."""
    from a2a_agent.app import ApiKeyMiddleware
    import a2a_agent.app as app_module

    async def home(request):
        return JSONResponse({"ok": True})

    async def agent_card_endpoint(request):
        return JSONResponse({"name": "test-agent"})

    app = Starlette(routes=[
        Route("/test", home),
        Route("/.well-known/agent-card.json", agent_card_endpoint),
    ])
    app.add_middleware(ApiKeyMiddleware)

    # Temporarily set valid keys
    original_keys = app_module.VALID_API_KEYS
    if valid_keys is not None:
        app_module.VALID_API_KEYS = set(valid_keys)

    return app, original_keys


class TestApiKeyMiddlewareDispatch:
    """Cover lines 74-87: ApiKeyMiddleware.dispatch() logic."""

    def test_agent_card_bypasses_auth(self):
        """Line 74-75: /.well-known/agent-card.json is publicly accessible."""
        import a2a_agent.app as app_module
        app, original = _make_starlette_app_with_api_key(["valid-key-1"])
        try:
            client = TestClient(app)
            resp = client.get("/.well-known/agent-card.json")
            assert resp.status_code == 200
            assert resp.json() == {"name": "test-agent"}
        finally:
            app_module.VALID_API_KEYS = original

    def test_missing_api_key_returns_401(self):
        """Lines 77-81: Missing X-API-Key header returns 401 Unauthorized."""
        import a2a_agent.app as app_module
        app, original = _make_starlette_app_with_api_key(["valid-key-1"])
        try:
            client = TestClient(app)
            resp = client.get("/test")  # No X-API-Key header
            assert resp.status_code == 401
            body = resp.json()
            assert body["error"] == "Unauthorized"
            assert "X-API-Key header required" in body["detail"]
        finally:
            app_module.VALID_API_KEYS = original

    def test_invalid_api_key_returns_403(self):
        """Lines 82-86: Invalid API key returns 403 Forbidden."""
        import a2a_agent.app as app_module
        app, original = _make_starlette_app_with_api_key(["valid-key-1"])
        try:
            client = TestClient(app)
            resp = client.get("/test", headers={"X-API-Key": "wrong-key"})
            assert resp.status_code == 403
            body = resp.json()
            assert body["error"] == "Forbidden"
            assert "Invalid API key" in body["detail"]
        finally:
            app_module.VALID_API_KEYS = original

    def test_valid_api_key_passes(self):
        """Line 87: Valid API key allows the request through."""
        import a2a_agent.app as app_module
        app, original = _make_starlette_app_with_api_key(["my-secret-key"])
        try:
            client = TestClient(app)
            resp = client.get("/test", headers={"X-API-Key": "my-secret-key"})
            assert resp.status_code == 200
            assert resp.json() == {"ok": True}
        finally:
            app_module.VALID_API_KEYS = original

    def test_empty_api_key_returns_401(self):
        """Empty string X-API-Key is treated as missing (falsy)."""
        import a2a_agent.app as app_module
        app, original = _make_starlette_app_with_api_key(["valid-key-1"])
        try:
            client = TestClient(app)
            resp = client.get("/test", headers={"X-API-Key": ""})
            # Empty string is falsy in Python → 401
            assert resp.status_code == 401
        finally:
            app_module.VALID_API_KEYS = original
