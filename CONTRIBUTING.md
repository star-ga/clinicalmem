# Contributing to ClinicalMem

ClinicalMem is built by [STARGA Inc.](https://star.ga) for the [Agents Assemble Healthcare AI Hackathon](https://agents-assemble.devpost.com/).

## Development Setup

```bash
# Clone
git clone https://github.com/star-ga/clinicalmem.git
cd clinicalmem

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v
```

## Running Locally

```bash
# MCP Server
python -m mcp_server

# A2A Agent
uvicorn a2a_agent.app:a2a_app --host 0.0.0.0 --port 8001

# Both via Docker
docker compose up --build
```

## Test Coverage

All changes must maintain 90/90 tests passing. Run the full suite before submitting:

```bash
python -m pytest tests/ -v --tb=short
```

## Code Standards

- Python 3.12+
- Type hints on all public functions
- Immutable data patterns (no in-place mutation)
- Functions under 50 lines
- Files under 800 lines
