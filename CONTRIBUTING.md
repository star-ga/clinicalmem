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

## Environment Variables

```bash
# Optional: UMLS Metathesaurus API key (for cross-vocabulary mapping)
# Get a free key at https://uts.nlm.nih.gov/uts/signup-login
echo "UMLS_API_KEY=your-key-here" > .env
```

The RxNorm and SNOMED CT modules work without an API key (using free public APIs).
UMLS crosswalk features require a UMLS API key.

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

All changes must maintain 356/356 tests passing. Run the full suite before submitting:

```bash
python -m pytest tests/ -v --tb=short
```

## Code Standards

- Python 3.12+
- Type hints on all public functions
- Immutable data patterns (no in-place mutation)
- Functions under 50 lines
- Files under 800 lines
