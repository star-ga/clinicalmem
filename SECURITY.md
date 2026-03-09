# Security Policy

## Synthetic Data Only

ClinicalMem uses **synthetic patient data exclusively**. No Protected Health Information (PHI) is stored, processed, or transmitted. All patient records (including the Sarah Mitchell demo) are fictional test fixtures.

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it responsibly:

- **Email:** noreply@star.ga
- **Do not** open a public GitHub issue for security vulnerabilities

## Security Measures

### Authentication
- A2A Agent requires API key authentication via `X-API-Key` header
- MCP Server uses SHARP-on-MCP header-based context delivery

### API Key Management
- UMLS API key loaded from environment variable (`UMLS_API_KEY`), never hardcoded
- `.env` file is gitignored; secrets stored in Azure Key Vault for production
- RxNorm and SNOMED CT modules degrade gracefully without API keys

### Network Security
- FHIR client validates server URLs against an allowlist (SSRF protection)
- All external API calls use HTTPS with timeout limits
- Docker containers run as non-root where possible

### Audit Trail
- Every clinical decision is logged in a SHA-256 hash-chained audit trail
- Merkle chain provides tamper-evident integrity verification
- Audit entries include timestamps, operation types, and evidence block IDs

### Dependencies
- All dependencies are pinned to minimum versions in `pyproject.toml`
- CI runs on every push to verify test integrity
