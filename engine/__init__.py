"""ClinicalMem Engine — clinical memory powered by mind-mem.

Modules:
- clinical_memory: Core FHIR-to-memory pipeline with hybrid search
- clinical_scoring: MIND kernel-inspired scoring (abstention, importance, adversarial)
- consensus_engine: Multi-LLM consensus verification (GPT-5.4, MedGemma, Gemini)
- fda_client: openFDA drug safety alerts (FAERS, labels, recalls)
- fhir_client: FHIR R4 REST client with SSRF protection
- hallucination_detector: Evidence grounding gate — FHIR-traced claim verification
- llm_synthesizer: Evidence-grounded clinical narrative generation
- phi_detector: PHI detection and redaction (HIPAA categories)
- rxnorm_client: RxNorm drug normalization and interaction lookup (NIH API)
- snomed_client: SNOMED CT coded clinical terminology
- trials_client: ClinicalTrials.gov trial matching (API v2)
- umls_mapper: UMLS cross-vocabulary concept mapping
- what_if: Digital twin what-if scenario simulation
"""
