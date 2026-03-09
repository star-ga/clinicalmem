"""ClinicalMem Engine — clinical memory powered by mind-mem.

Modules:
- clinical_memory: Core FHIR-to-memory pipeline with hybrid search
- clinical_scoring: MIND kernel-inspired scoring (abstention, importance, adversarial)
- fhir_client: FHIR R4 REST client with SSRF protection
- llm_synthesizer: Evidence-grounded clinical narrative generation
- rxnorm_client: RxNorm drug normalization and interaction lookup (NIH API)
- snomed_client: SNOMED CT coded clinical terminology
- umls_mapper: UMLS cross-vocabulary concept mapping
"""
