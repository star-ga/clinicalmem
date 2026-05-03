# IRB Exemption Basis — ClinicalMem Synthea Demo Cohort

**Document date:** 2026-05-03
**File described:** `docs/synthea_demo_cohort.json`
**Project:** ClinicalMem — A2A Clinical Decision Support (Google A2A Hackathon 2026)
**Organization:** STARGA, Inc.

---

## Summary

The synthetic patient cohort in `synthea_demo_cohort.json` does not constitute human subjects research and therefore does not require Institutional Review Board (IRB) review or approval under the U.S. Department of Health and Human Services (DHHS) Common Rule.

---

## Regulatory Basis

The Federal Policy for the Protection of Human Subjects (the "Common Rule"), codified at 45 C.F.R. § 46, applies only to **research involving human subjects**. Under 45 C.F.R. § 46.102(e)(1), a "human subject" is defined as:

> "a living individual about whom an investigator (whether professional or student) conducting research: (i) Obtains information or biospecimens through intervention or interaction with the individual; or (ii) Obtains, uses, studies, analyzes, or generates **identifiable private information** or **identifiable biospecimens**."

The cohort in `synthea_demo_cohort.json` does not meet this definition for the following reasons:

### 1. No Real Individuals

All 13 patients are entirely fictional, computer-generated characters. Their names (e.g., "Raymond Okafor", "Sandra Kowalski", "Adelaide Vasquez", "Walter Hawthorne"), dates of birth, addresses, and medical histories were constructed solely for demonstration purposes using synthetic identifiers. No real person corresponds to any entry.

### 2. No Real Protected Health Information (PHI)

Under the HIPAA Privacy Rule (45 C.F.R. § 164.514), de-identification requires that 18 enumerated identifiers be absent. This cohort goes further: it contains no actual health information in the first place. Every clinical value (diagnoses, medication doses, lab results) is fabricated to illustrate archetypal drug-drug interaction scenarios drawn from published pharmacology literature — not sourced from any individual's medical record.

All MRN values use the prefix `SYN-MRN-` and are not drawn from any real EHR system. No Social Security Numbers, real NPIs, insurance identifiers, or device identifiers are present.

### 3. No Real NPIs

All Practitioner NPI values were generated deterministically using `engine.npi_registry.generate_test_npi()` with labelled seed strings (e.g., `"synthea-cardio-1"`). Each NPI passes the CMS Luhn check digit algorithm (documented at the NPI checkdigit specification) and is structurally valid but is NOT registered in the CMS NPPES database. Each Practitioner resource is explicitly tagged `_meta.npi_source = "DEMO_LUHN_GENERATED"` to prevent confusion with real providers.

### 4. Synthea-Style Generation Pattern

The cohort follows the Synthea open-source synthetic patient generator approach (https://github.com/synthetichealth/synthea), which is the recognized standard for generating FHIR-compliant fictional patient data for healthcare software testing. Synthea data is explicitly designed to be shareable without privacy constraints. Our cohort uses real SNOMED CT codes, RxNorm codes, and LOINC codes (publicly licensed terminologies) applied to fictional patients.

### 5. Purpose Is Software Demonstration, Not Human Subjects Research

The purpose of this cohort is to demonstrate the capabilities of the ClinicalMem clinical decision support system at a software hackathon. It is not designed to draw conclusions about any population, test any hypothesis about human health, or generate generalizable knowledge about human subjects. Under 45 C.F.R. § 46.102(l), "research" means "a systematic investigation, including research development, testing, and evaluation, designed to develop or contribute to generalizable knowledge." Software testing with synthetic data does not meet this definition.

---

## Conclusion

Because the cohort contains no data obtained from real living individuals, no identifiable private information, and no real PHI, the activity does not constitute "research involving human subjects" under 45 C.F.R. § 46.102(e). IRB review is not required.

This document is provided for transparency and audit purposes. If the ClinicalMem system were ever to be evaluated using real patient data (de-identified or otherwise), a new IRB determination would be required prior to that activity.

---

**License:** Apache-2.0
**Author:** STARGA, Inc.
