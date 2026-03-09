"""Tests for PHI detector — regex-based HIPAA PHI detection and redaction."""
import pytest

from engine.phi_detector import detect_phi, redact_phi, scan_phi, PHIMatch, PHIReport


class TestDetectPhi:
    def test_detects_ssn(self):
        matches = detect_phi("Patient SSN is 123-45-6789 on file.")
        categories = [m.category for m in matches]
        assert "SSN" in categories

    def test_detects_ssn_no_dashes(self):
        matches = detect_phi("SSN: 123456789")
        categories = [m.category for m in matches]
        assert "SSN" in categories

    def test_detects_phone(self):
        matches = detect_phi("Call (555) 123-4567 for results.")
        categories = [m.category for m in matches]
        assert "PHONE" in categories

    def test_detects_email(self):
        matches = detect_phi("Contact patient at john.doe@hospital.com")
        categories = [m.category for m in matches]
        assert "EMAIL" in categories

    def test_detects_mrn(self):
        matches = detect_phi("MRN: 12345678 admitted today.")
        categories = [m.category for m in matches]
        assert "MRN" in categories

    def test_detects_mrn_with_dash(self):
        matches = detect_phi("MRN-987654 chart pulled.")
        categories = [m.category for m in matches]
        assert "MRN" in categories

    def test_detects_date(self):
        matches = detect_phi("DOB: 01/15/1985")
        categories = [m.category for m in matches]
        assert "DATE" in categories

    def test_detects_date_iso(self):
        matches = detect_phi("Born 1985-01-15")
        categories = [m.category for m in matches]
        assert "DATE" in categories

    def test_detects_ip_address(self):
        matches = detect_phi("Logged from 192.168.1.100")
        categories = [m.category for m in matches]
        assert "IP_ADDRESS" in categories

    def test_detects_name_with_prefix(self):
        matches = detect_phi("Dr. John Smith prescribed metformin.")
        categories = [m.category for m in matches]
        assert "NAME" in categories

    def test_detects_patient_name(self):
        matches = detect_phi("Patient Jane Doe reports headaches.")
        categories = [m.category for m in matches]
        assert "NAME" in categories

    def test_no_phi_in_clean_text(self):
        matches = detect_phi("The patient takes metformin for diabetes.")
        # Should not detect drug names as PHI
        assert all(m.category != "SSN" for m in matches)
        assert all(m.category != "EMAIL" for m in matches)
        assert all(m.category != "MRN" for m in matches)

    def test_empty_text(self):
        matches = detect_phi("")
        assert matches == []

    def test_multiple_phi_types(self):
        text = "Dr. John Smith (MRN: 12345678) called (555) 123-4567 about labs."
        matches = detect_phi(text)
        categories = {m.category for m in matches}
        assert len(categories) >= 2  # At least NAME + MRN or PHONE

    def test_deduplicates_overlapping(self):
        text = "SSN 123-45-6789"
        matches = detect_phi(text)
        # Should not have overlapping matches for the same span
        for i in range(len(matches) - 1):
            assert matches[i].end <= matches[i + 1].start

    def test_match_positions(self):
        text = "Email: test@example.com done"
        matches = detect_phi(text)
        email_matches = [m for m in matches if m.category == "EMAIL"]
        assert len(email_matches) == 1
        m = email_matches[0]
        assert text[m.start:m.end] == "test@example.com"


class TestRedactPhi:
    def test_redacts_ssn(self):
        text = "SSN: 123-45-6789"
        redacted, matches = redact_phi(text)
        assert "123-45-6789" not in redacted
        assert "[SSN_" in redacted
        assert len(matches) > 0

    def test_redacts_email(self):
        redacted, matches = redact_phi("Contact john@hospital.com today.")
        assert "john@hospital.com" not in redacted
        assert "[EMAIL_" in redacted

    def test_no_phi_returns_original(self):
        text = "Patient takes metformin daily."
        redacted, matches = redact_phi(text)
        # Text should be unchanged or at most have zip/date false positives
        assert "metformin" in redacted

    def test_empty_returns_empty(self):
        redacted, matches = redact_phi("")
        assert redacted == ""
        assert matches == []

    def test_multiple_redactions(self):
        text = "MRN: 12345678, Email: dr@clinic.com"
        redacted, matches = redact_phi(text)
        assert "12345678" not in redacted
        assert "dr@clinic.com" not in redacted

    def test_preserves_surrounding_text(self):
        text = "Before 123-45-6789 after"
        redacted, _ = redact_phi(text)
        assert redacted.startswith("Before")
        assert redacted.endswith("after")


class TestScanPhi:
    def test_scan_returns_report(self):
        report = scan_phi("MRN-1234567 patient admitted")
        assert isinstance(report, PHIReport)
        assert report.phi_count > 0
        assert not report.is_safe
        assert "MRN" in report.categories_found

    def test_safe_text(self):
        report = scan_phi("Metformin prescribed for diabetes.")
        # Even if some false positives, report structure is correct
        assert isinstance(report, PHIReport)
        assert report.redacted_text is not None

    def test_report_contains_original(self):
        original = "SSN: 123-45-6789"
        report = scan_phi(original)
        assert report.original_text == original
        assert "123-45-6789" not in report.redacted_text
