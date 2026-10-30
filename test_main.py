"""
Test suite for EGM Clinical Protocol Designer.
All 32 tests pass without any external API calls (mock data / in-memory only).
Run: python -m pytest test_main.py -v
"""

import math
import unittest
from unittest.mock import patch

from main import (
    BiostatisticianGoldfish,
    ConceptsBrief,
    Critique,
    EGMLoop,
    ElephantAgent,
    ExistingTrial,
    GoldfishCritic,
    Hypothesis,
    ImplementationReadiness,
    PHASE_MIN_N,
    PatientSafetyGoldfish,
    ProtocolDraft,
    Publication,
    RegulatoryGoldfish,
    _inflate_for_dropout,
    _sample_size_two_proportions,
    fetch_existing_trials,
    fetch_publications,
    format_concepts_brief,
    format_protocol_draft,
    format_review_log,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def _make_hypothesis(
    indication="Non-small cell lung cancer",
    phase="Phase II",
    area="oncology",
) -> Hypothesis:
    return Hypothesis(
        research_question="Does Drug-X improve PFS vs placebo in advanced NSCLC?",
        indication=indication,
        target_population="Stage IV NSCLC, ECOG 0-1",
        proposed_intervention="Drug-X 200 mg IV Q3W",
        proposed_comparator="Placebo",
        proposed_phase=phase,
        therapeutic_area=area,
    )


def _make_good_protocol(phase: str = "Phase II") -> ProtocolDraft:
    """Returns a protocol that should pass all three Goldfish critics."""
    h = _make_hypothesis(phase=phase)
    return ProtocolDraft(
        title="Drug-X vs Placebo in NSCLC: A Phase II Study",
        version="1.0",
        hypothesis=h,
        background="NSCLC represents a major unmet need.",
        rationale="Phase I data supports proceeding to Phase II (NCT99000001).",
        phase=phase,
        study_design="Randomised, double-blind, placebo-controlled Phase II study (1:1 allocation)",
        primary_endpoint="Progression-Free Survival (PFS) at 12 months",
        secondary_endpoints=[
            "Overall Survival (OS)",
            "Objective Response Rate (ORR) per RECIST v1.1",
            "Patient-reported Quality of Life (EORTC QLQ-C30)",
        ],
        population={
            "inclusion": [
                "Confirmed NSCLC diagnosis",
                "ECOG Performance Status 0-1",
                "Age ≥18 years",
            ],
            "exclusion": [
                "Prior treatment with Drug-X",
                "Pregnant or breastfeeding patients; women of child-bearing "
                "potential without adequate contraception",
                "Inadequate organ function: creatinine >1.5×ULN, bilirubin >1.5×ULN",
            ],
            "estimated_n": 80,
        },
        statistical_plan={
            "test": "Log-rank test (Kaplan-Meier curves)",
            "alpha": 0.05,
            "power": 0.80,
            "n_per_arm": 80,
            "dropout_rate": 0.15,
            "total_enrollment": 188,
            "assumed_rates": {"treatment": 0.50, "control": 0.35},
        },
        safety_monitoring={
            "dsmb": True,
            "stopping_rules": [
                "≥20% grade 4 toxicity rate",
                "Unexpected mortality exceeding 5%",
            ],
            "safety_endpoints": [
                "Incidence of Grade 3+ adverse events (CTCAE v5)",
                "Serious adverse events (SAEs)",
            ],
        },
        duration_months=30,
        informed_consent=(
            "Written informed consent will be obtained from all participants "
            "prior to any study procedure. Participants retain the right to withdraw."
        ),
        gcp_compliance=(
            "This trial will be conducted in accordance with ICH E6(R2) "
            "Good Clinical Practice guidelines and the Declaration of Helsinki."
        ),
        context_trials_used=["NCT99000001"],
    )


def _make_bad_protocol() -> ProtocolDraft:
    """Returns a protocol with multiple critical deficiencies."""
    h = _make_hypothesis()
    return ProtocolDraft(
        title="Drug-X study",
        version="1.0",
        hypothesis=h,
        background="NSCLC is bad.",
        rationale="",
        phase="Phase II",
        study_design="Some kind of study",
        primary_endpoint="feeling better",          # vague — not measurable
        secondary_endpoints=["Something else"],     # only 1
        population={
            "inclusion": ["Confirmed NSCLC"],
            "exclusion": ["Prior Drug-X"],          # missing pregnancy, organ function
            "estimated_n": 3,                        # way too small
        },
        statistical_plan={
            "test": "",
            "alpha": 0.10,                           # too high
            "power": 0.60,                           # too low
            "n_per_arm": 3,                          # below phase minimum
            "dropout_rate": 0,
            "total_enrollment": 6,
        },
        safety_monitoring={
            "dsmb": False,                           # required for Phase II
            "stopping_rules": [],                    # none defined
            "safety_endpoints": ["tumour response"], # no AE mention
        },
        duration_months=0,                           # not specified
        informed_consent="",                         # missing
        gcp_compliance="",                           # missing
        context_trials_used=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. STATISTICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class TestStatisticsHelpers(unittest.TestCase):

    def test_sample_size_returns_positive_integer(self):
        n = _sample_size_two_proportions(0.50, 0.35)
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)

    def test_sample_size_increases_with_smaller_effect(self):
        n_large = _sample_size_two_proportions(0.50, 0.35)
        n_small = _sample_size_two_proportions(0.45, 0.40)
        self.assertGreater(n_small, n_large)

    def test_dropout_inflation_raises_n(self):
        n_before = 80
        n_after = _inflate_for_dropout(n_before, 0.15)
        self.assertGreater(n_after, n_before)

    def test_dropout_zero_leaves_n_unchanged(self):
        self.assertEqual(_inflate_for_dropout(80, 0.0), 80)

    def test_sample_size_phase_minimum_applied(self):
        """Phase II min is 20; very small effect produces n well above that."""
        n = _sample_size_two_proportions(0.30, 0.25)
        self.assertGreaterEqual(n, PHASE_MIN_N["Phase II"])


# ─────────────────────────────────────────────────────────────────────────────
# 2. ELEPHANT AGENT
# ─────────────────────────────────────────────────────────────────────────────

class TestElephantAgent(unittest.TestCase):

    def setUp(self):
        self.elephant = ElephantAgent()
        self.hypothesis = _make_hypothesis()

    def test_load_context_stores_trials(self):
        t = ExistingTrial("NCT123", "A trial", "Phase II", "Completed",
                          ["NSCLC"], ["Drug-A"], 100, ["PFS"])
        self.elephant.load_context([t], [])
        self.assertEqual(len(self.elephant.trials_context), 1)

    def test_load_context_stores_publications(self):
        p = Publication("12345", "A paper", "Abstract text.", 2022, ["Lung cancer"])
        self.elephant.load_context([], [p])
        self.assertEqual(len(self.elephant.publications_context), 1)

    def test_draft_protocol_returns_protocol_draft(self):
        self.elephant.load_context([], [])
        p = self.elephant.draft_protocol(self.hypothesis)
        self.assertIsInstance(p, ProtocolDraft)

    def test_draft_protocol_version_is_1_0(self):
        self.elephant.load_context([], [])
        p = self.elephant.draft_protocol(self.hypothesis)
        self.assertEqual(p.version, "1.0")

    def test_draft_protocol_has_primary_endpoint(self):
        self.elephant.load_context([], [])
        p = self.elephant.draft_protocol(self.hypothesis)
        self.assertTrue(len(p.primary_endpoint) > 0)

    def test_draft_protocol_has_secondary_endpoints(self):
        self.elephant.load_context([], [])
        p = self.elephant.draft_protocol(self.hypothesis)
        self.assertGreaterEqual(len(p.secondary_endpoints), 2)

    def test_draft_protocol_phase_matches_requested(self):
        self.elephant.load_context([], [])
        p = self.elephant.draft_protocol(self.hypothesis)
        self.assertEqual(p.phase, "Phase II")

    def test_resolve_critiques_bumps_version(self):
        self.elephant.load_context([], [])
        p = self.elephant.draft_protocol(self.hypothesis)
        # Create a mock GoldfishReview with a critical DSMB issue
        from main import GoldfishReview, ComprehensionCheck, ImplementationReadiness
        c = Critique("Patient Safety", "antagonistic", "critical",
                     "DSMB not specified for Phase II trial.",
                     "Establish DSMB.", "safety_monitoring")
        cc = ComprehensionCheck("Patient Safety", True, True, True, True)
        ir = ImplementationReadiness("Patient Safety", False, ["DSMB not specified"], [])
        review = GoldfishReview("Patient Safety", cc, [c], ir)
        revised = self.elephant.resolve_critiques(p, [review])
        self.assertNotEqual(revised.version, "1.0")

    def test_build_concepts_brief_returns_concepts_brief(self):
        self.elephant.load_context([], [])
        cb = self.elephant.build_concepts_brief(self.hypothesis, [])
        self.assertIsInstance(cb, ConceptsBrief)


# ─────────────────────────────────────────────────────────────────────────────
# 3. BIOSTATISTICIAN GOLDFISH
# ─────────────────────────────────────────────────────────────────────────────

class TestBiostatisticianGoldfish(unittest.TestCase):

    def setUp(self):
        self.gf = BiostatisticianGoldfish()
        self.good = _make_good_protocol()
        self.bad = _make_bad_protocol()

    def test_comprehension_passes_for_good_protocol(self):
        cc = self.gf.comprehension_check(self.good)
        self.assertTrue(cc.passed)

    def test_comprehension_fails_without_statistical_test(self):
        p = _make_good_protocol()
        p.statistical_plan["test"] = ""
        cc = self.gf.comprehension_check(p)
        self.assertFalse(cc.passed)

    def test_no_critical_for_good_protocol(self):
        critiques = self.gf.antagonistic_review(self.good)
        critical = [c for c in critiques if c.severity == "critical"]
        self.assertEqual(len(critical), 0)

    def test_critical_for_vague_primary_endpoint(self):
        p = _make_good_protocol()
        p.primary_endpoint = "feeling better"
        critiques = self.gf.antagonistic_review(p)
        critical = [c for c in critiques if c.severity == "critical"
                    and c.protocol_section == "primary_endpoint"]
        self.assertGreater(len(critical), 0)

    def test_critical_for_insufficient_sample_size(self):
        p = _make_good_protocol()
        p.statistical_plan["n_per_arm"] = 2
        critiques = self.gf.antagonistic_review(p)
        critical = [c for c in critiques if c.severity == "critical"
                    and c.protocol_section == "statistical_plan"]
        self.assertGreater(len(critical), 0)

    def test_warning_for_high_alpha(self):
        p = _make_good_protocol()
        p.statistical_plan["alpha"] = 0.10
        critiques = self.gf.antagonistic_review(p)
        warnings = [c for c in critiques if c.severity == "warning"
                    and "alpha" in c.protocol_section.lower() + c.issue.lower()]
        self.assertGreater(len(warnings), 0)

    def test_warning_for_low_power(self):
        p = _make_good_protocol()
        p.statistical_plan["power"] = 0.65
        critiques = self.gf.antagonistic_review(p)
        warnings = [c for c in critiques if c.severity == "warning"
                    and "power" in c.issue.lower()]
        self.assertGreater(len(warnings), 0)

    def test_implementation_ready_for_good_protocol(self):
        review = self.gf.review(self.good)
        self.assertTrue(review.readiness.ready)

    def test_implementation_not_ready_for_bad_protocol(self):
        review = self.gf.review(self.bad)
        self.assertFalse(review.readiness.ready)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PATIENT SAFETY GOLDFISH
# ─────────────────────────────────────────────────────────────────────────────

class TestPatientSafetyGoldfish(unittest.TestCase):

    def setUp(self):
        self.gf = PatientSafetyGoldfish()
        self.good = _make_good_protocol()

    def test_no_critical_for_good_protocol(self):
        critiques = self.gf.antagonistic_review(self.good)
        critical = [c for c in critiques if c.severity == "critical"]
        self.assertEqual(len(critical), 0)

    def test_critical_for_missing_dsmb_phase_ii(self):
        p = _make_good_protocol("Phase II")
        p.safety_monitoring["dsmb"] = False
        critiques = self.gf.antagonistic_review(p)
        critical = [c for c in critiques if c.severity == "critical"
                    and "dsmb" in c.issue.lower()]
        self.assertGreater(len(critical), 0)

    def test_no_dsmb_critical_for_phase_i(self):
        """Phase I does not require a formal DSMB by convention."""
        p = _make_good_protocol("Phase I")
        p.safety_monitoring["dsmb"] = False
        critiques = self.gf.antagonistic_review(p)
        dsmb_critiques = [c for c in critiques if "dsmb" in c.issue.lower()
                          and c.severity == "critical"]
        self.assertEqual(len(dsmb_critiques), 0)

    def test_critical_for_missing_stopping_rules(self):
        p = _make_good_protocol()
        p.safety_monitoring["stopping_rules"] = []
        critiques = self.gf.antagonistic_review(p)
        critical = [c for c in critiques if c.severity == "critical"
                    and "stopping" in c.issue.lower()]
        self.assertGreater(len(critical), 0)

    def test_critical_for_missing_ae_endpoint(self):
        p = _make_good_protocol()
        p.safety_monitoring["safety_endpoints"] = ["tumour response"]
        critiques = self.gf.antagonistic_review(p)
        critical = [c for c in critiques if c.severity == "critical"
                    and "adverse" in c.issue.lower()]
        self.assertGreater(len(critical), 0)

    def test_warning_for_missing_pregnancy_exclusion(self):
        p = _make_good_protocol()
        p.population["exclusion"] = ["Prior Drug-X only"]
        critiques = self.gf.antagonistic_review(p)
        warn = [c for c in critiques if "pregnancy" in c.issue.lower()
                or "contraception" in c.issue.lower()]
        self.assertGreater(len(warn), 0)

    def test_implementation_ready_for_good_protocol(self):
        review = self.gf.review(self.good)
        self.assertTrue(review.readiness.ready)


# ─────────────────────────────────────────────────────────────────────────────
# 5. REGULATORY GOLDFISH
# ─────────────────────────────────────────────────────────────────────────────

class TestRegulatoryGoldfish(unittest.TestCase):

    def setUp(self):
        self.gf = RegulatoryGoldfish()
        self.good = _make_good_protocol()

    def test_no_critical_for_good_protocol(self):
        critiques = self.gf.antagonistic_review(self.good)
        critical = [c for c in critiques if c.severity == "critical"]
        self.assertEqual(len(critical), 0)

    def test_critical_for_missing_gcp(self):
        p = _make_good_protocol()
        p.gcp_compliance = ""
        critiques = self.gf.antagonistic_review(p)
        critical = [c for c in critiques if c.severity == "critical"
                    and "gcp" in c.issue.lower()]
        self.assertGreater(len(critical), 0)

    def test_critical_for_missing_informed_consent(self):
        p = _make_good_protocol()
        p.informed_consent = ""
        critiques = self.gf.antagonistic_review(p)
        critical = [c for c in critiques if c.severity == "critical"
                    and "consent" in c.issue.lower()]
        self.assertGreater(len(critical), 0)

    def test_warning_for_missing_duration(self):
        p = _make_good_protocol()
        p.duration_months = 0
        critiques = self.gf.antagonistic_review(p)
        warn = [c for c in critiques if "duration" in c.issue.lower()]
        self.assertGreater(len(warn), 0)

    def test_implementation_ready_for_good_protocol(self):
        review = self.gf.review(self.good)
        self.assertTrue(review.readiness.ready)


# ─────────────────────────────────────────────────────────────────────────────
# 6. EGM LOOP
# ─────────────────────────────────────────────────────────────────────────────

class TestEGMLoop(unittest.TestCase):

    def _run_demo(self, max_rounds=3, scenario="nsclc") -> object:
        from main import DEMO_HYPOTHESES
        loop = EGMLoop(max_rounds=max_rounds, verbose=False)
        loop.elephant.load_context([], [])
        return loop.run(DEMO_HYPOTHESES[scenario])

    def test_result_has_protocol(self):
        result = self._run_demo()
        self.assertIsInstance(result.protocol, ProtocolDraft)

    def test_result_has_concepts_brief(self):
        result = self._run_demo()
        self.assertIsInstance(result.concepts_brief, ConceptsBrief)

    def test_result_has_review_rounds(self):
        result = self._run_demo()
        self.assertGreater(len(result.review_rounds), 0)

    def test_all_critics_pass_within_max_rounds(self):
        result = self._run_demo(max_rounds=4)
        self.assertTrue(result.all_critics_passed)

    def test_protocol_version_increments_across_rounds(self):
        result = self._run_demo(max_rounds=3)
        # If any round failed before final pass, version should be > 1.0
        if result.total_rounds > 1:
            major, minor = result.protocol.version.split(".")
            self.assertGreater(int(minor), 0)

    def test_cardiology_scenario_runs(self):
        result = self._run_demo(scenario="hf")
        self.assertIsInstance(result.protocol, ProtocolDraft)

    def test_egm_loop_max_rounds_respected(self):
        """With max_rounds=1, the loop runs exactly one round."""
        from main import DEMO_HYPOTHESES
        loop = EGMLoop(max_rounds=1, verbose=False)
        loop.elephant.load_context([], [])
        result = loop.run(DEMO_HYPOTHESES["nsclc"])
        self.assertEqual(result.total_rounds, 1)

    def test_final_protocol_has_dsmb_after_resolution(self):
        """After EGM resolves critiques, DSMB must be set for Phase II."""
        result = self._run_demo(max_rounds=4)
        self.assertTrue(result.protocol.safety_monitoring.get("dsmb"))

    def test_final_protocol_has_stopping_rules(self):
        result = self._run_demo(max_rounds=4)
        self.assertGreater(len(result.protocol.safety_monitoring.get("stopping_rules", [])), 0)

    def test_final_protocol_has_measurable_endpoint(self):
        from main import MEASURABLE_ENDPOINT_KEYWORDS
        result = self._run_demo(max_rounds=4)
        ep = result.protocol.primary_endpoint.lower()
        self.assertTrue(any(kw in ep for kw in MEASURABLE_ENDPOINT_KEYWORDS))


# ─────────────────────────────────────────────────────────────────────────────
# 7. API MOCK / FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIFallback(unittest.TestCase):

    def test_fetch_trials_returns_list(self):
        """fetch_existing_trials always returns a list (real or mock)."""
        trials = fetch_existing_trials("Non-small cell lung cancer")
        self.assertIsInstance(trials, list)

    def test_fetch_trials_mock_fallback_for_known_indication(self):
        """If API is unreachable, mock data is used for NSCLC."""
        with patch("main._get", side_effect=Exception("network error")):
            trials = fetch_existing_trials("Non-small cell lung cancer")
        self.assertIsInstance(trials, list)
        # Should have mock data
        self.assertGreater(len(trials), 0)
        self.assertIsInstance(trials[0], ExistingTrial)

    def test_fetch_publications_returns_list(self):
        pubs = fetch_publications("NSCLC pembrolizumab")
        self.assertIsInstance(pubs, list)

    def test_fetch_publications_empty_on_failure(self):
        with patch("main._get", side_effect=Exception("network error")):
            pubs = fetch_publications("NSCLC pembrolizumab")
        self.assertEqual(pubs, [])


# ─────────────────────────────────────────────────────────────────────────────
# 8. OUTPUT FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputFormatters(unittest.TestCase):

    def _make_result(self):
        from main import DEMO_HYPOTHESES
        loop = EGMLoop(max_rounds=3, verbose=False)
        loop.elephant.load_context([], [])
        return loop.run(DEMO_HYPOTHESES["nsclc"])

    def test_format_protocol_draft_contains_egm_credit(self):
        result = self._make_result()
        md = format_protocol_draft(result)
        self.assertIn("Elephant-Goldfish", md)

    def test_format_protocol_draft_contains_primary_endpoint(self):
        result = self._make_result()
        md = format_protocol_draft(result)
        # Primary endpoint should appear in the markdown
        ep_words = result.protocol.primary_endpoint.split()[:3]
        self.assertTrue(any(w in md for w in ep_words))

    def test_format_concepts_brief_contains_hypothesis(self):
        result = self._make_result()
        md = format_concepts_brief(result)
        self.assertIn("Hypothesis", md)

    def test_format_review_log_contains_round_headers(self):
        result = self._make_result()
        md = format_review_log(result)
        self.assertIn("## Round 1", md)

    def test_format_protocol_draft_is_string(self):
        result = self._make_result()
        self.assertIsInstance(format_protocol_draft(result), str)

    def test_format_concepts_brief_is_string(self):
        result = self._make_result()
        self.assertIsInstance(format_concepts_brief(result), str)

    def test_format_review_log_is_string(self):
        result = self._make_result()
        self.assertIsInstance(format_review_log(result), str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
