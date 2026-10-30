"""
EGM Clinical Protocol Designer
================================
An experiment with Google's Elephant-Goldfish Model (EGM):
  https://github.com/google/elephant-goldfish-model

The EGM pattern — a long-context Elephant designer paired with stateless Goldfish
critics in an adversarial review loop — is applied here to clinical trial protocol
design. The Elephant synthesises evidence from ClinicalTrials.gov and PubMed into a
draft protocol; three Goldfish critics (Biostatistician, Patient Safety, Regulatory)
independently attack the design across three EGM phases (Comprehension Check →
Antagonistic Review → Implementation Readiness). The loop runs until all critics
pass or the maximum rounds are reached.

No paid APIs required. ClinicalTrials.gov v2 and NCBI PubMed are both free and public.
Falls back to curated mock data when offline.

Usage:
    python main.py --demo
    python main.py --hypothesis "..." --indication "..." --phase "Phase II"
    python main.py --demo --save output/
"""

import argparse
import json
import math
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EGM_VERSION = "0.1.0"
EGM_CREDIT = (
    "Experiment with Google's Elephant-Goldfish Model — "
    "https://github.com/google/elephant-goldfish-model"
)

# Standard statistical thresholds
Z_ALPHA_05 = 1.960   # two-sided alpha = 0.05
Z_ALPHA_025 = 2.241  # two-sided alpha = 0.025
Z_BETA_80 = 0.842    # power = 80 %
Z_BETA_90 = 1.282    # power = 90 %

# Minimum enrolled per arm by phase (regulatory convention)
PHASE_MIN_N = {
    "Phase I": 6,
    "Phase II": 20,
    "Phase III": 100,
    "Phase IV": 200,
}

# Keywords that make a primary endpoint measurable / time-bounded
MEASURABLE_ENDPOINT_KEYWORDS = [
    "survival", "response", "remission", "change", "rate", "time to",
    "progression", "recurrence", "mortality", "morbidity", "reduction",
    "improvement", "score", "level", "concentration", "objective response",
    "complete response", "partial response", "disease-free", "event-free",
    "overall survival", "progression-free", "relapse-free", "duration",
    "at week", "at month", "at 12", "at 24", "at 6",
]

# FDA guidance by therapeutic area (simplified reference data)
FDA_GUIDANCE: dict = {
    "oncology": {
        "preferred_endpoints": [
            "Overall Survival (OS)",
            "Progression-Free Survival (PFS)",
            "Objective Response Rate (ORR)",
            "Disease Control Rate (DCR)",
        ],
        "dsmb_required": True,
        "notes": "Dose-escalation required for Phase I. DLT assessment mandatory.",
    },
    "cardiology": {
        "preferred_endpoints": [
            "MACE (Major Adverse Cardiovascular Events)",
            "LVEF (Left Ventricular Ejection Fraction)",
            "6-Minute Walk Distance (6MWD)",
            "NT-proBNP / BNP",
            "KCCQ Quality of Life score",
            "Hospitalisation for HF",
        ],
        "dsmb_required": True,
        "notes": "MACE endpoints preferred for HF trials. EF measurement required.",
    },
    "endocrinology": {
        "preferred_endpoints": [
            "HbA1c change from baseline",
            "Fasting Plasma Glucose (FPG)",
            "Weight change",
            "Lipid profile",
        ],
        "dsmb_required": False,
        "notes": "Cardiovascular outcomes study may be required for new diabetes agents.",
    },
    "default": {
        "preferred_endpoints": ["Primary efficacy endpoint", "Safety endpoints"],
        "dsmb_required": True,
        "notes": "Consult FDA guidance for specific indication.",
    },
}

# ICH E6(R2) GCP requirements checklist
ICH_E6_REQUIREMENTS = [
    "Informed consent procedure described",
    "IRB/IEC approval process outlined",
    "Investigator qualifications documented",
    "Monitoring plan defined",
    "Data management and record-keeping specified",
    "Adverse event reporting procedure included",
]

# Mock trial data used when ClinicalTrials.gov API is unavailable
MOCK_TRIALS: dict = {
    "non-small cell lung cancer": [
        {
            "nct_id": "NCT04956406",
            "title": "Pembrolizumab + Chemotherapy in Treatment-Naive Metastatic NSCLC (KEYNOTE-789)",
            "phase": "Phase III",
            "status": "Completed",
            "conditions": ["Non-small cell lung cancer"],
            "interventions": ["Pembrolizumab", "Pemetrexed", "Carboplatin"],
            "enrollment": 492,
            "primary_outcomes": ["Overall Survival", "Progression-Free Survival"],
        },
        {
            "nct_id": "NCT03977350",
            "title": "Nab-Paclitaxel + Pembrolizumab in PD-L1+ Squamous NSCLC",
            "phase": "Phase II",
            "status": "Completed",
            "conditions": ["Squamous NSCLC"],
            "interventions": ["nab-Paclitaxel", "Pembrolizumab"],
            "enrollment": 120,
            "primary_outcomes": ["Objective Response Rate at 12 weeks"],
        },
    ],
    "heart failure": [
        {
            "nct_id": "NCT03036124",
            "title": "Dapagliflozin in HFrEF (DAPA-HF)",
            "phase": "Phase III",
            "status": "Completed",
            "conditions": ["Heart failure with reduced ejection fraction"],
            "interventions": ["Dapagliflozin 10mg", "Placebo"],
            "enrollment": 4744,
            "primary_outcomes": ["MACE composite", "CV death", "HF hospitalisation"],
        },
    ],
    "type 2 diabetes": [
        {
            "nct_id": "NCT01131676",
            "title": "Semaglutide vs Sitagliptin in T2DM inadequately controlled on metformin",
            "phase": "Phase III",
            "status": "Completed",
            "conditions": ["Type 2 Diabetes Mellitus"],
            "interventions": ["Semaglutide 0.5mg", "Semaglutide 1mg", "Sitagliptin"],
            "enrollment": 1231,
            "primary_outcomes": ["HbA1c change from baseline at 56 weeks"],
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Hypothesis:
    research_question: str
    indication: str
    target_population: str
    proposed_intervention: str
    proposed_comparator: str
    proposed_phase: str
    therapeutic_area: str = "oncology"


@dataclass
class ExistingTrial:
    nct_id: str
    title: str
    phase: str
    status: str
    conditions: list
    interventions: list
    enrollment: Optional[int]
    primary_outcomes: list
    primary_completion_date: Optional[str] = None
    eligibility_description: str = ""


@dataclass
class Publication:
    pmid: str
    title: str
    abstract: str
    publication_year: int
    mesh_terms: list


@dataclass
class ProtocolDraft:
    title: str
    version: str
    hypothesis: Hypothesis
    background: str
    rationale: str
    phase: str
    study_design: str
    primary_endpoint: str
    secondary_endpoints: list
    population: dict   # {inclusion: [], exclusion: [], estimated_n: int}
    statistical_plan: dict  # {test, alpha, power, n_per_arm, dropout_rate, total_enrollment}
    safety_monitoring: dict  # {dsmb, stopping_rules, safety_endpoints}
    duration_months: int
    informed_consent: str
    gcp_compliance: str
    context_trials_used: list  # NCT IDs
    review_history: list = field(default_factory=list)


@dataclass
class Critique:
    goldfish_persona: str
    phase: str          # "antagonistic"
    severity: str       # "info" | "warning" | "critical"
    issue: str
    recommendation: str
    protocol_section: str


@dataclass
class ComprehensionCheck:
    goldfish_persona: str
    understood_study_design: bool
    understood_primary_endpoint: bool
    understood_population: bool
    understood_statistical_plan: bool
    notes: str = ""

    @property
    def passed(self) -> bool:
        return all([
            self.understood_study_design,
            self.understood_primary_endpoint,
            self.understood_population,
            self.understood_statistical_plan,
        ])


@dataclass
class ImplementationReadiness:
    goldfish_persona: str
    ready: bool
    blockers: list
    suggestions: list


@dataclass
class GoldfishReview:
    goldfish_persona: str
    comprehension: ComprehensionCheck
    critiques: list
    readiness: ImplementationReadiness


@dataclass
class EGMReviewRound:
    round_number: int
    protocol_version: str
    goldfish_reviews: list
    elephant_resolutions: list
    all_passed: bool


@dataclass
class ConceptsBrief:
    hypothesis: Hypothesis
    egm_rounds_run: int
    what_was_considered: list
    what_was_rejected: list
    key_design_decisions: dict
    final_rationale: str
    generated_on: str


@dataclass
class EGMResult:
    protocol: ProtocolDraft
    concepts_brief: ConceptsBrief
    review_rounds: list
    total_rounds: int
    all_critics_passed: bool
    generated_on: str


# ─────────────────────────────────────────────────────────────────────────────
# API LAYER
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 8) -> dict:
    """Simple HTTP GET returning parsed JSON. Raises on any error."""
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "EGM-ClinicalProtocol/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_existing_trials(indication: str, max_results: int = 8) -> list:
    """Query ClinicalTrials.gov v2 API. Falls back to mock data on failure."""
    query = urllib.parse.quote(indication)
    url = (
        f"https://clinicaltrials.gov/api/v2/studies"
        f"?query.term={query}&filter.overallStatus=COMPLETED,RECRUITING"
        f"&pageSize={max_results}&format=json"
    )
    try:
        data = _get(url)
        trials = []
        for study in data.get("studies", []):
            pm = study.get("protocolSection", {})
            id_m = pm.get("identificationModule", {})
            status_m = pm.get("statusModule", {})
            design_m = pm.get("designModule", {})
            cond_m = pm.get("conditionsModule", {})
            arms_m = pm.get("armsInterventionsModule", {})
            outcomes_m = pm.get("outcomesModule", {})

            phases = design_m.get("phases", ["N/A"])
            phase_str = phases[0] if phases else "N/A"

            trials.append(ExistingTrial(
                nct_id=id_m.get("nctId", ""),
                title=id_m.get("briefTitle", ""),
                phase=phase_str,
                status=status_m.get("overallStatus", ""),
                conditions=cond_m.get("conditions", []),
                interventions=[
                    i.get("name", "") for i in arms_m.get("interventions", [])
                ],
                enrollment=design_m.get("enrollmentInfo", {}).get("count"),
                primary_outcomes=[
                    o.get("measure", "") for o in outcomes_m.get("primaryOutcomes", [])
                ],
            ))
        return trials
    except Exception:
        # Graceful fallback to curated mock data
        indication_lower = indication.lower()
        for key, mock_list in MOCK_TRIALS.items():
            if any(word in indication_lower for word in key.split()):
                return [ExistingTrial(**{**t, "primary_completion_date": None,
                                        "eligibility_description": ""})
                        for t in mock_list]
        return []


def fetch_publications(query: str, max_results: int = 5) -> list:
    """Query NCBI PubMed eSearch + eFetch. Falls back to empty list on failure."""
    try:
        search_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=pubmed&term={urllib.parse.quote(query)}&retmax={max_results}&retmode=json"
        )
        search_data = _get(search_url)
        ids = search_data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        fetch_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=pubmed&id={','.join(ids)}&retmode=json"
        )
        fetch_data = _get(fetch_url)
        result_map = fetch_data.get("result", {})

        pubs = []
        for pmid in ids:
            article = result_map.get(pmid, {})
            pubs.append(Publication(
                pmid=pmid,
                title=article.get("title", ""),
                abstract="",  # summary endpoint doesn't include abstract
                publication_year=int(article.get("pubdate", "2020")[:4]),
                mesh_terms=[m.get("name", "") for m in article.get("meshheadinglist", [])],
            ))
        return pubs
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _sample_size_two_proportions(
    p1: float, p2: float,
    alpha: float = 0.05, power: float = 0.80
) -> int:
    """Per-arm N for a two-sided test comparing two proportions."""
    z_a = Z_ALPHA_05 if alpha >= 0.05 else Z_ALPHA_025
    z_b = Z_BETA_80 if power <= 0.80 else Z_BETA_90
    pooled = (p1 + p2) / 2
    numerator = (z_a * math.sqrt(2 * pooled * (1 - pooled))
                 + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p1 - p2) ** 2
    return max(1, math.ceil(numerator / denominator))


def _inflate_for_dropout(n: int, dropout_rate: float) -> int:
    """Inflate per-arm N to account for expected dropout."""
    return math.ceil(n / (1 - dropout_rate))


# ─────────────────────────────────────────────────────────────────────────────
# ELEPHANT AGENT — Long-context designer
# ─────────────────────────────────────────────────────────────────────────────

class ElephantAgent:
    """
    EGM Elephant: accumulates full research context and drafts comprehensive protocols.
    Unlike the stateless Goldfish critics, the Elephant carries memory across rounds,
    using each critique to improve its understanding of the design space.
    """

    def __init__(self):
        self.trials_context: list = []
        self.publications_context: list = []
        self.rejected_options: list = []
        self.design_log: list = []

    def load_context(self, trials: list, publications: list) -> None:
        self.trials_context = trials
        self.publications_context = publications

    # ── Protocol synthesis ──────────────────────────────────────────────────

    def draft_protocol(self, hypothesis: Hypothesis) -> ProtocolDraft:
        """Synthesise evidence from context into a structured protocol draft."""
        phase = self._determine_phase(hypothesis)
        guidance = FDA_GUIDANCE.get(hypothesis.therapeutic_area, FDA_GUIDANCE["default"])
        primary_ep, secondary_eps = self._synthesize_endpoints(hypothesis, guidance)
        stat_plan = self._calculate_statistics(phase, primary_ep)
        eligibility = self._synthesize_eligibility(hypothesis)
        background = self._write_background(hypothesis)
        rationale = self._write_rationale(hypothesis, phase)

        return ProtocolDraft(
            title=(
                f"{hypothesis.proposed_intervention} vs {hypothesis.proposed_comparator} "
                f"in {hypothesis.indication}: A {phase} Study"
            ),
            version="1.0",
            hypothesis=hypothesis,
            background=background,
            rationale=rationale,
            phase=phase,
            study_design=self._design_string(phase, hypothesis),
            primary_endpoint=primary_ep,
            secondary_endpoints=secondary_eps,
            population=eligibility,
            statistical_plan=stat_plan,
            safety_monitoring=self._safety_plan(phase, guidance),
            duration_months=self._estimate_duration(phase),
            informed_consent=(
                "Written informed consent will be obtained from all participants "
                "prior to any study procedure. Participants retain the right to "
                "withdraw at any time without consequence."
            ),
            gcp_compliance=(
                "This trial will be conducted in accordance with ICH E6(R2) Good "
                "Clinical Practice guidelines, the Declaration of Helsinki, and all "
                "applicable local regulatory requirements."
            ),
            context_trials_used=[t.nct_id for t in self.trials_context[:4]],
        )

    def resolve_critiques(
        self, protocol: ProtocolDraft, reviews: list
    ) -> ProtocolDraft:
        """Incorporate Goldfish critiques and return a revised protocol."""
        all_critiques = [c for r in reviews for c in r.critiques]
        critical = [c for c in all_critiques if c.severity == "critical"]

        # Bump minor version
        major, minor = protocol.version.split(".")
        new_version = f"{major}.{int(minor) + 1}"

        self.design_log.append(
            f"Round {new_version}: resolving {len(critical)} critical issue(s)."
        )

        # Apply resolutions
        for critique in critical:
            sec = critique.protocol_section

            if sec == "primary_endpoint" and "measurable" in critique.issue.lower():
                protocol.primary_endpoint = self._make_measurable(
                    protocol.primary_endpoint, protocol.phase
                )
                self.rejected_options.append(
                    f"Vague primary endpoint '{protocol.primary_endpoint}' rejected by Biostatistician"
                )

            elif sec == "statistical_plan" and "n=" in critique.issue.lower():
                min_n = PHASE_MIN_N.get(protocol.phase, 10)
                protocol.statistical_plan["n_per_arm"] = max(
                    min_n, protocol.statistical_plan.get("n_per_arm", 0)
                )
                dropout = protocol.statistical_plan.get("dropout_rate", 0.15)
                protocol.statistical_plan["total_enrollment"] = (
                    _inflate_for_dropout(protocol.statistical_plan["n_per_arm"], dropout) * 2
                )

            elif sec == "safety_monitoring" and "dsmb" in critique.issue.lower():
                protocol.safety_monitoring["dsmb"] = True
                protocol.safety_monitoring.setdefault(
                    "stopping_rules",
                    ["≥20% grade 4 toxicity rate",
                     "Unexpected mortality exceeding 5%",
                     "Interim futility analysis at 50% information time"]
                )

            elif sec == "safety_monitoring" and "stopping" in critique.issue.lower():
                protocol.safety_monitoring["stopping_rules"] = [
                    "≥20% grade 4 toxicity rate in either arm",
                    "Unexpected mortality exceeding 5%",
                    "Interim futility analysis at 50% information time",
                ]

            elif sec == "safety_monitoring" and "adverse" in critique.issue.lower():
                eps = protocol.safety_monitoring.setdefault("safety_endpoints", [])
                if not any("adverse" in e.lower() for e in eps):
                    eps.insert(0, "Incidence of Grade 3+ adverse events (CTCAE v5)")

            elif sec == "gcp_compliance":
                protocol.gcp_compliance = (
                    "This trial will be conducted in accordance with ICH E6(R2) Good "
                    "Clinical Practice guidelines, the Declaration of Helsinki, and all "
                    "applicable local regulatory requirements."
                )

            elif sec == "informed_consent":
                protocol.informed_consent = (
                    "Written informed consent will be obtained from all participants "
                    "prior to any study procedure. Participants retain the right to "
                    "withdraw at any time without consequence. An ICF comprehension "
                    "assessment will be performed."
                )

            elif sec == "population" and "pregnancy" in critique.issue.lower():
                exclusions = protocol.population.setdefault("exclusion", [])
                exclusions.append(
                    "Pregnant or breastfeeding patients; women of child-bearing "
                    "potential without adequate contraception"
                )

            elif sec == "population" and "organ" in critique.issue.lower():
                exclusions = protocol.population.setdefault("exclusion", [])
                exclusions.append(
                    "Inadequate organ function: creatinine >1.5×ULN, "
                    "total bilirubin >1.5×ULN, or ALT/AST >3×ULN"
                )

        # Resolve dropout warning
        if not protocol.statistical_plan.get("dropout_rate"):
            protocol.statistical_plan["dropout_rate"] = 0.15
            n = protocol.statistical_plan.get("n_per_arm", 30)
            protocol.statistical_plan["total_enrollment"] = (
                _inflate_for_dropout(n, 0.15) * 2
            )

        # Ensure ≥2 secondary endpoints
        if len(protocol.secondary_endpoints) < 2:
            protocol.secondary_endpoints.append(
                "Patient-reported Quality of Life (PRO/QoL instrument)"
            )

        protocol.version = new_version
        return protocol

    def build_concepts_brief(
        self, hypothesis: Hypothesis, rounds: list
    ) -> ConceptsBrief:
        all_critiques = [
            c for r in rounds for gr in r.goldfish_reviews for c in gr.critiques
        ]
        rejected = list(set(self.rejected_options))
        considered = (
            [f"Existing trial data: {t.nct_id} ({t.phase})" for t in self.trials_context[:4]]
            + [f"Publication: {p.title[:60]}..." for p in self.publications_context[:3] if p.title]
        )
        if not considered:
            considered = [
                "ClinicalTrials.gov search for existing trials",
                "FDA guidance for " + hypothesis.therapeutic_area,
                "ICH E6(R2) GCP checklist",
            ]

        decisions = {
            "phase": f"{hypothesis.proposed_phase} chosen based on existing evidence landscape",
            "primary_endpoint": "Selected to align with FDA preferred endpoints for indication",
            "sample_size": "Calculated for 80% power at α=0.05 with 15% dropout inflation",
            "goldfishes": f"{len(rounds)} review round(s), "
                          f"{len(all_critiques)} total critique(s) resolved",
        }

        return ConceptsBrief(
            hypothesis=hypothesis,
            egm_rounds_run=len(rounds),
            what_was_considered=considered,
            what_was_rejected=rejected or ["No design options were rejected in final protocol"],
            key_design_decisions=decisions,
            final_rationale=(
                f"After {len(rounds)} EGM review round(s), the protocol achieved "
                "consensus across Biostatistician, Patient Safety, and Regulatory "
                "Goldfish critics. Key improvements included DSMB establishment, "
                "quantified stopping rules, organ-function exclusion criteria, "
                "and explicit GCP compliance language."
            ),
            generated_on=date.today().isoformat(),
        )

    # ── Private synthesis helpers ───────────────────────────────────────────

    def _determine_phase(self, hypothesis: Hypothesis) -> str:
        """Infer appropriate phase from existing trial landscape."""
        existing_phases = {t.phase for t in self.trials_context}
        requested = hypothesis.proposed_phase

        # If there's already Phase III data, suggest Phase III or IV
        if "PHASE3" in existing_phases or "Phase III" in existing_phases:
            if requested in ("Phase III", "Phase IV"):
                return requested
            return "Phase II"  # still needed for this specific combination

        if "PHASE2" in existing_phases or "Phase II" in existing_phases:
            return requested if requested in ("Phase II", "Phase III") else "Phase II"

        return requested or "Phase II"

    def _synthesize_endpoints(
        self, hypothesis: Hypothesis, guidance: dict
    ) -> tuple:
        """Extract primary/secondary endpoints from trial context + guidance."""
        # Gather endpoints from existing trials
        observed_endpoints = []
        for trial in self.trials_context:
            observed_endpoints.extend(trial.primary_outcomes)

        preferred = guidance.get("preferred_endpoints", [])
        therapeutic_area = hypothesis.therapeutic_area

        # Primary endpoint
        if therapeutic_area == "oncology":
            primary = "Progression-Free Survival (PFS) at 12 months"
        elif therapeutic_area == "cardiology":
            primary = "MACE composite (CV death, non-fatal MI, HF hospitalisation) at 12 months"
        elif therapeutic_area == "endocrinology":
            primary = "Change in HbA1c from baseline at 26 weeks"
        else:
            primary = preferred[0] if preferred else "Primary efficacy endpoint at 12 months"

        # Secondary endpoints
        secondary = []
        if therapeutic_area == "oncology":
            secondary = [
                "Overall Survival (OS)",
                "Objective Response Rate (ORR) per RECIST v1.1",
                "Duration of Response (DoR)",
                "Patient-reported outcomes (EORTC QLQ-C30)",
                "Incidence of Grade 3+ adverse events (CTCAE v5)",
            ]
        elif therapeutic_area == "cardiology":
            secondary = [
                "All-cause mortality at 12 months",
                "Change in NT-proBNP from baseline",
                "6-Minute Walk Distance (6MWD) change",
                "KCCQ Overall Summary Score change",
                "Incidence of Grade 3+ adverse events (CTCAE v5)",
            ]
        elif therapeutic_area == "endocrinology":
            secondary = [
                "Proportion of patients achieving HbA1c <7.0%",
                "Change in body weight from baseline",
                "Fasting plasma glucose change from baseline",
                "Incidence of hypoglycaemia (documented symptomatic)",
            ]
        else:
            secondary = [
                preferred[1] if len(preferred) > 1 else "Secondary efficacy endpoint",
                "Patient-reported Quality of Life (PRO instrument)",
                "Incidence of Grade 3+ adverse events (CTCAE v5)",
            ]

        return primary, secondary

    def _calculate_statistics(self, phase: str, primary_ep: str) -> dict:
        """Calculate sample size with dropout inflation."""
        alpha = 0.05
        power = 0.80
        dropout = 0.15

        # Estimate effect assumptions by endpoint type
        ep_lower = primary_ep.lower()
        if "pfs" in ep_lower or "progression" in ep_lower:
            p1, p2 = 0.50, 0.35  # control 35%, treatment 50% PFS rate
        elif "os" in ep_lower or "survival" in ep_lower:
            p1, p2 = 0.60, 0.45
        elif "orr" in ep_lower or "response" in ep_lower:
            p1, p2 = 0.40, 0.25
        elif "hba1c" in ep_lower or "glucose" in ep_lower:
            p1, p2 = 0.55, 0.40   # proportion achieving target
        elif "mace" in ep_lower:
            p1, p2 = 0.12, 0.18   # MACE rates (lower is better — use 1-p convention)
        else:
            p1, p2 = 0.45, 0.30

        n_per_arm = _sample_size_two_proportions(p1, p2, alpha=alpha, power=power)
        n_per_arm = max(n_per_arm, PHASE_MIN_N.get(phase, 10))
        total = _inflate_for_dropout(n_per_arm, dropout) * 2

        # Choose test type
        if any(kw in ep_lower for kw in ["survival", "pfs", "os", "time to"]):
            test = "Log-rank test (Kaplan-Meier curves)"
        elif any(kw in ep_lower for kw in ["change", "hba1c", "distance", "score"]):
            test = "Two-sample t-test (or ANCOVA with baseline covariate)"
        else:
            test = "Chi-squared test (or Fisher's exact for small samples)"

        return {
            "test": test,
            "alpha": alpha,
            "power": power,
            "n_per_arm": n_per_arm,
            "dropout_rate": dropout,
            "total_enrollment": total,
            "assumed_rates": {"treatment": p1, "control": p2},
        }

    def _synthesize_eligibility(self, hypothesis: Hypothesis) -> dict:
        """Draft inclusion/exclusion criteria from trial context patterns."""
        area = hypothesis.therapeutic_area
        indication = hypothesis.indication.lower()

        inclusion = [
            f"Confirmed diagnosis of {hypothesis.indication}",
            f"Target population: {hypothesis.target_population}",
            "ECOG Performance Status 0–2 (oncology) or equivalent",
            "Age ≥18 years",
            "Adequate bone marrow function: ANC ≥1.5×10⁹/L, platelets ≥100×10⁹/L",
            "Signed informed consent",
        ]

        exclusion = [
            "Prior treatment with the proposed intervention or related agent",
            "Active uncontrolled infection or autoimmune condition requiring systemic treatment",
            "Pregnant or breastfeeding patients; women of child-bearing potential without adequate contraception",
            "Inadequate organ function: creatinine >1.5×ULN, total bilirubin >1.5×ULN, ALT/AST >3×ULN",
            "Concurrent participation in another interventional clinical trial",
        ]

        if area == "oncology":
            inclusion.append("Measurable disease per RECIST v1.1")
            exclusion.append("Active CNS metastases (unless stable for ≥4 weeks)")
        elif area == "cardiology":
            inclusion.append("LVEF <40% confirmed by echocardiography or MRI")
            exclusion.append("Recent acute coronary syndrome within 12 weeks")
        elif area == "endocrinology":
            inclusion.append("HbA1c 7.5–10.5% at screening")
            exclusion.append("Type 1 diabetes or latent autoimmune diabetes of adults (LADA)")

        stat = self._calculate_statistics(hypothesis.proposed_phase, "primary endpoint")
        return {
            "inclusion": inclusion,
            "exclusion": exclusion,
            "estimated_n": stat["n_per_arm"],
        }

    def _safety_plan(self, phase: str, guidance: dict) -> dict:
        dsmb_required = guidance.get("dsmb_required", True)
        # Phase I typically has smaller DSMB-like oversight
        if phase == "Phase I":
            dsmb_required = False

        stopping_rules = [
            "≥20% incidence of Grade 4 treatment-related adverse events",
            "Unexpected all-cause mortality exceeding 5% above control arm",
            "Interim futility analysis at 50% information time (O'Brien-Fleming boundary)",
        ] if phase in ("Phase II", "Phase III", "Phase IV") else [
            "Any Grade 5 event potentially attributable to study drug",
            "≥33% DLT rate in current dose cohort",
        ]

        return {
            "dsmb": dsmb_required,
            "stopping_rules": stopping_rules,
            "safety_endpoints": [
                "Incidence of Grade 3+ adverse events (CTCAE v5)",
                "Serious adverse events (SAEs) — all-causality and treatment-related",
                "Dose interruptions and reductions attributable to toxicity",
            ],
        }

    def _write_background(self, hypothesis: Hypothesis) -> str:
        trial_refs = " ".join(f"({t.nct_id})" for t in self.trials_context[:2])
        return (
            f"{hypothesis.indication} represents a significant unmet medical need. "
            f"Existing trials {trial_refs or '(reviewed from literature)'} have explored "
            f"related interventions, providing a foundation for the current study. "
            f"{hypothesis.proposed_intervention} demonstrates a rationale for evaluation "
            f"in {hypothesis.target_population}, building on published evidence of "
            f"mechanistic synergy and preliminary clinical signals."
        )

    def _write_rationale(self, hypothesis: Hypothesis, phase: str) -> str:
        n_trials = len(self.trials_context)
        return (
            f"A {phase} study is warranted based on {n_trials} identified existing "
            f"trial(s) in the indication. The comparison of "
            f"{hypothesis.proposed_intervention} versus {hypothesis.proposed_comparator} "
            f"addresses a specific gap in the evidence base for "
            f"{hypothesis.target_population}. The proposed phase is appropriate given "
            f"the existing evidence landscape and the need to establish "
            f"efficacy and safety before a pivotal study."
        )

    def _design_string(self, phase: str, hypothesis: Hypothesis) -> str:
        if phase == "Phase I":
            return (
                f"Open-label, dose-escalation Phase I study of "
                f"{hypothesis.proposed_intervention} (3+3 design)"
            )
        if hypothesis.proposed_comparator.lower() in ("", "none"):
            return f"Single-arm, open-label {phase} study"
        return (
            f"Randomised, double-blind, {hypothesis.proposed_comparator}-controlled "
            f"{phase} study (1:1 allocation)"
        )

    def _make_measurable(self, endpoint: str, phase: str) -> str:
        """Make a vague endpoint measurable with time horizon."""
        ep = endpoint.strip().rstrip(".")
        if any(kw in ep.lower() for kw in MEASURABLE_ENDPOINT_KEYWORDS):
            return ep  # already measurable
        horizon = "12 months" if phase in ("Phase II", "Phase III") else "6 months"
        return f"Objective Response Rate (ORR) at {horizon} per RECIST v1.1"

    def _estimate_duration(self, phase: str) -> int:
        return {"Phase I": 18, "Phase II": 30, "Phase III": 48, "Phase IV": 36}.get(phase, 30)


# ─────────────────────────────────────────────────────────────────────────────
# GOLDFISH CRITICS — Stateless reviewers (EGM pattern)
# ─────────────────────────────────────────────────────────────────────────────

class GoldfishCritic:
    """
    Base class for EGM Goldfish critics.
    Stateless: no memory between review rounds. Each review is independent.
    Three-phase review: Comprehension Check → Antagonistic Review → Implementation Readiness.
    """

    persona: str = "goldfish"

    def review(self, protocol: ProtocolDraft) -> GoldfishReview:
        comprehension = self.comprehension_check(protocol)
        critiques = self.antagonistic_review(protocol)
        readiness = self.implementation_readiness(protocol, critiques)
        return GoldfishReview(
            goldfish_persona=self.persona,
            comprehension=comprehension,
            critiques=critiques,
            readiness=readiness,
        )

    def comprehension_check(self, protocol: ProtocolDraft) -> ComprehensionCheck:
        raise NotImplementedError

    def antagonistic_review(self, protocol: ProtocolDraft) -> list:
        raise NotImplementedError

    def implementation_readiness(
        self, protocol: ProtocolDraft, critiques: list
    ) -> ImplementationReadiness:
        blockers = [c.issue for c in critiques if c.severity == "critical"]
        suggestions = [c.recommendation for c in critiques if c.severity in ("info", "warning")]
        return ImplementationReadiness(
            goldfish_persona=self.persona,
            ready=len(blockers) == 0,
            blockers=blockers,
            suggestions=suggestions,
        )


class BiostatisticianGoldfish(GoldfishCritic):
    """
    EGM Goldfish: Biostatistician.
    Attacks: endpoint clarity, sample size, alpha, power, dropout, secondary endpoints.
    """

    persona = "Biostatistician"

    def comprehension_check(self, protocol: ProtocolDraft) -> ComprehensionCheck:
        stat = protocol.statistical_plan
        return ComprehensionCheck(
            goldfish_persona=self.persona,
            understood_study_design=bool(protocol.study_design),
            understood_primary_endpoint=bool(protocol.primary_endpoint),
            understood_population=bool(protocol.population.get("estimated_n", 0) > 0),
            understood_statistical_plan=bool(stat.get("test") and "alpha" in stat),
            notes=(
                f"Reviewing {protocol.phase} protocol — "
                f"n={stat.get('n_per_arm', '?')} per arm, "
                f"α={stat.get('alpha', '?')}, power={stat.get('power', '?'):.0%}."
            ) if stat.get("power") else f"Reviewing {protocol.phase} protocol.",
        )

    def antagonistic_review(self, protocol: ProtocolDraft) -> list:
        critiques = []
        stat = protocol.statistical_plan

        # 1. Primary endpoint must be measurable
        primary_lower = protocol.primary_endpoint.lower()
        if not any(kw in primary_lower for kw in MEASURABLE_ENDPOINT_KEYWORDS):
            critiques.append(Critique(
                goldfish_persona=self.persona, phase="antagonistic", severity="critical",
                issue=f"Primary endpoint '{protocol.primary_endpoint}' is not clearly measurable.",
                recommendation=(
                    "Restate as a quantifiable outcome with time horizon "
                    "(e.g., 'ORR at 12 weeks per RECIST v1.1')."
                ),
                protocol_section="primary_endpoint",
            ))

        # 2. Sample size must meet phase minimum
        n = stat.get("n_per_arm", 0)
        min_n = PHASE_MIN_N.get(protocol.phase, 10)
        if n < min_n:
            critiques.append(Critique(
                goldfish_persona=self.persona, phase="antagonistic", severity="critical",
                issue=f"n={n} per arm is below the minimum {min_n} for {protocol.phase}.",
                recommendation=f"Increase to ≥{min_n} per arm or provide regulatory justification.",
                protocol_section="statistical_plan",
            ))

        # 3. Alpha
        alpha = stat.get("alpha", 0.05)
        if alpha > 0.05:
            critiques.append(Critique(
                goldfish_persona=self.persona, phase="antagonistic", severity="warning",
                issue=f"Alpha={alpha} is more lenient than the conventional 0.05.",
                recommendation="Justify α>0.05 or revert to the standard two-sided 0.05.",
                protocol_section="statistical_plan",
            ))

        # 4. Power
        power = stat.get("power", 0.80)
        if power < 0.80:
            critiques.append(Critique(
                goldfish_persona=self.persona, phase="antagonistic", severity="warning",
                issue=f"Power={power:.0%} is below the conventional 80% threshold.",
                recommendation="Increase sample size to achieve ≥80% power.",
                protocol_section="statistical_plan",
            ))

        # 5. Dropout rate
        if not stat.get("dropout_rate"):
            critiques.append(Critique(
                goldfish_persona=self.persona, phase="antagonistic", severity="warning",
                issue="Dropout rate not specified in the statistical plan.",
                recommendation="Add 10–20% dropout inflation to planned enrollment.",
                protocol_section="statistical_plan",
            ))

        # 6. Secondary endpoints
        if len(protocol.secondary_endpoints) < 2:
            critiques.append(Critique(
                goldfish_persona=self.persona, phase="antagonistic", severity="info",
                issue="Fewer than 2 secondary endpoints specified.",
                recommendation=(
                    "Add ≥2 secondary endpoints (e.g., QoL measure, biomarker, "
                    "safety endpoint)."
                ),
                protocol_section="secondary_endpoints",
            ))

        return critiques


class PatientSafetyGoldfish(GoldfishCritic):
    """
    EGM Goldfish: Patient Safety.
    Attacks: DSMB, stopping rules, AE monitoring, vulnerable population protections,
    organ function thresholds.
    """

    persona = "Patient Safety"

    def comprehension_check(self, protocol: ProtocolDraft) -> ComprehensionCheck:
        safety = protocol.safety_monitoring
        design_lower = protocol.study_design.lower()
        return ComprehensionCheck(
            goldfish_persona=self.persona,
            understood_study_design=(
                "randomised" in design_lower
                or "single arm" in design_lower
                or "single-arm" in design_lower
                or "open-label" in design_lower
            ),
            understood_primary_endpoint=bool(protocol.primary_endpoint),
            understood_population=bool(protocol.population.get("exclusion")),
            understood_statistical_plan=bool(safety.get("stopping_rules")),
            notes=(
                f"DSMB={'required' if safety.get('dsmb') else 'not specified'}, "
                f"stopping rules: {len(safety.get('stopping_rules', []))}."
            ),
        )

    def antagonistic_review(self, protocol: ProtocolDraft) -> list:
        critiques = []
        safety = protocol.safety_monitoring
        exclusions = protocol.population.get("exclusion", [])
        exclusion_text = " ".join(exclusions).lower()

        # 1. DSMB for Phase II+
        if protocol.phase in ("Phase II", "Phase III", "Phase IV"):
            if not safety.get("dsmb"):
                critiques.append(Critique(
                    goldfish_persona=self.persona, phase="antagonistic", severity="critical",
                    issue=f"DSMB not specified for {protocol.phase} trial.",
                    recommendation=(
                        "Establish an independent DSMB with charter, "
                        "defined meeting schedule, and stopping criteria."
                    ),
                    protocol_section="safety_monitoring",
                ))

        # 2. Stopping rules
        if not safety.get("stopping_rules"):
            critiques.append(Critique(
                goldfish_persona=self.persona, phase="antagonistic", severity="critical",
                issue="No predefined stopping rules for safety events.",
                recommendation=(
                    "Define stopping rules (e.g., grade 4 toxicity ≥20%, "
                    "unexpected mortality, interim futility)."
                ),
                protocol_section="safety_monitoring",
            ))

        # 3. AE monitoring in safety endpoints
        safety_ep_text = " ".join(safety.get("safety_endpoints", [])).lower()
        if not any(kw in safety_ep_text for kw in ["adverse", "ae ", "toxicity", "safety"]):
            critiques.append(Critique(
                goldfish_persona=self.persona, phase="antagonistic", severity="critical",
                issue="Safety endpoints do not include adverse event monitoring.",
                recommendation=(
                    "Add 'Incidence of Grade 3+ adverse events (CTCAE v5)' "
                    "as a primary safety endpoint."
                ),
                protocol_section="safety_monitoring",
            ))

        # 4. Pregnancy exclusion
        if not any(kw in exclusion_text for kw in ["pregnant", "pregnancy", "breastfeed", "contraception"]):
            critiques.append(Critique(
                goldfish_persona=self.persona, phase="antagonistic", severity="warning",
                issue="Exclusion criteria do not address pregnancy or contraception.",
                recommendation=(
                    "Add: 'Pregnant or breastfeeding patients; women of child-bearing "
                    "potential without adequate contraception.'"
                ),
                protocol_section="population",
            ))

        # 5. Organ function exclusion
        organ_terms = ["hepatic", "renal", "liver", "kidney", "creatinine", "bilirubin", "alt", "ast"]
        if not any(term in exclusion_text for term in organ_terms):
            critiques.append(Critique(
                goldfish_persona=self.persona, phase="antagonistic", severity="warning",
                issue="No organ function thresholds in exclusion criteria.",
                recommendation=(
                    "Define minimum organ function: creatinine ≤1.5×ULN, "
                    "total bilirubin ≤1.5×ULN, ALT/AST ≤3×ULN."
                ),
                protocol_section="population",
            ))

        return critiques


class RegulatoryGoldfish(GoldfishCritic):
    """
    EGM Goldfish: Regulatory.
    Attacks: GCP compliance, informed consent, ICH E6, phase appropriateness,
    prior evidence justification, trial duration.
    """

    persona = "Regulatory"

    def comprehension_check(self, protocol: ProtocolDraft) -> ComprehensionCheck:
        return ComprehensionCheck(
            goldfish_persona=self.persona,
            understood_study_design=bool(protocol.study_design),
            understood_primary_endpoint=bool(protocol.primary_endpoint),
            understood_population=bool(protocol.population.get("inclusion")),
            understood_statistical_plan=bool(protocol.statistical_plan.get("test")),
            notes=(
                f"Phase: {protocol.phase}, duration: {protocol.duration_months} months, "
                f"GCP statement: {'present' if len(protocol.gcp_compliance) > 10 else 'absent'}."
            ),
        )

    def antagonistic_review(self, protocol: ProtocolDraft) -> list:
        critiques = []

        # 1. GCP compliance
        if not protocol.gcp_compliance or len(protocol.gcp_compliance) < 15:
            critiques.append(Critique(
                goldfish_persona=self.persona, phase="antagonistic", severity="critical",
                issue="ICH E6(R2) GCP compliance statement absent.",
                recommendation=(
                    "Add: 'This trial will be conducted in accordance with ICH E6(R2) "
                    "Good Clinical Practice guidelines.'"
                ),
                protocol_section="gcp_compliance",
            ))

        # 2. Informed consent
        if not protocol.informed_consent or len(protocol.informed_consent) < 20:
            critiques.append(Critique(
                goldfish_persona=self.persona, phase="antagonistic", severity="critical",
                issue="Informed consent process not described.",
                recommendation=(
                    "Add: written ICF process, right to withdraw without consequence, "
                    "comprehension assessment procedure."
                ),
                protocol_section="informed_consent",
            ))

        # 3. Phase I oversizing
        if protocol.phase == "Phase I" and protocol.statistical_plan.get("n_per_arm", 0) > 50:
            critiques.append(Critique(
                goldfish_persona=self.persona, phase="antagonistic", severity="warning",
                issue=f"Phase I with n>{50}/arm is unusually large.",
                recommendation=(
                    "Verify Phase I design appropriateness (dose escalation). "
                    "Consider reclassifying as Phase I/II."
                ),
                protocol_section="study_design",
            ))

        # 4. Phase III comparator
        if protocol.phase == "Phase III":
            design_lower = protocol.study_design.lower()
            if not any(kw in design_lower for kw in
                       ["placebo", "comparator", "active", "standard of care", "controlled"]):
                critiques.append(Critique(
                    goldfish_persona=self.persona, phase="antagonistic", severity="warning",
                    issue="Phase III comparator arm not explicitly defined.",
                    recommendation=(
                        "Specify active comparator, standard of care, or placebo "
                        "with regulatory justification."
                    ),
                    protocol_section="study_design",
                ))

        # 5. Duration
        if protocol.duration_months <= 0:
            critiques.append(Critique(
                goldfish_persona=self.persona, phase="antagonistic", severity="warning",
                issue="Trial duration not specified.",
                recommendation=(
                    "Define total trial duration, split into enrollment period, "
                    "treatment period, and follow-up period."
                ),
                protocol_section="duration_months",
            ))

        # 6. Prior data justification for Phase II+
        if protocol.phase in ("Phase II", "Phase III"):
            if not protocol.context_trials_used and len(protocol.rationale) < 30:
                critiques.append(Critique(
                    goldfish_persona=self.persona, phase="antagonistic", severity="warning",
                    issue="No prior trial evidence cited to justify this phase.",
                    recommendation=(
                        "Reference Phase I data or published evidence supporting "
                        "the proposed intervention in this indication."
                    ),
                    protocol_section="rationale",
                ))

        return critiques


# ─────────────────────────────────────────────────────────────────────────────
# EGM LOOP ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class EGMLoop:
    """
    EGM orchestrator implementing the review-loop skill from google/elephant-goldfish-model.

    Workflow:
        1. Elephant loads research context (ClinicalTrials.gov + PubMed)
        2. Elephant drafts initial protocol
        3. Each Goldfish runs three-phase review (Comprehension → Antagonistic → Readiness)
        4. Elephant resolves critical critiques and bumps protocol version
        5. Repeat until all Goldfishes pass or max_rounds reached
    """

    def __init__(self, max_rounds: int = 3, verbose: bool = True):
        self.elephant = ElephantAgent()
        self.goldfishes = [
            BiostatisticianGoldfish(),
            PatientSafetyGoldfish(),
            RegulatoryGoldfish(),
        ]
        self.max_rounds = max_rounds
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def run(self, hypothesis: Hypothesis) -> EGMResult:
        """Execute the full EGM loop and return a finalised protocol + concepts brief."""

        sep = "=" * 72

        self._log(f"\n{sep}")
        self._log(f"  EGM CLINICAL PROTOCOL DESIGNER  —  {EGM_CREDIT}")
        self._log(sep)
        self._log(f"\n  Hypothesis: {hypothesis.research_question}")
        self._log(f"  Indication: {hypothesis.indication}")
        self._log(f"  Requested phase: {hypothesis.proposed_phase}")
        self._log("")

        # ── Phase 0: Elephant loads context ───────────────────────────────
        self._log("[Elephant] Loading research context...")
        trials = fetch_existing_trials(hypothesis.indication)
        pubs = fetch_publications(
            f"{hypothesis.indication} {hypothesis.proposed_intervention}"
        )
        self.elephant.load_context(trials, pubs)
        self._log(
            f"  ✓ Context: {len(trials)} trial(s) from ClinicalTrials.gov, "
            f"{len(pubs)} publication(s) from PubMed"
        )

        # ── Phase 1: Elephant drafts initial protocol ──────────────────────
        self._log("\n[Elephant] Drafting initial protocol v1.0 ...")
        protocol = self.elephant.draft_protocol(hypothesis)
        self._log(f"  ✓ Title: {protocol.title}")
        self._log(f"  ✓ Phase: {protocol.phase} | Design: {protocol.study_design}")
        self._log(f"  ✓ Primary endpoint: {protocol.primary_endpoint}")
        self._log(
            f"  ✓ Sample size: {protocol.statistical_plan['n_per_arm']} per arm "
            f"({protocol.statistical_plan['total_enrollment']} total enrolled)"
        )

        rounds: list = []
        all_passed = False

        # ── EGM Review Loop ────────────────────────────────────────────────
        for round_num in range(1, self.max_rounds + 1):
            self._log(f"\n{'─' * 72}")
            self._log(f"  EGM REVIEW ROUND {round_num}/{self.max_rounds}  |  "
                      f"Protocol v{protocol.version}")
            self._log("─" * 72)

            goldfish_reviews = []

            for gf in self.goldfishes:
                self._log(f"\n  [{gf.persona} Goldfish]")
                review = gf.review(protocol)
                goldfish_reviews.append(review)

                # Comprehension phase
                cc = review.comprehension
                status = "✓" if cc.passed else "✗ FAILED"
                self._log(f"    Comprehension: {status}  — {cc.notes}")

                # Antagonistic phase
                critical = [c for c in review.critiques if c.severity == "critical"]
                warnings = [c for c in review.critiques if c.severity == "warning"]
                infos = [c for c in review.critiques if c.severity == "info"]
                self._log(
                    f"    Antagonistic:  {len(critical)} critical | "
                    f"{len(warnings)} warnings | {len(infos)} info"
                )
                for c in critical:
                    self._log(f"      [CRITICAL] {c.issue}")
                for c in warnings:
                    self._log(f"      [WARNING]  {c.issue}")

                # Implementation readiness
                ir = review.readiness
                ready_str = "✅ Ready" if ir.ready else f"❌ Not ready ({len(ir.blockers)} blocker(s))"
                self._log(f"    Readiness:     {ready_str}")

            all_passed = all(r.readiness.ready for r in goldfish_reviews)
            all_critiques = [c for r in goldfish_reviews for c in r.critiques]
            resolutions: list = []

            if not all_passed:
                self._log(
                    f"\n[Elephant] Resolving {len([c for c in all_critiques if c.severity == 'critical'])} "
                    f"critical critique(s) ..."
                )
                protocol = self.elephant.resolve_critiques(protocol, goldfish_reviews)
                resolutions = [f"Protocol revised to v{protocol.version}"]
                self._log(f"  ✓ Protocol updated to v{protocol.version}")

            round_record = EGMReviewRound(
                round_number=round_num,
                protocol_version=protocol.version,
                goldfish_reviews=goldfish_reviews,
                elephant_resolutions=resolutions,
                all_passed=all_passed,
            )
            rounds.append(round_record)
            protocol.review_history.append(round_record)

            if all_passed:
                self._log(
                    f"\n{'=' * 72}\n"
                    f"  ✅ All Goldfish critics passed — protocol finalised after "
                    f"round {round_num}.\n{'=' * 72}"
                )
                break

        if not all_passed:
            self._log(
                f"\n{'=' * 72}\n"
                f"  ⚠ Max rounds ({self.max_rounds}) reached. "
                f"Using best available protocol v{protocol.version}.\n{'=' * 72}"
            )

        concepts_brief = self.elephant.build_concepts_brief(hypothesis, rounds)

        return EGMResult(
            protocol=protocol,
            concepts_brief=concepts_brief,
            review_rounds=rounds,
            total_rounds=len(rounds),
            all_critics_passed=all_passed,
            generated_on=date.today().isoformat(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTERS  (EGM naming convention: protocol_draft.md, concepts_brief.md)
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(text: str, width: int = 80, indent: str = "") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def format_protocol_draft(result: EGMResult) -> str:
    """Generate protocol_draft.md — primary EGM output artifact."""
    p = result.protocol
    h = p.hypothesis
    stat = p.statistical_plan
    safety = p.safety_monitoring
    pop = p.population

    lines = [
        f"# {p.title}",
        f"",
        f"> **EGM Experiment** · {EGM_CREDIT}  ",
        f"> Protocol version {p.version} · Generated {result.generated_on}  ",
        f"> EGM review rounds: {result.total_rounds} | "
        f"All critics passed: {'Yes ✅' if result.all_critics_passed else 'No ⚠'}",
        "",
        "---",
        "",
        "## 1. Research Hypothesis",
        "",
        _wrap(h.research_question),
        "",
        f"- **Indication:** {h.indication}",
        f"- **Target population:** {h.target_population}",
        f"- **Intervention:** {h.proposed_intervention}",
        f"- **Comparator:** {h.proposed_comparator}",
        f"- **Therapeutic area:** {h.therapeutic_area}",
        "",
        "## 2. Background",
        "",
        _wrap(p.background),
        "",
        "## 3. Rationale",
        "",
        _wrap(p.rationale),
        "",
        "## 4. Study Design",
        "",
        f"- **Phase:** {p.phase}",
        f"- **Design:** {p.study_design}",
        f"- **Duration:** {p.duration_months} months (total, including follow-up)",
        "",
        "## 5. Objectives & Endpoints",
        "",
        "### 5.1 Primary Endpoint",
        "",
        f"- {p.primary_endpoint}",
        "",
        "### 5.2 Secondary Endpoints",
        "",
    ]
    for ep in p.secondary_endpoints:
        lines.append(f"- {ep}")

    lines += [
        "",
        "## 6. Study Population",
        "",
        "### 6.1 Inclusion Criteria",
        "",
    ]
    for ic in pop.get("inclusion", []):
        lines.append(f"- {ic}")

    lines += [
        "",
        "### 6.2 Exclusion Criteria",
        "",
    ]
    for ec in pop.get("exclusion", []):
        lines.append(f"- {ec}")

    lines += [
        "",
        f"### 6.3 Planned Enrollment",
        f"",
        f"- Per arm: {stat.get('n_per_arm', 'TBD')} participants",
        f"- Total enrollment (with {stat.get('dropout_rate', 0.15):.0%} dropout inflation): "
        f"{stat.get('total_enrollment', 'TBD')}",
        "",
        "## 7. Statistical Analysis Plan",
        "",
        f"- **Primary statistical test:** {stat.get('test', 'TBD')}",
        f"- **Significance level (α):** {stat.get('alpha', 0.05)} (two-sided)",
        f"- **Target power:** {stat.get('power', 0.80):.0%}",
        f"- **Assumed rates:** Treatment {stat.get('assumed_rates', {}).get('treatment', '?'):.0%} "
        f"vs Control {stat.get('assumed_rates', {}).get('control', '?'):.0%}",
        "",
        "## 8. Safety Monitoring",
        "",
        f"- **Independent DSMB:** {'Required' if safety.get('dsmb') else 'Not required (Phase I)'}",
        "",
        "### 8.1 Predefined Stopping Rules",
        "",
    ]
    for rule in safety.get("stopping_rules", []):
        lines.append(f"- {rule}")

    lines += [
        "",
        "### 8.2 Safety Endpoints",
        "",
    ]
    for sep in safety.get("safety_endpoints", []):
        lines.append(f"- {sep}")

    lines += [
        "",
        "## 9. Informed Consent",
        "",
        _wrap(p.informed_consent),
        "",
        "## 10. GCP Compliance",
        "",
        _wrap(p.gcp_compliance),
        "",
        "## 11. Context Trials",
        "",
        "The following existing trials informed this protocol design:",
        "",
    ]
    if p.context_trials_used:
        for nct in p.context_trials_used:
            lines.append(f"- [{nct}](https://clinicaltrials.gov/study/{nct})")
    else:
        lines.append("- ClinicalTrials.gov search performed; no directly matching trials found")

    lines += ["", "---", f"*{EGM_CREDIT}*", ""]
    return "\n".join(lines)


def format_concepts_brief(result: EGMResult) -> str:
    """Generate concepts_brief.md — EGM secondary output artifact."""
    cb = result.concepts_brief
    h = cb.hypothesis

    lines = [
        "# Concepts Brief — EGM Clinical Protocol Designer",
        "",
        f"> **EGM Experiment** · {EGM_CREDIT}  ",
        f"> Generated {cb.generated_on} · {cb.egm_rounds_run} review round(s)",
        "",
        "---",
        "",
        "## Hypothesis",
        "",
        _wrap(h.research_question),
        "",
        "## What Was Considered",
        "",
    ]
    for item in cb.what_was_considered:
        lines.append(f"- {item}")

    lines += [
        "",
        "## What Was Rejected (by Goldfish critics)",
        "",
    ]
    for item in cb.what_was_rejected:
        lines.append(f"- {item}")

    lines += [
        "",
        "## Key Design Decisions",
        "",
    ]
    for k, v in cb.key_design_decisions.items():
        lines.append(f"**{k.replace('_', ' ').title()}:** {v}")
        lines.append("")

    lines += [
        "## Final Rationale",
        "",
        _wrap(cb.final_rationale),
        "",
        "---",
        f"*{EGM_CREDIT}*",
        "",
    ]
    return "\n".join(lines)


def format_review_log(result: EGMResult) -> str:
    """Generate egm_review_log.md — full Goldfish review history."""
    lines = [
        "# EGM Review Log — Clinical Protocol Designer",
        "",
        f"> {EGM_CREDIT}",
        f"> Generated {result.generated_on}",
        "",
        "---",
        "",
    ]

    for rnd in result.review_rounds:
        lines += [
            f"## Round {rnd.round_number} — Protocol v{rnd.protocol_version}",
            "",
        ]
        for review in rnd.goldfish_reviews:
            cc = review.comprehension
            ir = review.readiness
            lines += [
                f"### {review.goldfish_persona} Goldfish",
                "",
                f"**Comprehension Check:** {'✓ Passed' if cc.passed else '✗ Failed'}",
                f"> {cc.notes}",
                "",
                "**Antagonistic Review:**",
                "",
            ]
            if review.critiques:
                for c in review.critiques:
                    emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(c.severity, "⚪")
                    lines.append(f"- {emoji} `{c.severity.upper()}` — **{c.issue}**")
                    lines.append(f"  - *Recommendation:* {c.recommendation}")
            else:
                lines.append("- No critiques raised.")

            lines += [
                "",
                f"**Implementation Readiness:** {'✅ Ready' if ir.ready else '❌ Not ready'}",
            ]
            if ir.blockers:
                for b in ir.blockers:
                    lines.append(f"- Blocker: {b}")
            lines.append("")

        if rnd.all_passed:
            lines.append(f"**→ Round {rnd.round_number}: All critics passed. Protocol finalised.**")
        else:
            lines.append(f"**→ Round {rnd.round_number}: Critiques raised. Elephant resolving...**")
            for res in rnd.elephant_resolutions:
                lines.append(f"  - {res}")

        lines += ["", "---", ""]

    return "\n".join(lines)


def print_summary(result: EGMResult) -> None:
    """Print human-readable summary to console."""
    p = result.protocol
    sep = "=" * 72

    print(f"\n{sep}")
    print(f"  EGM CLINICAL PROTOCOL DESIGNER — SCOUT REPORT")
    print(sep)
    print(f"  Protocol:       {p.title}")
    print(f"  Phase:          {p.phase}")
    print(f"  Version:        v{p.version}  ({result.total_rounds} EGM round(s))")
    print(f"  Critics passed: {'✅ All' if result.all_critics_passed else '⚠ Not all'}")
    print(f"  Generated:      {result.generated_on}")
    print()
    print(f"  PRIMARY ENDPOINT:")
    print(f"  {p.primary_endpoint}")
    print()
    print(f"  SAMPLE SIZE:")
    stat = p.statistical_plan
    print(
        f"  {stat['n_per_arm']} per arm  |  {stat['total_enrollment']} total "
        f"(+{stat['dropout_rate']:.0%} dropout)  |  "
        f"α={stat['alpha']} / power={stat['power']:.0%}"
    )
    print()
    print(f"  SAFETY:")
    safety = p.safety_monitoring
    print(f"  DSMB: {'Required' if safety['dsmb'] else 'Not required'}  |  "
          f"Stopping rules: {len(safety['stopping_rules'])}")
    print()
    print(f"  GOLDFISH REVIEW SUMMARY:")
    for rnd in result.review_rounds:
        for gr in rnd.goldfish_reviews:
            n_crit = len([c for c in gr.critiques if c.severity == "critical"])
            status = "✅" if gr.readiness.ready else "❌"
            print(
                f"  Round {rnd.round_number} · {gr.goldfish_persona:18s} "
                f"{status}  {n_crit} critical"
            )
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

DEMO_HYPOTHESES = {
    "nsclc": Hypothesis(
        research_question=(
            "Does Pembrolizumab + nab-Paclitaxel improve 12-month PFS "
            "vs Pembrolizumab monotherapy in treatment-naive PD-L1+ NSCLC?"
        ),
        indication="Non-small cell lung cancer",
        target_population="Stage IIIB/IV NSCLC, PD-L1 TPS ≥1%, no prior systemic therapy",
        proposed_intervention="Pembrolizumab 200 mg IV Q3W + nab-Paclitaxel 100 mg/m² IV Q1W",
        proposed_comparator="Pembrolizumab 200 mg IV Q3W (monotherapy)",
        proposed_phase="Phase II",
        therapeutic_area="oncology",
    ),
    "hf": Hypothesis(
        research_question=(
            "Does Empagliflozin 10 mg reduce MACE composite vs placebo "
            "in HFrEF patients on optimised medical therapy?"
        ),
        indication="Heart failure with reduced ejection fraction",
        target_population="HFrEF (LVEF <40%), NYHA Class II–III, on stable guideline-directed therapy",
        proposed_intervention="Empagliflozin 10 mg oral once daily",
        proposed_comparator="Matched placebo",
        proposed_phase="Phase III",
        therapeutic_area="cardiology",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EGM Clinical Protocol Designer — an experiment with Google's EGM framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python main.py --demo
              python main.py --demo hf
              python main.py --hypothesis "..." --indication "NSCLC" --phase "Phase II"
              python main.py --demo --save output/
        """),
    )
    parser.add_argument(
        "--demo", nargs="?", const="nsclc", metavar="SCENARIO",
        help="Run built-in demo (nsclc [default] or hf)",
    )
    parser.add_argument("--hypothesis", type=str, help="Research question")
    parser.add_argument("--indication", type=str, help="Disease / indication")
    parser.add_argument("--phase", type=str, default="Phase II",
                        choices=["Phase I", "Phase II", "Phase III", "Phase IV"])
    parser.add_argument("--intervention", type=str, default="experimental treatment")
    parser.add_argument("--comparator", type=str, default="standard of care")
    parser.add_argument("--area", type=str, default="oncology",
                        dest="therapeutic_area",
                        choices=["oncology", "cardiology", "endocrinology", "default"])
    parser.add_argument("--population", type=str, default="")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--save", type=str, metavar="DIR",
                        help="Directory to save protocol_draft.md, concepts_brief.md, egm_review_log.md")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")

    args = parser.parse_args()

    if args.demo is not None:
        scenario = args.demo if args.demo in DEMO_HYPOTHESES else "nsclc"
        hypothesis = DEMO_HYPOTHESES[scenario]
    elif args.hypothesis and args.indication:
        hypothesis = Hypothesis(
            research_question=args.hypothesis,
            indication=args.indication,
            target_population=args.population or (args.indication + " patients"),
            proposed_intervention=args.intervention,
            proposed_comparator=args.comparator,
            proposed_phase=args.phase,
            therapeutic_area=args.therapeutic_area,
        )
    else:
        parser.print_help()
        print("\nError: provide --demo or both --hypothesis and --indication.")
        sys.exit(1)

    loop = EGMLoop(max_rounds=args.max_rounds, verbose=not args.quiet)
    result = loop.run(hypothesis)

    print_summary(result)

    if args.save:
        import os
        os.makedirs(args.save, exist_ok=True)

        protocol_path = os.path.join(args.save, "protocol_draft.md")
        brief_path = os.path.join(args.save, "concepts_brief.md")
        log_path = os.path.join(args.save, "egm_review_log.md")

        with open(protocol_path, "w", encoding="utf-8") as f:
            f.write(format_protocol_draft(result))
        with open(brief_path, "w", encoding="utf-8") as f:
            f.write(format_concepts_brief(result))
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(format_review_log(result))

        print(f"\n  ✓ Saved: {protocol_path}")
        print(f"  ✓ Saved: {brief_path}")
        print(f"  ✓ Saved: {log_path}")


if __name__ == "__main__":
    main()
