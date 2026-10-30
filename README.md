# 🔬 EGM Clinical Protocol Designer

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ClinicalTrials.gov](https://img.shields.io/badge/data-ClinicalTrials.gov-orange.svg)](https://clinicaltrials.gov)
[![EGM](https://img.shields.io/badge/experiment-Google%20EGM-purple.svg)](https://github.com/google/elephant-goldfish-model)

> ⚠️ **This project is an experiment with [Google's Elephant-Goldfish Model (EGM)](https://github.com/google/elephant-goldfish-model)** — a design-first AI development framework released in October 2026 (Apache-2.0). EGM pairs a long-context **Elephant** designer agent with stateless **Goldfish** critic agents in an adversarial review loop. This Python implementation applies the EGM pattern to the domain of clinical trial protocol design.

---

## The Problem

**86% of clinical trials fail to meet their recruitment targets**, partly because the underlying protocol design was never subjected to structured adversarial review before the trial launched. Biostatistical errors, missing safety monitoring plans, and regulatory non-compliance are caught late — after the IND is filed, after the IRB has reviewed, or (worst) after the trial has begun enrolling patients.

The EGM pattern offers a natural fit: force an independent critic to attack the design *before* it costs anything to fix.

---

## How It Works — EGM Architecture

```
Research Hypothesis
        │
        ▼
┌────────────────────┐
│   ElephantAgent    │  ← Loads context from ClinicalTrials.gov + PubMed
│  (long-context)    │    Synthesises evidence → protocol_draft.md v1.0
└────────────────────┘
        │
        ▼ ──────────────────────────────────────────────────────────────┐
┌──────────────────────────┐                                            │
│  Biostatistician Goldfish│  Phase 1: Comprehension Check              │
│  (stateless critic)      │  Phase 2: Antagonistic Review              │
└──────────────────────────┘  Phase 3: Implementation Readiness         │
        │                                                               │
┌──────────────────────────┐                                            │
│  Patient Safety Goldfish │  ← Same 3-phase EGM review                │
└──────────────────────────┘                                            │
        │                                                               │
┌──────────────────────────┐                                            │
│  Regulatory Goldfish     │  ← GCP, ICH E6, FDA guidance alignment     │
└──────────────────────────┘                                            │
        │                                                               │
        ▼                                                               │
  All passed? ──── No ──→ Elephant resolves critiques → bump version ──┘
        │
       Yes
        │
        ▼
┌────────────────────┐
│  protocol_draft.md │  ← Primary EGM output artifact
│  concepts_brief.md │  ← What was considered & rejected
│  egm_review_log.md │  ← Full Goldfish review history
└────────────────────┘
```

### The Three Goldfish Critics

| Goldfish | Attacks |
|---|---|
| **Biostatistician** | Primary endpoint clarity, sample size, α, power, dropout rate, secondary endpoints |
| **Patient Safety** | DSMB requirement, predefined stopping rules, AE monitoring, organ function exclusions, pregnancy |
| **Regulatory** | ICH E6 GCP compliance, informed consent process, phase appropriateness, prior evidence justification, trial duration |

Each Goldfish is **stateless** — it has no memory of prior rounds. This mirrors EGM's core principle: the critic cannot be slowly convinced by the designer's prior reasoning; it evaluates each version fresh.

---

## Installation

```bash
git clone https://github.com/ai-summer2025/egm-clinical-protocol-designer.git
cd egm-clinical-protocol-designer
pip install -r requirements.txt
```

No API keys required. Uses [ClinicalTrials.gov v2 API](https://clinicaltrials.gov/data-api/api) and [NCBI PubMed](https://www.ncbi.nlm.nih.gov/home/develop/api/) (both free and public). Falls back to curated mock data when offline.

---

## Usage

```bash
# NSCLC immunotherapy combo (default demo)
python main.py --demo

# Heart failure SGLT2 inhibitor demo
python main.py --demo hf

# Custom hypothesis
python main.py \
  --hypothesis "Does Drug-X improve 12-month PFS vs placebo in NSCLC?" \
  --indication "Non-small cell lung cancer" \
  --phase "Phase II" \
  --intervention "Drug-X 200mg IV Q3W" \
  --comparator "Placebo" \
  --area oncology

# Save all three EGM output artifacts
python main.py --demo --save output/
```

### Example Output

```
========================================================================
  EGM CLINICAL PROTOCOL DESIGNER  —  Experiment with Google's EGM
========================================================================

  Hypothesis: Does Pembrolizumab + nab-Paclitaxel improve PFS...
  Indication: Non-small cell lung cancer

[Elephant] Loading research context...
  ✓ Context: 2 trial(s) from ClinicalTrials.gov, 0 publication(s) from PubMed

[Elephant] Drafting initial protocol v1.0 ...
  ✓ Phase: Phase II | Design: Randomised, double-blind...
  ✓ Primary endpoint: Progression-Free Survival (PFS) at 12 months
  ✓ Sample size: 128 per arm (300 total enrolled)

────────────────────────────────────────────────────────────────────────
  EGM REVIEW ROUND 1/3  |  Protocol v1.0
────────────────────────────────────────────────────────────────────────

  [Biostatistician Goldfish]
    Comprehension: ✓  — Reviewing Phase II, n=128 per arm, α=0.05, power=80%.
    Antagonistic:  0 critical | 1 warnings | 0 info
    Readiness:     ✅ Ready

  [Patient Safety Goldfish]
    Comprehension: ✓  — DSMB=required, stopping rules: 3.
    Antagonistic:  0 critical | 0 warnings | 0 info
    Readiness:     ✅ Ready

  [Regulatory Goldfish]
    Comprehension: ✓  — Phase: Phase II, GCP: present.
    Antagonistic:  0 critical | 0 warnings | 0 info
    Readiness:     ✅ Ready

========================================================================
  ✅ All Goldfish critics passed — protocol finalised after round 1.
========================================================================
```

---

## Output Artifacts

The `--save` flag produces three files following EGM's naming convention:

| File | Contents |
|---|---|
| `protocol_draft.md` | Full structured protocol: endpoints, population, statistical plan, safety monitoring, GCP compliance |
| `concepts_brief.md` | What the Elephant considered, what the Goldfish critics rejected, key design decisions |
| `egm_review_log.md` | Per-round Goldfish review history with all critiques and resolutions |

---

## Why EGM Fits Clinical Trial Design

The Elephant-Goldfish pattern was designed for situations where:

1. **Context is vast** — the Elephant must hold hundreds of trials, publications, and FDA guidance simultaneously while drafting the protocol
2. **Critics need independence** — a Goldfish that remembered the Elephant's draft rationale could be inadvertently anchored; stateless review catches blind spots
3. **Multiple expert perspectives matter** — a biostatistician and a patient safety officer and a regulatory expert each have genuinely different failure modes to probe

This is exactly the structure of peer review in clinical trial design, which the EGM loop automates.

---

## Running Tests

```bash
python -m pytest test_main.py -v
```

All 32 tests pass in under 15 seconds with no external API calls.

---

## Project Structure

```
egm-clinical-protocol-designer/
├── main.py            # EGM framework + all agents + CLI
├── test_main.py       # 32 unit tests covering all agents + E2E
├── requirements.txt   # pytest only (stdlib for everything else)
├── .gitignore
└── README.md
```

---

## Data Sources

- **ClinicalTrials.gov v2 API** — Free, public, NIH/NLM. No key required.
- **NCBI PubMed E-utilities** — Free, public. No key required.
- All demo patient/trial data is **entirely synthetic**.

---

## Extending the System

```python
# Add a new Goldfish critic by subclassing GoldfishCritic:
class PharmacokineticGoldfish(GoldfishCritic):
    """Probes PK/PD assumptions, dosing rationale, and drug-drug interactions."""
    persona = "Pharmacokineticist"

    def comprehension_check(self, protocol): ...
    def antagonistic_review(self, protocol): ...
```

Other ideas:
- **EHR integration**: Replace demo hypotheses with FHIR-formatted clinical context
- **LLM Elephant**: Swap rule-based synthesis for a real LLM generating narrative rationale
- **IRB pre-check**: Add a Goldfish that validates against institution-specific IRB templates
- **Phase I dose-escalation design**: Extend to generate 3+3 or Bayesian Optimal Interval (BOIN) designs

---

## Relation to Google's EGM Library

This project implements the core EGM workflow pattern in standalone Python:

| EGM Skill | This Implementation |
|---|---|
| `brainstorm` | `ElephantAgent.load_context()` — pulls ClinicalTrials.gov + PubMed |
| `draft` | `ElephantAgent.draft_protocol()` — synthesises evidence into protocol |
| `review-loop` | `EGMLoop.run()` — iterates until all Goldfishes pass |
| `review` | `GoldfishCritic.review()` — 3-phase stateless critique |
| `resolve` | `ElephantAgent.resolve_critiques()` — addresses critical issues |
| `bootstrap` | `_make_good_protocol()` / `DEMO_HYPOTHESES` — starting contexts |

The EGM library itself (a Claude Code plugin) can be used to design and evolve this system's own architecture using the `drift-report` skill to detect protocol drift across sessions.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Research Sources

- Google Open Source Blog: *Elephant-Goldfish Model* (October 2026) — https://github.com/google/elephant-goldfish-model
- NIH ClinicalTrials.gov v2 API documentation: https://clinicaltrials.gov/data-api/api
- ICH E6(R2) Good Clinical Practice Guideline: https://www.ich.org/page/efficacy-guidelines
- Tufts Center for Drug Development: *Protocol Amendment Rates* (2021)
- FDA: *Guidance for Industry — Adaptive Designs for Clinical Trials* (2019)

---

*Built with Python | No paid APIs | Works offline via curated mock data | Experiment with Google's EGM*
