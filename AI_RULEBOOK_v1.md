# AI Classification Rule Book v1.0

**Date:** 2026-04-27
**Scope:** Sector classification + Garage tag prediction
**Model:** GPT-4o-mini (temperature=0, structured JSON output)
**Data Source:** 45,111 Irish planning application descriptions
**Target Accuracy:** Sector ≥95%, Garage F1 non-regressive

---

## 1. VALID OUTPUTS

### Sector Classifications (10 valid labels)
1. **Agriculture** – Farm buildings, agricultural sheds, silage facilities, slurry tanks, milking parlors
2. **Civil** – Infrastructure: roads, carpools, roundabouts, power generation, water/sewerage systems
3. **Commercial & Retail** – Retail units, offices, bars/restaurants, hotels, service stations, car showrooms
4. **Education** – Schools, universities, colleges, classrooms, educational facilities
5. **Industrial** – Factories, warehouses, data centers, light industrial, processing facilities
6. **Medical** – Hospitals, care homes, medical centers, clinics
7. **Miscellaneous** – Regulatory amendments, retention-only permissions, change of use without construction, decommissioning, signage, fence work, non-construction modifications
8. **Residential** – Multi-unit dwellings, apartment blocks, mixed-use residential, student accommodation
9. **Self Build** – Single detached dwellings, house extensions, alterations, domestic garages, single-family renovations
10. **Social** – Churches, community halls, museums, sports facilities, gaa clubs, public buildings

### Tag Classifications (binary: Garage vs None)
- **garage** – Explicit domestic garage, carport, or covered parking structure for residential use
- **none** – Any description that is not a garage

---

## 2. DECISION HIERARCHY & LOGIC

### Step 1: Identify Core Intent
Read the description and classify its **primary construction/use purpose**:

| Intent Signal | Indicative Sector |
|---|---|
| Single dwelling + modifications | Self Build |
| Multiple units, apartment blocks, mixed residential | Residential |
| Agricultural operations, farm sheds | Agriculture |
| Retail, food/beverage, office, service station | Commercial & Retail |
| School, college, university | Education |
| Factory, warehouse, processing, manufacturing | Industrial |
| Hospital, care facility, medical clinic | Medical |
| Church, community, sports, museum, public facility | Social |
| Road, carpark, power plant, sewerage infrastructure | Civil |
| Permission for prior-approved modification, retention of existing, change of use w/o construction, administrative/regulatory items | Miscellaneous |

### Step 2: Construction vs Non-Construction
**Construction-based signals** (presence indicates active development):
- Words: construction, erection, new build, new dwelling, new building, new extension, installation, creation, demolition, alteration, conversion
- Presence of these = **construction intent exists**

**Non-construction signals** (presence indicates regulatory or administrative activity):
- Words: retention, change of use, regularization, amendment, modification, variation of condition, subject to condition, previous planning, approved under, registered office, signage
- Absence of construction words + presence of these = **non-construction (likely Miscellaneous)**

### Step 3: Boundary Resolution
When multiple sectors are plausible, apply these tie-breakers in order:

#### Domestic Dwelling Tie-breaker
- Single dwelling + extensions/alterations + domestic garage mention = **Self Build** (not Residential)
- Multiple dwellings explicitly mentioned = **Residential**

#### Miscellaneous Tie-breaker
- No explicit construction words AND (retention OR change of use OR prior-approval phrasing) = **Miscellaneous**
- High confidence in another sector + retention mention = **keep primary sector** (retention is coexistent activity)

#### Miscellaneous Procedural Override (High Priority)
- If wording indicates procedural modification of previously approved works, classify as **Miscellaneous** even when sector keywords appear.
- Strong procedural signals include:
  - "change of design"
  - "change of house type"
  - "amendment to" / "amendments to"
  - "revisions"
  - "previously approved" / "approved under"
  - "retention permission"
- Infrastructure-only adjustment language without explicit primary building construction should be **Miscellaneous** (not Civil), e.g.:
  - "new vehicular entrance"
  - "boundary wall"
  - "gates / fencing"
  - "site works"
- If both construction and procedural signals coexist, treat as mixed-signal and bias toward **Miscellaneous** unless explicit new primary construction is dominant.
- Policy/reference framing ("under policy", "planning ref", "reg ref") without clear new-build detail defaults to **Miscellaneous**.

#### Ambiguous Planning Reference Tie-breaker
- Explicit reference like "planning ref 1234" or "reg no 5678" without clear intent = ambiguous, default to primary signal
- Prior-approval language ("previously approved under", "subject to condition", "amendment to") = context matters (check for construction)

### Step 4: Garage Tag Logic
**Garage tag ONLY applies within Self Build context.**

- If **sector is NOT Self Build**, tag = **none**
- If **sector is Self Build**, search for garage-specific language:
  - **Positive signals for garage:**
    - "domestic garage"
    - "detached garage"
    - "carport"
    - "covered parking"
    - "garage" (in residential context, not factory/warehouse)
    - "attached garage" OR "garage extension"
  - **Negative signals (not garage):**
    - Warehouse, factory, industrial storage
    - Covered shed, agricultural shed
    - Parking facility (carpark, multi-story)
  - **Confidence rule:**
    - Explicit "domestic garage" or "detached garage" mention = high confidence (0.95)
    - Generic "garage" in Self Build context + dwelling mention = medium confidence (0.75-0.85)
    - No garage mention in Self Build = tag = **none**

---

## 3. EXAMPLES & COUNTEREXAMPLES

### Self Build + Garage Examples
✅ "Construction of a single storey extension to the side of existing single storey dwelling, demolition of existing garage, and new detached garage with all associated works"
→ Sector: **Self Build**, Tag: **garage** (explicit "detached garage" in dwelling context)

✅ "Retention permission of a domestic garage and permission for the construction of a dwelling house and a new vehicular entrance"
→ Sector: **Self Build**, Tag: **garage** (explicit "domestic garage" + dwelling construction)

✅ "Two storey extension at the side and rear with alterations to rear and side garden boundary"
→ Sector: **Self Build**, Tag: **none** (no garage mention, extension only)

### Residential Examples
✅ "Construction of 100 no. apartments comprising 20 studios, 40 1-bed, 40 2-bed units"
→ Sector: **Residential**, Tag: **none** (multi-unit, not domestic)

✅ "Demolition of existing derelict cottage and construction of a replacement part two storey detached dwelling, single storey garage, carport"
→ Sector: **Self Build**, Tag: **garage** (single dwelling + "garage" and "carport")

### Miscellaneous Examples
✅ "Amendment to planning ref 24/5678 to amend the approved site layout"
→ Sector: **Miscellaneous**, Tag: **none** (amendment to prior approval, no construction)

✅ "Retention permission for alterations previously approved under ref 22/1234"
→ Sector: **Miscellaneous**, Tag: **none** (retention of prior work, regulatory)

✅ "Change of use from retail unit to office space"
→ Sector: **Miscellaneous**, Tag: **none** (change of use, likely no construction)

### Agriculture Examples
✅ "Construction of a steel-framed agricultural shed to house a rotary milking parlour"
→ Sector: **Agriculture**, Tag: **none** (farm building, not domestic)

### Civil Examples
✅ "Extension to existing carpark with 500 additional spaces, new entrance, lighting"
→ Sector: **Civil**, Tag: **none** (infrastructure, not domestic)

---

## 4. CONFIDENCE RUBRIC

For each prediction, assign a confidence score (0–1) reflecting certainty:

| Confidence | Criteria | Example |
|---|---|---|
| **0.95–1.0** | Sector unambiguous (strong intent signals + no conflicting language); sector + tag alignment clear | "Construction of a 3-bed detached house with domestic garage" |
| **0.85–0.94** | Sector likely but one minor conflicting signal; tag likely given context | "Single storey extension with garage (storage shed)" → slight ambiguity on shed vs garage |
| **0.70–0.84** | Sector plausible but tie-breaker applied; some ambiguity remains | "Planning ref 1234 for retention + possible new construction" → default to primary, flag uncertainty |
| **0.50–0.69** | High ambiguity; multiple sectors equally plausible; recommend manual review | "Development for mixed use comprising retail and residential with prior conditions" |
| **<0.50** | Insufficient signals; description too vague or contradictory; escalate | Empty/corrupted description; mutually exclusive signals |

---

## 5. SPECIAL CASES & EDGE HANDLING

### Empty or Null Descriptions
- Input: `[empty]`, `""`, `null`
- **Output:** sector: `Miscellaneous`, tag: `none`, confidence: 0.0, rationale: "Insufficient description"

### Contradictory Signals
- E.g., "Demolition of residential apartment block to construct single dwelling"
- **Logic:** Favor the construction intent (single dwelling = Self Build)
- **Confidence:** medium (0.75–0.80) due to contradiction

### Prior Permission + New Construction
- E.g., "Permission for retention of approved dwelling + construction of new extension"
- **Logic:** New construction = Self Build (not pure Miscellaneous)
- **Confidence:** 0.85–0.90 (coexistence is valid)

### Ambiguous Planning References
- Reference numbers without clear intent (no construction, retention, or change-of-use language)
- **Logic:** Default to secondary signals; if none, default to Miscellaneous with low confidence
- **Confidence:** 0.60–0.70

---

## 6. OUTPUT CONTRACT (JSON Schema)

```json
{
  "pred_sector": "Self Build",
  "pred_sector_conf": 0.95,
  "pred_tag": "garage",
  "pred_tag_conf": 0.92,
  "rationale": "Single dwelling with explicit domestic garage mention; clear Self Build intent.",
  "rulebook_version": "1.0",
  "timestamp": "2026-04-27T12:34:56Z"
}
```

### Schema Rules
- `pred_sector`: One of the 10 valid labels
- `pred_sector_conf`: Float [0–1], rounded to 2 decimals
- `pred_tag`: One of `["garage", "none"]`
- `pred_tag_conf`: Float [0–1], rounded to 2 decimals
- `rationale`: Plain-text explanation (50–200 characters)
- `rulebook_version`: Always `"1.0"` for this book
- `timestamp`: ISO 8601 UTC

---

## 7. ITERATIVE CALIBRATION NOTES

**Pass A (Baseline):** Use this rulebook as-is.
**Pass B:** After first run, review top-25 sector mismatches and garage false-positives/false-negatives. Refine tie-breakers and examples.
**Pass C:** Hardening pass for residual errors after Pass B.

**Acceptance Criteria:**
- Sector accuracy ≥95% on test split
- Garage F1 ≥ (baseline - 5%)
- Repeat-run variance < 2% (within tolerance)

---

## CHANGELOG
- **v1.0** (2026-04-27): Initial rulebook derived from ML intent + label distribution
