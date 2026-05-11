import re
from typing import Dict, List


PROCEDURAL_SIGNALS = [
    "shec",
    "expp",
    "section 5 declaration",
    "declaration is sought",
    "outline",
    "renewal",
    "previously approved",
    "previously permitted",
    "parent permission",
    "planning ref",
    "reg ref",
    "amendment",
    "amendment to planning permission",
    "amendment to condition",
    "amended scheme",
    "variation",
    "modification of condition",
    "modification to planning permission",
    "modification of permitted development",
    "modification to previously approved permission",
    "revision",
    "change of design",
    "change of house type",
    "change of use",
    "continuation of use",
    "continue use",
    "retention only",
    "retain only",
]

ACTIVE_CONSTRUCTION_SIGNALS = [
    "construct",
    "construction",
    "erect",
    "build",
    "extension",
    "alter",
    "renovate",
    "refurbish",
    "convert",
    "demolish",
    "replacement dwelling",
    "new dwelling",
    "single storey",
    "two storey",
]

SINGLE_DWELLING_SIGNALS = [
    "dwelling house",
    "dwellinghouse",
    "single dwelling",
    "detached dwelling",
    "bungalow",
    "existing dwelling",
    "existing house",
    "cottage",
]

MULTI_UNIT_SIGNALS = [
    "apartments",
    "apartment block",
    "housing development",
    "student accommodation complex",
    "duplex scheme",
    "terraced houses",
]

SITE_STAGE_SIGNALS = [
    "site for dwelling",
    "proposed site",
    "outline",
    "renewal",
    "shec",
]

MINOR_SITE_WORKS_SIGNALS = [
    "entrance",
    "vehicular access",
    "agricultural entrance",
    "gateway",
    "boundary wall",
    "fencing",
    "gates",
    "signage",
    "material alterations",
    "fire safety",
    "car parking canopy",
    "small substation",
    "compact substation",
    "wastewater tank",
    "wastewater treatment tanks",
    "tertiary filter",
]

CHANGE_USE_SIGNALS = [
    "change of use",
    "continuation of use",
    "continue use",
    "hmo",
    "house of multiple occupation",
    "short-term let",
]

PRIOR_PERMISSION_SIGNALS = [
    "previously approved",
    "previously permitted",
    "extant planning reference",
    "planning ref",
    "reg ref",
    "under policy",
]

RETENTION_ONLY_SIGNALS = [
    "retention only",
    "retain only",
]

COMMERCIAL_PROCEDURAL_SIGNALS = [
    "tenant identification signage",
    "signage",
    "material alterations",
    "fire safety",
    "amendment",
    "revision",
    "modification",
    "protected structure",
    "amalgamation of units",
]

AGRI_MINOR_WORKS_SIGNALS = [
    "agricultural entrance",
    "vehicular entrance",
    "gateway",
    "soil",
    "field raising",
    "animal arena",
    "firewood kiln",
]

FARM_BUILDING_SIGNALS = [
    "shed",
    "barn",
    "silo",
    "silage",
    "slurry",
    "milking parlour",
    "poultry house",
    "calf house",
    "grain store",
    "farmyard",
    "cattle shed",
    "agricultural building",
]

CIVIL_MINOR_INFRA_SIGNALS = [
    "small substation",
    "compact substation",
    "esb substation",
    "modular substation",
    "wastewater treatment",
    "wastewater tank",
    "tertiary filter",
    "car parking canopy",
    "car park canopy",
    "canopy with solar",
    "solar canopy",
    "roof solar panels",
    "ev charging",
    "heat pump",
    "met mast",
    "meteorological mast",
    "lattice mast",
    "telecoms mast",
    "telecom mast",
    "telecommunications mast",
]

UTILITY_SCALE_SIGNALS = [
    "mw",
    "megawatt",
    "grid connection",
    "solar farm",
    "bess",
    "wind farm",
    "turbine",
    "quarry",
    "major wastewater treatment plant",
    "public utility",
]

SOCIAL_ASSET_SIGNALS = [
    "community centre",
    "community center",
    "community hall",
    "sports facility",
    "sports grounds",
    "gaa",
    "clubhouse",
    "jogging track",
    "walking track",
    "community garden",
    "public park",
    "playground",
]

SOCIAL_MINOR_ADMIN_SIGNALS = [
    "signage",
    "section 5 declaration",
    "declaration is sought",
    "amendment",
    "revision",
    "modification",
]

GARAGE_ACTIVE_SIGNALS = [
    "domestic garage",
    "detached garage",
    "private garage",
    "garage / fuel store",
    "garage/fuel store",
    "carport",
    "domestic shed",
    "shed/store",
    "new domestic shed",
    "erection of domestic garage",
    "construction of domestic garage",
]

GARAGE_NON_ACTIVE_SIGNALS = [
    "site for dwelling and garage",
    "proposed site for",
    "renewal",
    "change of design",
    "change of house type",
    "previously approved garage",
    "conversion of garage",
    "garage door",
    "garage window",
    "demolish existing garage",
    "retention only",
]

WASTEWATER_ONLY_SIGNALS = [
    "wastewater system",
    "wastewater treatment system",
    "percolation area",
    "percolation field",
    "septic tank",
    "on-site wastewater",
    "onsite wastewater",
    "effluent treatment",
    "polishing filter",
]

_ASSET_TERMS_RE = re.compile(
    r"\b(dwellings?|houses?|bungalows?|cottages?|apartments?|units?|extensions?|offices?|shops?|factory|factories|"
    r"warehouse|shed|barn|clinic|school|hospital|church|chapel|hall|clubhouse|buildings?|"
    r"garages?|carport|structure|facility|centre|center|store|depot|station|hotel|restaurants?|"
    r"pub|bar|developments?|block|complex|floor|room|wing|annex|annexe)\b",
    re.IGNORECASE,
)

_PROCEDURAL_PHRASE_PATTERNS: List[str] = [
    r"\boutline\b",
    r"\brenewal\b",
    r"\bshec\b",
    r"\bexpp\b",
    r"\bsection\s+5\s+declaration\b",
    r"\bdeclaration\s+is\s+sought\b",
    r"\bchange\s+of\s+house\s+type\b",
    r"\bchange\s+of\s+design\b",
    r"\bvariation\s+of\s+condition\b",
    r"\bvariation\s+to\s+(planning|condition|permission)\b",
    r"\bmodification\s+of\s+condition\b",
    r"\bmodification\s+to\s+planning\s+permission\b",
    r"\bmodification\s+of\s+permitted\s+development\b",
    r"\bmodification\s+to\s+previously\s+approved\s+permission\b",
    r"\bamendment\s+to\s+planning\s+permission\b",
    r"\bamendment\s+to\s+condition\b",
    r"\bamended\s+scheme\b",
    r"\bretention\s+only\b",
    r"\bretain\s+only\b",
    r"\bunder\s+p(?:lanning)?\s+policy\b",
    r"\bunder\s+policy\b",
    r"\bplanning\s+ref\b",
    r"\breg(?:ulation)?\s+ref\b",
    r"\bparent\s+permission\b",
    r"\bproposed\s+site\b",
    r"\bsite\s+for\s+dwelling\b",
]
_PROCEDURAL_PHRASE_RES = [re.compile(p, re.IGNORECASE) for p in _PROCEDURAL_PHRASE_PATTERNS]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _contains_any(text: str, terms: List[str]) -> bool:
    return any(term in text for term in terms)


def _collect_signals(text: str, terms: List[str], prefix: str) -> List[str]:
    return [f"{prefix}:{term}" for term in terms if term in text]


def _has_multi_residential_count(text: str) -> bool:
    return _has(
        r"\b([2-9]\d*)\s*(no\.?|nos\.?|number\s+of)?\s*"
        r"(detached\s+|semi-?detached\s+|terraced\s+|two\s+storey\s+|"
        r"one\s+and\s+a\s+half\s+storey\s+)?"
        r"(dwelling\s+houses?|dwellings?|houses?|apartments?|flats?)\b"
        r"|\b([2-9]\d*)\s+houses\s+in\s+total\b"
        r"|\bterrace\s+of\s+([2-9]\d*)\s*(no\.?|nos\.?)?\s*(dwellings?|houses?)\b",
        text,
    )


def _has_multi_unit_signal(text: str) -> bool:
    if _has(
        r"\b(retail|commercial|industrial|office|radio\s+remote|shopping\s+centre)\s+units\b",
        text,
    ):
        return False
    if _has_multi_residential_count(text):
        return True
    return _has(
        r"\b(apartment\s+block|housing\s+development|student\s+accommodation|duplex\s+scheme|"
        r"residential\s+units|dwelling\s+units)\b",
        text,
    ) or _contains_any(text, MULTI_UNIT_SIGNALS)


def _has_single_dwelling_signal(text: str) -> bool:
    if _has_multi_residential_count(text):
        return False
    return _contains_any(text, SINGLE_DWELLING_SIGNALS) or _has(
        r"\b(single|one|1)\s*(no\.?|number)?\s*(dwelling|house)\b|\bdwelling\b",
        text,
    )


def _has_active_construction_signal(text: str) -> bool:
    return _contains_any(text, ACTIVE_CONSTRUCTION_SIGNALS) or _has(
        r"\b(construct|construction|erect|build|extension|alter|renovat|refurbish|convert|demolish|replacement dwelling|completion|completing|proposed\s+\w+\s+dwelling|development\s+comprising)\w*\b",
        text,
    )


def _has_utility_scale_signal(text: str) -> bool:
    return _contains_any(text, UTILITY_SCALE_SIGNALS) or bool(re.search(
        r"\b\d+(\.\d+)?\s*kW(p)?\b|"
        r"\bground[-\s]?mounted\s+solar\b|"
        r"\bsolar\s+p\.?v\.?\s+plant\b|"
        r"\bsolar\s+pv\s+array\b|"
        r"\bsand\s+and\s+gravel\s+pit\b|"
        r"\bgravel\s+pit\b|"
        r"\bextraction\s+footprint\b|"
        r"\bextraction\s+and\s+processing\b|"
        r"\bquarry\b|"
        r"\bhvo\s+generator\b",
        text,
        re.I,
    ))


def _has_procedural_phrases(text: str) -> bool:
    return any(pat.search(text) for pat in _PROCEDURAL_PHRASE_RES)


def _active_work_dominates(text: str) -> bool:
    if not _has_active_construction_signal(text):
        return False
    return bool(_ASSET_TERMS_RE.search(text))


def _procedural_dominates(text: str) -> bool:
    return _has_procedural_phrases(text) and not _active_work_dominates(text)


def _has_commercial_cou_destination(text: str) -> bool:
    return bool(re.search(
        r"\b(office|offices|retail|shop|pharmacy|pharmacist|restaurant|bar|pub|hotel|"
        r"guest\s+house|tourist\s+accommodation|motor\s+sales|service\s+facilit|"
        r"cinema|cafe|coffee\s+shop|takeaway|off-?licence)\b",
        text,
        re.IGNORECASE,
    ))


def _has_residential_cou_result(text: str) -> bool:
    if re.search(
        r"\b[2-9]\d*\s*(no\.?|number\s+of\s+)?\s*(apartment|unit|flat|dwelling)s?\b",
        text,
        re.IGNORECASE,
    ):
        return True
    return bool(re.search(
        r"\b(apartments|residential\s+units|dwelling\s+units)\b",
        text,
        re.IGNORECASE,
    ))


def _is_standalone_wastewater(text: str) -> bool:
    if not _contains_any(text, WASTEWATER_ONLY_SIGNALS):
        return False
    dwelling_construction = bool(re.search(
        r"\b(new\s+dwelling|new\s+house|replacement\s+dwelling"
        r"|extension\s+to\s+(the\s+|existing\s+)?(dwelling|house|cottage|bungalow)"
        r"|renovation\s+of\s+(the\s+|existing\s+)?(dwelling|house|cottage|bungalow)"
        r"|conversion\s+of\s+(the\s+|existing\s+)?(dwelling|house|cottage|bungalow)"
        r"|construction\s+of\s+(a\s+)?(new\s+)?(dwelling|house|cottage|bungalow)"
        r"|new\s+extension|single\s+storey\s+extension|two\s+storey\s+extension"
        r"|rear\s+extension|side\s+extension|front\s+extension)\b",
        text,
        re.IGNORECASE,
    ))
    return not dwelling_construction


def _is_change_house_type_or_design_application(text: str) -> bool:
    return bool(re.search(
        r"\bchange\s+of\s+(house\s+type|design)\b",
        text,
        re.I,
    )) and bool(re.search(
        r"\b(previously\s+(approved|permitted|granted)|planning\s+ref|reg\.?\s*ref|"
        r"extant\s+planning|under\s+planning\s+reference|file\s+ref|"
        r"from\s+la\d+|from\s+[a-z]{1,3}\d{1,4}/\d+)\b",
        text,
        re.I,
    ))


def _has_positive_garage_signal(text: str) -> bool:
    return bool(re.search(
        r"\b(attached\s+garage|detached\s+garage|domestic\s+garage|private\s+garage|"
        r"garage\s*/\s*home\s*gym|garage\s*/\s*store|garage/store|garage\s+and|"
        r"and\s+garage|garage,|garage\b|carport|car\s+port|domestic\s+shed)\b",
        text,
        re.I,
    ))


def _is_pure_garage_non_active_context(text: str) -> bool:
    return bool(re.search(
        r"\b(garage\s+door|garage\s+window|conversion\s+of\s+(the\s+)?garage|"
        r"garage\s+converted\s+habitable|converted\s+garage|demolish\s+existing\s+garage)\b",
        text,
        re.I,
    ))


def _signal_flags(text: str) -> Dict[str, bool]:
    return {
        "procedural": _has_procedural_phrases(text),
        "site_stage": _contains_any(text, SITE_STAGE_SIGNALS),
        "active": _has_active_construction_signal(text),
        "single_dwelling": _has_single_dwelling_signal(text),
        "multi_unit": _has_multi_unit_signal(text),
        "change_use": _contains_any(text, CHANGE_USE_SIGNALS),
        "prior_permission": _contains_any(text, PRIOR_PERMISSION_SIGNALS),
        "retention_only": _contains_any(text, RETENTION_ONLY_SIGNALS),
        "commercial_procedural": _contains_any(text, COMMERCIAL_PROCEDURAL_SIGNALS),
        "agri_minor": _contains_any(text, AGRI_MINOR_WORKS_SIGNALS),
        "farm_building": _contains_any(text, FARM_BUILDING_SIGNALS),
        "civil_minor": _contains_any(text, CIVIL_MINOR_INFRA_SIGNALS),
        "utility_scale": _has_utility_scale_signal(text),
        "social_asset": _contains_any(text, SOCIAL_ASSET_SIGNALS),
        "social_minor_admin": _contains_any(text, SOCIAL_MINOR_ADMIN_SIGNALS),
        "garage_active": _contains_any(text, GARAGE_ACTIVE_SIGNALS),
        "garage_non_active": _contains_any(text, GARAGE_NON_ACTIVE_SIGNALS),
    }


def extract_classifier_features(description: str) -> Dict[str, bool]:
    text = _normalize_text(description)
    flags = _signal_flags(text)
    return {
        "has_change_house_type": _has(r"\bchange\s+of\s+house\s+type\b", text),
        "has_change_design": _has(r"\bchange\s+of\s+design\b", text),
        "has_policy_cty": _has(r"\bpolicy\s+cty\b|\bcty\s*\d+\b", text),
        "has_outline": _has(r"\boutline\b", text),
        "has_renewal": _has(r"\brenewal\b", text),
        "has_section_5": _has(r"\bsection\s+5\b", text),
        "has_retention_only": flags["retention_only"],
        "has_retention_plus_new_work": _has(r"\bretention\b", text) and flags["active"],
        "has_standalone_wastewater": _is_standalone_wastewater(text),
        "has_multi_residential_count": _has_multi_residential_count(text),
        "has_single_dwelling_signal": flags["single_dwelling"],
        "has_commercial_cou_destination": _has_commercial_cou_destination(text),
        "has_residential_cou_result": _has_residential_cou_result(text),
        "has_farm_building": flags["farm_building"],
        "has_agri_minor": flags["agri_minor"],
        "has_civil_minor": flags["civil_minor"],
        "has_utility_scale": flags["utility_scale"],
        "has_social_asset": flags["social_asset"],
        "has_garage_signal": _has_positive_garage_signal(text),
        "has_active_work_dominates": _active_work_dominates(text),
        "has_procedural_dominates": _procedural_dominates(text),
    }