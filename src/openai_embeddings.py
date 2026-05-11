"""OpenAI embeddings and translation with batching and retry logic."""

import json
import os
from typing import List, Optional, Tuple, NamedTuple, Dict
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError

# Load environment variables from .env file
load_dotenv()


@dataclass
class TranslationResult:
    """Result of Irish-to-English translation."""
    text: str  # The text to use (translated if Irish, original if English)
    original_text: str  # The original input text
    translation_applied: bool  # Whether translation was performed


@dataclass
class SemanticReviewResult:
    """Semantic manual-review decision from GPT validator."""
    manual_review: bool
    reason: str
    confidence: float
    rationale: str


def get_client() -> OpenAI:
    """Get OpenAI client with API key from environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(6),
)
def _embed_batch(
    client: OpenAI,
    texts: List[str],
    model: str,
) -> List[List[float]]:
    """Embed a single batch of texts with retry logic."""
    response = client.embeddings.create(
        input=texts,
        model=model,
    )
    # Sort by index to ensure order matches input
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


def batch_embed(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    client: Optional[OpenAI] = None,
) -> np.ndarray:
    """
    Compute embeddings for a list of texts using OpenAI API.

    Args:
        texts: List of strings to embed.
        model: OpenAI embedding model name.
        batch_size: Number of texts per API call.
        client: Optional OpenAI client (creates one if not provided).

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    if client is None:
        client = get_client()

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Replace empty strings with a placeholder
        batch = [t if t.strip() else "[empty]" for t in batch]
        embeddings = _embed_batch(client, batch, model)
        all_embeddings.extend(embeddings)

    return np.array(all_embeddings)


# ============================================================================
# Irish-to-English Translation
# ============================================================================

# Common Irish words/patterns to detect Irish text
IRISH_INDICATORS = [
    # Common Irish words
    r'\b(agus|ar|an|na|le|i|do|de|a|is|go|ní|tá|bhí|atá|beidh|raibh|mé|tú|sé|sí|muid|sibh|siad)\b',
    # Irish prefixes and suffixes
    r'\b(teach|scoil|garáiste|foirgneamh|tógáil|leathnú|síneadh|athchóiriú|athchóir|oibreacha)\b',
    # Common planning terms in Irish
    r'\b(cead|pleanála|iarratas|forbairt|láithreán|eastát|talamh|baile|contae)\b',
    # Irish place name patterns
    r'\b(baile|cill|dún|ráth|cluain|droim|lios|ceann|carraig)\b',
]

# Compiled regex for efficiency
import re
_IRISH_PATTERN = re.compile('|'.join(IRISH_INDICATORS), re.IGNORECASE)


def is_likely_irish(text: str) -> bool:
    """
    Detect if text is likely in Irish (Gaeilge).

    Uses a heuristic based on common Irish words and patterns.
    Returns True if the text appears to be Irish, False otherwise.
    """
    if not text or not text.strip():
        return False

    # Count Irish indicator matches
    matches = _IRISH_PATTERN.findall(text.lower())

    # If we find multiple Irish words, it's likely Irish
    # Use a threshold based on text length
    word_count = len(text.split())
    match_count = len(matches)

    # If more than 20% of words are Irish indicators, treat as Irish
    if word_count > 0 and match_count / word_count > 0.2:
        return True

    # Or if we find at least 3 Irish-specific words
    if match_count >= 3:
        return True

    return False


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(6),
)
def _translate_text(
    client: OpenAI,
    text: str,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Translate Irish text to English using OpenAI.

    Does NOT summarize or rewrite - only translates faithfully.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a translator. Translate the following Irish (Gaeilge) text to English. "
                    "Do NOT summarize, interpret, or rewrite the text. "
                    "Provide only a faithful, direct translation. "
                    "If the text is already in English or contains a mix, translate only the Irish portions "
                    "and keep the English portions unchanged."
                )
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0.1,  # Low temperature for consistent translations
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


def translate_irish_to_english(
    text: str,
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini",
) -> TranslationResult:
    """
    Translate Irish text to English if needed.

    Args:
        text: Input text (may be Irish or English)
        client: Optional OpenAI client (creates one if not provided)
        model: OpenAI model to use for translation

    Returns:
        TranslationResult with text, original_text, and translation_applied flag
    """
    if not text or not text.strip():
        return TranslationResult(
            text=text,
            original_text=text,
            translation_applied=False
        )

    # Check if text is likely Irish
    if not is_likely_irish(text):
        return TranslationResult(
            text=text,
            original_text=text,
            translation_applied=False
        )

    # Translate the text
    if client is None:
        client = get_client()

    translated = _translate_text(client, text, model)

    return TranslationResult(
        text=translated,
        original_text=text,
        translation_applied=True
    )


def batch_translate_irish_to_english(
    texts: List[str],
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini",
) -> List[TranslationResult]:
    """
    Translate a batch of texts from Irish to English if needed.

    Only texts detected as Irish will be translated.
    English texts pass through unchanged.

    Args:
        texts: List of input texts
        client: Optional OpenAI client
        model: OpenAI model to use for translation

    Returns:
        List of TranslationResult objects
    """
    if client is None:
        client = get_client()

    results = []
    for text in texts:
        result = translate_irish_to_english(text, client, model)
        results.append(result)

    return results


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(6),
)
def _semantic_review_batch(
    client: OpenAI,
    texts: List[str],
    model: str = "gpt-4o-mini",
) -> List[dict]:
    """
    Ask GPT model to classify semantic ambiguity for manual review in batch.

    Returns list of dicts with keys:
      - index
      - category
      - confidence
      - rationale
    """
    payload = [{"index": i, "description": t} for i, t in enumerate(texts)]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict planning-application ambiguity classifier. "
                    "Determine whether each description should be routed to manual review. "
                    "Return ONLY valid JSON array with one object per input item. "
                    "Each object must have keys: index (int), category (string), confidence (0-1), rationale (string). "
                    "Allowed category values: clear, prior_permission_reference, amendment_or_variation, "
                    "retention_or_condition_dependency, other_ambiguity, uncertain. "
                    "Mark prior approvals, amendments, variations, section 54 references, retention dependencies, "
                    "or condition-linked language as non-clear categories."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=True),
            },
        ],
        temperature=0.0,
        max_tokens=4000,
    )

    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise ValueError("Empty semantic review response")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Best-effort recovery if model wraps JSON in prose.
        start = content.find("[")
        end = content.rfind("]")
        if start == -1 or end == -1 or end < start:
            raise
        parsed = json.loads(content[start:end + 1])

    if not isinstance(parsed, list):
        raise ValueError("Semantic review response must be a JSON array")

    return parsed


def _coerce_semantic_item(item: dict) -> SemanticReviewResult:
    """Normalize a semantic-review item into stable output with balanced escalation."""
    category = str(item.get("category", "uncertain")).strip().lower()
    rationale = str(item.get("rationale", "")).strip()

    try:
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    ambiguous_categories = {
        "prior_permission_reference",
        "amendment_or_variation",
        "retention_or_condition_dependency",
        "other_ambiguity",
        "uncertain",
    }

    if category in ambiguous_categories:
        return SemanticReviewResult(
            manual_review=True,
            reason=f"ai_{category}",
            confidence=confidence,
            rationale=rationale,
        )

    # Balanced policy: clear + low confidence still escalates.
    if category == "clear" and confidence < 0.70:
        return SemanticReviewResult(
            manual_review=True,
            reason="ai_low_confidence",
            confidence=confidence,
            rationale=rationale,
        )

    return SemanticReviewResult(
        manual_review=False,
        reason="",
        confidence=confidence,
        rationale=rationale,
    )


def batch_semantic_manual_review(
    texts: List[str],
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini",
    batch_size: int = 40,
) -> List[SemanticReviewResult]:
    """
    Run semantic manual-review classification over all descriptions in batches.

    Safety behavior:
      - On parsing or API failure for a batch, all rows in that batch are escalated.
    """
    if client is None:
        client = get_client()

    all_results: List[SemanticReviewResult] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        safe_batch = [t if str(t).strip() else "[empty]" for t in batch]

        try:
            raw_items = _semantic_review_batch(client, safe_batch, model=model)
            # Default each row to uncertain in case of missing indices.
            indexed: Dict[int, SemanticReviewResult] = {
                j: SemanticReviewResult(
                    manual_review=True,
                    reason="ai_uncertain",
                    confidence=0.0,
                    rationale="No item returned for index",
                )
                for j in range(len(batch))
            }

            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item.get("index"))
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(batch):
                    indexed[idx] = _coerce_semantic_item(item)

            for j in range(len(batch)):
                all_results.append(indexed[j])

        except Exception as exc:  # pragma: no cover - defensive fallback path
            fallback = SemanticReviewResult(
                manual_review=True,
                reason="ai_validation_fallback",
                confidence=0.0,
                rationale=f"AI validation failed: {type(exc).__name__}",
            )
            all_results.extend([fallback] * len(batch))

    return all_results
