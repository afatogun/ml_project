"""OpenAI embeddings and translation with batching and retry logic."""

import os
from typing import List, Optional, Tuple, NamedTuple
from dataclasses import dataclass

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError


@dataclass
class TranslationResult:
    """Result of Irish-to-English translation."""
    text: str  # The text to use (translated if Irish, original if English)
    original_text: str  # The original input text
    translation_applied: bool  # Whether translation was performed


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
