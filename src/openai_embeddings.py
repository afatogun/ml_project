"""OpenAI embeddings with batching and retry logic."""

import os
from typing import List, Optional

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APITimeoutError, APIConnectionError


def get_client() -> OpenAI:
    """Get OpenAI client with API key from environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
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
