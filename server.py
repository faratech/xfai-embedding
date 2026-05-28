#!/usr/bin/env python3
"""
XFAI Embedding Service - Hybrid OpenAI + Local embeddings.

- Query embeddings: OpenAI text-embedding-3-small (1536-dim) for compatibility with cached docs
- Document embeddings: Local model for batch indexing (not used in hybrid mode)
- Cross-encoder reranking: Local ms-marco model for fast, accurate reranking
"""

import os
import logging
import asyncio
import httpx
from typing import List, Optional
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "text-embedding-3-small"  # 1536 dimensions, matches cached embeddings
MAX_TOKENS = 8191  # text-embedding-3-small context limit

# Load tiktoken encoder once at startup
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def truncate_to_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Truncate text to fit within the token limit using exact token counting."""
    tokens = TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return TOKENIZER.decode(truncated_tokens)

# Load API key from .env if not in environment
if not OPENAI_API_KEY:
    env_file = "/web/.env"
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    OPENAI_API_KEY = line.split("=", 1)[1].strip()
                    break

# Cross-encoder config
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # Higher accuracy reranker (~130ms for 20 docs)
RERANK_MODEL = None  # Loaded at startup
# Cap workers: torch.set_num_threads(1) means parallel reranks just contend
# for one CPU. 4 is plenty for the load-once + per-request inference uses,
# and avoids 8MB virtual stack × dozens of idle threads.
THREAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="xfai-rerank")


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)
    type: str = Field(default="query", pattern="^(query|document)$")


class EmbedBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=500)


class EmbedResponse(BaseModel):
    embedding: List[float]
    dimensions: int


class EmbedBatchResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    count: int


class HealthResponse(BaseModel):
    status: str
    openai_configured: bool
    model: str
    rerank_model: Optional[str] = None


class RerankDocument(BaseModel):
    id: str = Field(..., description="Document ID for tracking")
    text: str = Field(..., min_length=1, max_length=5000, description="Document text to rank")


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    documents: List[RerankDocument] = Field(..., min_length=1, max_length=100, description="Documents to rerank")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of top results to return")


class RerankResult(BaseModel):
    id: str
    score: float
    index: int


class RerankResponse(BaseModel):
    results: List[RerankResult]
    model: str

HTTP_CLIENT: Optional[httpx.AsyncClient] = None

async def get_http_client() -> httpx.AsyncClient:
    global HTTP_CLIENT
    if HTTP_CLIENT is None:
        HTTP_CLIENT = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50)
        )
    return HTTP_CLIENT


async def get_openai_embedding(text: str) -> List[float]:
    """Get embedding from OpenAI API."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    text = truncate_to_tokens(text)
    client = await get_http_client()
    response = await client.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": OPENAI_MODEL,
            "input": text
        },
        timeout=30.0
    )

    if response.status_code != 200:
        logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {response.status_code}")

    data = response.json()
    return data["data"][0]["embedding"]


async def get_openai_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts from OpenAI API."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    texts = [truncate_to_tokens(t) for t in texts]
    client = await get_http_client()
    response = await client.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": OPENAI_MODEL,
            "input": texts
        },
        timeout=60.0
    )

    if response.status_code != 200:
        logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {response.status_code}")

    data = response.json()
    # Sort by index to maintain order
    sorted_data = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]


def load_rerank_model():
    """Load cross-encoder model for reranking."""
    global RERANK_MODEL
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        logger.info("Configured PyTorch to use 1 thread and disabled gradients globally")
    except ImportError:
        logger.warning("torch not installed - threading options skipped")
    except Exception as e:
        logger.warning(f"Failed to configure torch threading: {e}")

    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder model: {RERANK_MODEL_NAME}")
        RERANK_MODEL = CrossEncoder(RERANK_MODEL_NAME, max_length=512)
        logger.info("Cross-encoder model loaded successfully")
        return True
    except ImportError:
        logger.warning("sentence-transformers not installed - reranking disabled")
        return False
    except Exception as e:
        logger.error(f"Failed to load cross-encoder model: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models and check configuration."""
    if OPENAI_API_KEY:
        logger.info(f"OpenAI API configured, using model: {OPENAI_MODEL}")
    else:
        logger.warning("OpenAI API key not found - embeddings will fail")

    # Load cross-encoder model in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(THREAD_POOL, load_rerank_model)

    yield

    # Cleanup
    global HTTP_CLIENT
    if HTTP_CLIENT is not None:
        await HTTP_CLIENT.aclose()
        HTTP_CLIENT = None
        logger.info("Shared HTTP client closed gracefully")
    THREAD_POOL.shutdown(wait=False)


app = FastAPI(
    title="XFAI Embedding Service",
    description="Hybrid embedding service using OpenAI for queries",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health."""
    return HealthResponse(
        status="healthy" if OPENAI_API_KEY else "degraded",
        openai_configured=bool(OPENAI_API_KEY),
        model=OPENAI_MODEL,
        rerank_model=RERANK_MODEL_NAME if RERANK_MODEL else None
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Generate embedding for a single text using OpenAI."""
    try:
        embedding = await get_openai_embedding(request.text)
        return EmbedResponse(
            embedding=embedding,
            dimensions=len(embedding)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed_batch", response_model=EmbedBatchResponse)
async def embed_batch(request: EmbedBatchRequest):
    """Generate embeddings for multiple texts using OpenAI."""
    try:
        texts = [t for t in request.texts if t and t.strip()]
        if not texts:
            raise HTTPException(status_code=400, detail="No valid texts provided")

        embeddings = await get_openai_embeddings_batch(texts)
        return EmbedBatchResponse(
            embeddings=embeddings,
            dimensions=len(embeddings[0]) if embeddings else 0,
            count=len(texts)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_cross_encoder_rerank(query: str, documents: List[RerankDocument], top_k: int) -> List[RerankResult]:
    """Run cross-encoder inference (blocking, called via thread pool)."""
    if RERANK_MODEL is None:
        raise ValueError("Cross-encoder model not loaded")

    # Build query-document pairs
    pairs = [(query, doc.text) for doc in documents]

    # Get scores from cross-encoder
    scores = RERANK_MODEL.predict(pairs)

    # Create results with original indices
    results = []
    for i, (doc, score) in enumerate(zip(documents, scores)):
        results.append(RerankResult(
            id=doc.id,
            score=float(score),
            index=i
        ))

    # Sort by score descending
    results.sort(key=lambda x: x.score, reverse=True)

    # Return top_k results
    return results[:top_k]


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Rerank documents by relevance to query using cross-encoder.

    Uses the ms-marco-MiniLM-L-12-v2 cross-encoder model for
    accurate semantic reranking. Typically ~130ms for 20 documents.
    """
    if RERANK_MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Cross-encoder model not available. Install sentence-transformers to enable reranking."
        )

    try:
        # Run inference in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            THREAD_POOL,
            run_cross_encoder_rerank,
            request.query,
            request.documents,
            request.top_k
        )

        return RerankResponse(
            results=results,
            model=RERANK_MODEL_NAME
        )

    except Exception as e:
        logger.error(f"Rerank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    host = os.getenv("XFAI_EMBED_HOST", "127.0.0.1")
    port = int(os.getenv("XFAI_EMBED_PORT", "8765"))

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
