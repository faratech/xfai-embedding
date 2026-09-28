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
from collections import OrderedDict
import hashlib
import threading
import time
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
# Every OpenAI call this service makes lands one row in wf_ai_usage_log via the
# aiapi tracker (loopback-trusted). ~1,200 calls/day were invisible to the
# ledger until 2026-09-18.
TOKEN_TRACK_URL = os.getenv("TOKEN_TRACK_URL", "http://127.0.0.1:8001/token/track")
MAX_TOKENS = 8191  # text-embedding-3-small context limit
MAX_BATCH_TOKENS = 200_000  # API limit is 300k; legacy corpus counts ~30% higher server-side
MAX_BATCH_RETRIES = 6
BATCH_REQUEST_DELAY_SECONDS = 2.0

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

# Deterministic in-memory LRU cache for cross-encoder scores: key -> score (float)
_RERANK_CACHE = OrderedDict()
_RERANK_CACHE_MAX = 50_000
_RERANK_CACHE_LOCK = threading.Lock()


def _get_pair_cache_key(query: str, doc_text: str) -> str:
    h = hashlib.sha256()
    h.update(query.strip().lower().encode("utf-8"))
    h.update(b"\x00")
    h.update(doc_text.encode("utf-8"))
    return h.hexdigest()


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


class RerankPair(BaseModel):
    id: Optional[str] = Field(None, description="Optional caller id echoed back")
    query: str = Field(..., min_length=1, max_length=1000, description="Query text")
    document: str = Field(..., min_length=1, max_length=5000, description="Document text")


class RerankRequest(BaseModel):
    query: Optional[str] = Field(None, min_length=1, max_length=1000, description="Search query")
    documents: Optional[List[RerankDocument]] = Field(None, min_length=1, max_length=100, description="Documents to rerank")
    # Independent (query, document) pairs — audit 2026-09-03 #373: the news
    # dedupe gate scores BOTH directions of every pair, so per-pair requests
    # doubled the per-request overhead. Mutually exclusive with query/documents.
    pairs: Optional[List[RerankPair]] = Field(None, min_length=1, max_length=100, description="Independent query/document pairs to score")
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
            http2=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50)
        )
    return HTTP_CLIENT


async def _meter_embedding_usage(data: dict, request_source: str, n_inputs: int) -> None:
    """Best-effort ledger row for one embeddings response; never raises."""
    try:
        usage = data.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
        if prompt <= 0:
            return
        client = await get_http_client()
        await client.post(TOKEN_TRACK_URL, json={
            "user_id": 125694,
            "model": data.get("model") or OPENAI_MODEL,
            "input_text": f"{n_inputs} embedding input(s)",
            "output_text": "",
            "request_source": request_source,
            "usage_data": {"input_tokens": prompt, "output_tokens": 0, "total_tokens": prompt},
        }, timeout=5.0)
    except Exception as exc:
        logger.warning("embedding usage metering failed: %s", exc)


async def get_openai_embedding(text: str) -> List[float]:
    """Get embedding from OpenAI API."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    text = truncate_to_tokens(text)
    client = await get_http_client()
    started = time.monotonic()
    try:
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
    except httpx.HTTPError as e:
        logger.warning("Embedding upstream error type=%s elapsed_ms=%d", type(e).__name__, round((time.monotonic() - started) * 1000))
        raise
    elapsed_ms = round((time.monotonic() - started) * 1000)
    if elapsed_ms >= 1000:
        logger.warning("Embedding upstream slow status=%d elapsed_ms=%d", response.status_code, elapsed_ms)

    if response.status_code != 200:
        logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {response.status_code}")

    data = response.json()
    asyncio.ensure_future(_meter_embedding_usage(data, "xfai_embed", 1))
    return data["data"][0]["embedding"]


async def get_openai_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts from OpenAI API."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    texts = [truncate_to_tokens(t) for t in texts]
    client = await get_http_client()
    token_counts = [len(TOKENIZER.encode(text)) for text in texts]
    batches: List[List[str]] = []
    current_batch: List[str] = []
    current_tokens = 0

    for text, token_count in zip(texts, token_counts):
        if current_batch and current_tokens + token_count > MAX_BATCH_TOKENS:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(text)
        current_tokens += token_count

    if current_batch:
        batches.append(current_batch)

    embeddings: List[List[float]] = []
    for batch_index, batch in enumerate(batches):
        response = None
        for attempt in range(MAX_BATCH_RETRIES):
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENAI_MODEL,
                    "input": batch
                },
                timeout=60.0
            )

            if response.status_code != 429:
                break

            retry_after = response.headers.get("retry-after")
            try:
                delay = max(1.0, float(retry_after)) if retry_after else min(8.0, 2.0 ** attempt)
            except ValueError:
                delay = min(8.0, 2.0 ** attempt)
            logger.warning("Embedding rate limit reached; retrying batch in %.1fs", delay)
            await asyncio.sleep(delay)

        if response is None or response.status_code != 200:
            status = response.status_code if response is not None else 502
            detail = response.text if response is not None else "No response"
            logger.error(f"OpenAI API error: {status} - {detail}")
            raise HTTPException(status_code=502, detail=f"OpenAI API error: {status}")

        data = response.json()
        asyncio.ensure_future(_meter_embedding_usage(data, "xfai_embed_batch", len(batch)))
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        embeddings.extend(item["embedding"] for item in sorted_data)
        if batch_index + 1 < len(batches):
            await asyncio.sleep(BATCH_REQUEST_DELAY_SECONDS)

    return embeddings


def load_rerank_model():
    """Load cross-encoder model for reranking."""
    global RERANK_MODEL
    try:
        import torch
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        logger.info("Configured PyTorch to use 4 threads and disabled gradients globally")
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


def _malloc_trim() -> None:
    """Release freed glibc heap pages back to the OS."""
    import ctypes
    try:
        libc = ctypes.CDLL(None)
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass


async def memory_trimmer_task(interval: int = 300):
    """Periodically trim glibc heap. No gc.collect() — Python's refcount GC
    handles dead objects synchronously; forcing a full cycle collection here
    has no upside for this service and risks unexpected finalizer behavior."""
    while True:
        await asyncio.sleep(interval)
        try:
            _malloc_trim()
            logger.debug("Periodic heap trim completed.")
        except Exception:
            pass


async def openai_connection_keepalive(interval: int = 45):
    """Keep HTTP/2 connection alive to api.openai.com to avoid TLS handshake penalty."""
    while True:
        await asyncio.sleep(interval)
        try:
            if OPENAI_API_KEY:
                client = await get_http_client()
                await client.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}, timeout=5.0)
        except Exception:
            pass


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

    # Start periodic memory trimmer
    trim_task = asyncio.create_task(memory_trimmer_task(300))
    app.state.memory_trimmer = trim_task

    # Start HTTP/2 keepalive task to OpenAI
    keepalive_task = asyncio.create_task(openai_connection_keepalive(45))
    app.state.keepalive_task = keepalive_task

    # Release model-loading memory spikes back to the OS
    _malloc_trim()

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
    """Run cross-encoder inference with deterministic LRU score cache."""
    if RERANK_MODEL is None:
        raise ValueError("Cross-encoder model not loaded")

    scores = [None] * len(documents)
    uncached_indices = []
    uncached_pairs = []

    # Check LRU cache first (< 0.01ms)
    with _RERANK_CACHE_LOCK:
        for i, doc in enumerate(documents):
            key = _get_pair_cache_key(query, doc.text)
            if key in _RERANK_CACHE:
                scores[i] = _RERANK_CACHE[key]
                _RERANK_CACHE.move_to_end(key)
            else:
                uncached_indices.append(i)
                uncached_pairs.append((query, doc.text))

    # Run inference only on uncached pairs
    if uncached_pairs:
        computed_scores = RERANK_MODEL.predict(uncached_pairs, show_progress_bar=False)
        with _RERANK_CACHE_LOCK:
            for idx, (q, txt), sc in zip(uncached_indices, uncached_pairs, computed_scores):
                score_val = float(sc)
                scores[idx] = score_val
                key = _get_pair_cache_key(q, txt)
                _RERANK_CACHE[key] = score_val
                if len(_RERANK_CACHE) > _RERANK_CACHE_MAX:
                    _RERANK_CACHE.popitem(last=False)

    # Create results with original indices
    results = []
    for i, (doc, score) in enumerate(zip(documents, scores)):
        results.append(RerankResult(
            id=doc.id,
            score=float(score if score is not None else 0.0),
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

    if request.pairs and (request.documents or request.query):
        raise HTTPException(
            status_code=422,
            detail="Use either 'pairs' or 'query'+'documents', not both."
        )
    if request.pairs and not (request.documents and request.query):
        # Independent (query, document) pairs — each scored on its own.
        def score_pairs() -> List[RerankResult]:
            scored = [
                (pair, float(score))
                for pair, score in zip(
                    request.pairs,
                    RERANK_MODEL.predict(
                        [(p.query, p.document) for p in request.pairs],
                        show_progress_bar=False,
                    ),
                )
            ]
            results = [
                RerankResult(id=pair.id if pair.id else str(i), score=score, index=i)
                for i, (pair, score) in enumerate(scored)
            ]
            results.sort(key=lambda r: r.score, reverse=True)
            return results[: request.top_k]

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(THREAD_POOL, score_pairs)
        return RerankResponse(results=results, model=RERANK_MODEL_NAME)

    if not (request.documents and request.query):
        raise HTTPException(
            status_code=422,
            detail="Provide either 'pairs' or both 'query' and 'documents'."
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
