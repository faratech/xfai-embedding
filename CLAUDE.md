# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FastAPI microservice providing text embeddings (OpenAI `text-embedding-3-small`, 1536-dim) and cross-encoder reranking (local `ms-marco-MiniLM-L-12-v2`) for the XFAI XenForo add-on. Runs on port 8765, localhost only.

## Commands

```bash
# Development
python3 server.py

# Production (systemd)
sudo systemctl restart xfai-embedding
sudo systemctl status xfai-embedding
journalctl -u xfai-embedding -f

# Batch operations (run directly, not through the service)
python3 generate_all_embeddings.py --yes       # Full sync: MySQL + ES, 3-phase
python3 generate_all_embeddings.py --dry-run   # Count missing only
python3 bulk_vectorize.py --workers=8 --batch=50  # Parallel ES-only vectorization
python3 fill_missing.py                        # Backfill missing ES vectors via OpenAI
python3 fast_reindex.py                        # Rebuild ES from cached MySQL embeddings (destructive)
python3 full_reindex.py                        # Full rebuild: MySQL -> embeddings -> ES (destructive)
```

## Architecture

### Embedding Pipeline

All embedding text is prepared identically across the codebase: `title + " " + strip_tags(message)`, then truncated to 8191 tokens using the `cl100k_base` tokenizer. The `strip_tags()` helper (duplicated in each script) removes HTML and decodes common entities (`&nbsp;`, `&amp;`, etc.) to match the PHP side.

### Dual Storage

Embeddings live in two stores that must stay in sync:
- **MySQL**: `openai_embeddings` table — binary BLOB (float32 × 1536) or native `VECTOR(1536)` column. Keyed by `post_id` / `thread_id`. Has `content_hash` and `needs_update` for staleness detection.
- **Elasticsearch**: `wf_wf` index, `content_vector` field (dense_vector, 1536-dim). Document IDs are `post-{id}` or `thread-{id}`.

`generate_all_embeddings.py` is the primary sync tool with 3 phases: (1) generate new embeddings for MySQL-missing items and write to both stores, (2) backfill ES from MySQL (free), (3) backfill MySQL from ES (free). It uses adaptive rate limiting with exponential backoff and supports graceful shutdown via SIGINT/SIGTERM.

### Server Concurrency

The cross-encoder model runs blocking inference, so `server.py` dispatches it through a `ThreadPoolExecutor` to avoid blocking the async event loop. OpenAI API calls use `httpx.AsyncClient` natively.

### External Dependencies

- OpenAI API key: read from `OPENAI_API_KEY` env var, falls back to parsing `/web/.env`
- Model cache: `./cache` symlinked to `/opt/xfai-embedding/cache` (HuggingFace models)
- `generate_all_embeddings.py` imports `db_pool` and `embedding_hash_utils` from `/web/fastapi_app`
- Called by `XFAI\Service\EmbeddingClient` in the XenForo add-on at `/web/public_html/src/addons/XFAI/`

## Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check, returns model status |
| `/embed` | POST | Single text embedding |
| `/embed_batch` | POST | Batch embeddings (up to 500 texts) |
| `/rerank` | POST | Cross-encoder reranking (up to 100 docs) |

## Configuration

Environment variables (all optional with defaults):
- `XFAI_EMBED_HOST` (default: `127.0.0.1`)
- `XFAI_EMBED_PORT` (default: `8765`)
- `OPENAI_API_KEY` (or reads from `/web/.env`)
- `HF_HOME` / `TRANSFORMERS_CACHE` — model cache directory
