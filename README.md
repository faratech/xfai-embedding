# xfai-embedding

FastAPI microservice that provides text embeddings and cross-encoder reranking for the XFAI XenForo add-on.

Embeddings use OpenAI `text-embedding-3-small` (1536 dimensions). Reranking uses a local cross-encoder model, `cross-encoder/ms-marco-MiniLM-L-12-v2`. The service listens on localhost only by default.

This repository also contains batch and reindex scripts that keep embeddings in sync between MySQL and Elasticsearch. Those scripts are operational tools for the host that runs XenForo; they are not part of the HTTP service.

## Overview

The service has two jobs:

1. **Embeddings** — `POST /embed` and `POST /embed_batch` proxy text to the OpenAI embeddings API and return 1536-dimensional vectors. The vectors match the embeddings already cached in MySQL and Elasticsearch, so query-time and document-time vectors are comparable.
2. **Reranking** — `POST /rerank` scores documents against a query with a local cross-encoder model. No OpenAI call is made for reranking. The model is loaded once at startup and inference runs in a thread pool so it does not block the async event loop.

Supporting behaviour in [`server.py`](server.py):

- Single and batch inputs are truncated server-side: first to 8,000 characters, then to 8,191 tokens with the `cl100k_base` tokenizer (`tiktoken`).
- Batch embedding splits large requests into sub-batches below a 200,000-token aggregate limit, retries HTTP `429` responses up to 6 times with backoff, and pauses 2 seconds between sub-batches.
- Reranking keeps a deterministic in-memory LRU cache of document-mode scores (50,000 entries). An optional exact ordered-pair cache with async miss coalescing is available for `pairs` mode; see [Configuration](#configuration).
- Every minute the service logs a `rerank_performance` JSON record to the journal with cumulative request/pair counts, cache counters, inference metrics, and process CPU. This is a private journal record; it is not an HTTP endpoint.
- Each OpenAI embeddings response is metered on a best-effort basis to a token-tracking endpoint (see `TOKEN_TRACK_URL` in [Configuration](#configuration)). A metering failure is logged and never fails the embedding request.

## Who it's for

- **XFAI XenForo add-on (primary caller).** The add-on's `XFAI\Service\EmbeddingClient` at `/web/public_html/src/addons/XFAI/` calls this service over loopback. If you are not running that add-on, the HTTP API is still usable as a small localhost embedding/reranking service, but the batch scripts assume the XenForo database and index described below.
- **Operators of the XenForo host.** They run the service under systemd and run the sync/reindex scripts in [Utility scripts](#utility-scripts) directly from a shell.

There is no public or authenticated remote API here: the default bind address is `127.0.0.1`, and the code contains no API-key or auth check for inbound requests. Do not expose it beyond loopback without adding your own access control in front of it.

## Quick start

### Prerequisites

- Python with the packages in [`requirements.txt`](requirements.txt). The file pins minimum versions for `fastapi`, `uvicorn[standard]`, `sentence-transformers`, `numpy`, `pydantic`, `tiktoken`, and `torch`.
- An OpenAI API key. Without it the process still starts, but `/health` reports `degraded` and the embedding endpoints return `503`.
- Disk space for the Hugging Face model cache. The rerank model (`cross-encoder/ms-marco-MiniLM-L-12-v2`, loaded with `max_length=512`) is downloaded by `sentence-transformers` on first start. In this repo `cache` is a symlink to `/opt/xfai-embedding/cache`; on a fresh clone that target may not exist, so set `HF_HOME`/`TRANSFORMERS_CACHE` to a writable directory (see [Configuration](#configuration)).

> **Dependency note (verified against the code):** `server.py` also imports `httpx`, which is **not** listed in [`requirements.txt`](requirements.txt). The batch scripts import further packages that are likewise not in [`requirements.txt`](requirements.txt) — see the note in [Utility scripts](#utility-scripts). On the production host these are provided by the host environment. A bare `pip install -r requirements.txt` in a clean virtual environment is therefore not guaranteed to be sufficient to import `server.py`; this README does not invent versions for the unlisted packages.

### Install and run (development)

From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
python3 server.py
```

`python3 server.py` is the development entry point (also listed in [`CLAUDE.md`](CLAUDE.md)). It calls `uvicorn.run(...)` internally with the host/port from `XFAI_EMBED_HOST`/`XFAI_EMBED_PORT` (defaults `127.0.0.1:8765`), so no separate `uvicorn` command is needed.

If `OPENAI_API_KEY` is not set in the environment, `server.py` falls back to parsing `OPENAI_API_KEY=` from `/web/.env` when that file exists. The rerank performance settings (`XFAI_RERANK_*`) fall back to `/web/.env` and then `/web/.config` when not set in the environment; the first file to define a key wins and environment values always win.

### Smoke test

With the service running:

```bash
curl -s http://127.0.0.1:8765/health
```

Expected shape (values depend on whether the key and model loaded):

```json
{"status": "healthy", "openai_configured": true, "model": "text-embedding-3-small", "rerank_model": "cross-encoder/ms-marco-MiniLM-L-12-v2"}
```

`status` is `healthy` only when an API key is configured **and** the rerank model loaded; otherwise it is `degraded`. If `sentence-transformers`/`torch` are unavailable or the model fails to load, the service still starts, `/rerank` returns `503`, and `rerank_model` is `null`.

Single embedding:

```bash
curl -s -X POST http://127.0.0.1:8765/embed \
  -H 'Content-Type: application/json' \
  -d '{"text": "hello world", "type": "query"}'
```

## Running in production (systemd)

The unit file is [`xfai-embedding.service`](xfai-embedding.service). Verified facts from that file:

- `WorkingDirectory=/web/services/xfai-embedding`, run as user/group `nobody`.
- `ExecStart=/usr/bin/python3.14 server.py` — the service is started with the system Python, not a virtualenv, on the production host.
- Environment: `XFAI_EMBED_HOST=127.0.0.1`, `XFAI_EMBED_PORT=8765`, `HF_HOME=/opt/xfai-embedding/cache`, `TRANSFORMERS_CACHE=/opt/xfai-embedding/cache`, plus single-thread BLAS/OpenMP limits (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) and `MALLOC_ARENA_MAX=2`.
- Hardening: `NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome`, `PrivateTmp`; writable paths are limited to the service directory and the model cache, with `/web/.env` read-only.
- `Restart=always`, `KillSignal=SIGINT`, `TimeoutStopSec=10`; output goes to the journal.

Operator commands (from [`CLAUDE.md`](CLAUDE.md)):

```bash
sudo systemctl restart xfai-embedding
sudo systemctl status xfai-embedding
journalctl -u xfai-embedding -f
```

This README does not document an install command for the unit file because the repository contains none; copy/enable steps depend on the host's existing deployment layout (`/web/services/xfai-embedding`).

## API reference

Base URL (default): `http://127.0.0.1:8765`. All request/response bodies are JSON. Interactive OpenAPI docs are provided by FastAPI at `/docs` by default; the four application endpoints are below.

### `GET /health`

Response model `HealthResponse`:

| Field | Type | Meaning |
|---|---|---|
| `status` | string | `healthy` if API key configured and rerank model loaded, else `degraded` |
| `openai_configured` | bool | Whether an OpenAI API key was found |
| `model` | string | Always `text-embedding-3-small` |
| `rerank_model` | string \| null | Model name if loaded, else `null` |

### `POST /embed`

Request model `EmbedRequest`:

| Field | Type | Constraints |
|---|---|---|
| `text` | string | Required, minimum length 1 |
| `type` | string | Optional, `query` or `document` (default `query`). Accepted for caller compatibility; the current handler embeds via OpenAI either way |

Behaviour: the text is truncated to 8,000 characters and then to 8,191 tokens before the OpenAI call.

Response model `EmbedResponse`: `embedding` (list of floats), `dimensions` (length of the vector, 1536 for the configured model).

Errors: `503` if no OpenAI API key is configured; `502` if the OpenAI API returns a non-200 status; `500` for other failures.

```bash
curl -s -X POST http://127.0.0.1:8765/embed \
  -H 'Content-Type: application/json' \
  -d '{"text": "semantic search over forum posts"}'
```

### `POST /embed_batch`

Request model `EmbedBatchRequest`:

| Field | Type | Constraints |
|---|---|---|
| `texts` | list of strings | Required, 1 to 500 items |

Behaviour: each text is truncated to 8,000 characters, then token-truncated. Any blank/empty text fails the whole request with `422` (`blank or empty text in batch`) rather than silently shifting indices. Internally the list is split into sub-batches under the 200,000-token aggregate limit (see [Overview](#overview)).

Response model `EmbedBatchResponse`: `embeddings` (list of vectors, same order as input), `dimensions`, `count`.

Errors: `422` for validation failures including blank texts; `503`/`502`/`500` as for `/embed`.

```bash
curl -s -X POST http://127.0.0.1:8765/embed_batch \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["first document", "second document"]}'
```

### `POST /rerank`

Two mutually exclusive modes. Sending both returns `422`; sending neither returns `422`. If the cross-encoder model is not loaded, every call returns `503`.

**Mode 1 — `query` + `documents`:**

| Field | Type | Constraints |
|---|---|---|
| `query` | string | 1 to 1,000 characters |
| `documents` | list | 1 to 100 items, each `{id: string, text: string (1–5,000 chars)}` |
| `top_k` | int \| null | Optional, 1 to 100. Defaults to **10** in this mode when omitted |

**Mode 2 — `pairs`** (independent query/document pairs):

| Field | Type | Constraints |
|---|---|---|
| `pairs` | list | 1 to 100 items, each `{id?: string, query: string (1–1,000), document: string (1–5,000)}` |
| `top_k` | int \| null | Optional, 1 to 100. In this mode, omitting `top_k` returns **all** scored pairs (no default truncation) |

Response model `RerankResponse`: `results` (list of `{id, score, index}`, sorted by score descending; `index` is the caller's original position) and `model`. In `pairs` mode a missing pair `id` is echoed back as the string form of its index.

The module docstring in [`server.py`](server.py) notes typical rerank latency of roughly 130 ms for 20 documents.

```bash
curl -s -X POST http://127.0.0.1:8765/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query": "how to reset a password", "documents": [{"id": "post-1", "text": "Use the forgot-password link on the login page."}, {"id": "post-2", "text": "Banana bread recipe."}], "top_k": 2}'
```

```bash
curl -s -X POST http://127.0.0.1:8765/rerank \
  -H 'Content-Type: application/json' \
  -d '{"pairs": [{"id": "a-b", "query": "first query", "document": "some document text"}]}'
```

Internal modules used by `/rerank`: [`pair_score_cache.py`](pair_score_cache.py) (bounded exact ordered-pair cache with cancellation-safe coalescing; only complete finite outputs are cached) and [`rerank_metrics.py`](rerank_metrics.py) (inference counters; overlapping calls count process CPU once). [`es_vector_utils.py`](es_vector_utils.py) is used by the batch scripts, not the server.

## Utility scripts

All scripts below are run directly with `python3`, not through the HTTP service (per [`CLAUDE.md`](CLAUDE.md)). They share the same embedding text preparation: `title + " " + strip_tags(message)`, where the duplicated `strip_tags()` helper removes HTML and decodes common entities (`&nbsp;`, `&amp;`, `&lt;`, `&gt;`, `&quot;`, `&#39;`) to match the PHP side.

> **Extra dependencies (verified from imports, not from [`requirements.txt`](requirements.txt)):** these scripts import packages absent from [`requirements.txt`](requirements.txt), including `openai`, `elasticsearch`, `aiohttp`, `pymysql`, `python-dotenv` (`dotenv`), and `httpx`. `generate_all_embeddings.py` also inserts `/web/fastapi_app` into `sys.path` and imports `db_pool` and `embedding_hash_utils` from there. Versions are not specified anywhere in this repository, so they are omitted here rather than guessed; use the host environment that already runs these scripts.

### `generate_all_embeddings.py` — primary MySQL + Elasticsearch sync

[`generate_all_embeddings.py`](generate_all_embeddings.py) is the main sync tool. One OpenAI call per item writes to both stores:

- **MySQL:** `openai_embeddings` table, embedding stored as a float32 binary blob (`float32 × 1536`), keyed by `post_id`/`thread_id`, with `content_hash` and `needs_update` for staleness detection.
- **Elasticsearch:** index `wf_wf` at `http://localhost:9200`, field `content_vector` (dense_vector, 1536 dimensions), document IDs `post-{id}` / `thread-{id}`. Vectors are L2-normalized before the ES write (the `dot_product` mapping rejects non-unit vectors); MySQL keeps the raw vector.

It runs in three phases: (1) generate embeddings for items missing from MySQL and write to both stores, (2) backfill ES from existing MySQL embeddings (no API calls), (3) backfill MySQL from existing ES embeddings (no API calls). It uses adaptive rate limiting with exponential backoff, saves progress to `/web/services/xfai-embedding/generate_all_progress.json`, and handles `SIGINT`/`SIGTERM` by finishing the current batch. Posts use `title + strip_tags(message)`; threads use the title alone; both are truncated to 8,191 tokens.

```bash
python3 generate_all_embeddings.py --yes       # run with auto-confirm
python3 generate_all_embeddings.py --dry-run   # count missing only, no changes
python3 generate_all_embeddings.py --reset --yes  # clear progress file and run
```

Flags (from `argparse` in the file): `--yes`/`-y` auto-confirm, `--dry-run` count only, `--reset` delete the progress file first. Without `--yes` the script waits for an Enter keypress after printing counts and an estimated API cost. It exits with an error if `OPENAI_API_KEY` is not found in the environment or `/web/.env` (it loads both `/web/.env` and `/web/.config` via `dotenv`).

### `bulk_vectorize.py` — parallel ES-only vectorization via the service

[`bulk_vectorize.py`](bulk_vectorize.py) scans the ES index for documents missing `content_vector`, sends their text to the **running service** at `http://127.0.0.1:8765/embed_batch`, and bulk-updates ES. It bypasses the XenForo job system for throughput. Text is `title + " " + strip_tags(message)[:2000]` (2,000-character message cap, matching the PHP concatenation in the file's docstring). Vectors are L2-normalized before writing.

```bash
python3 bulk_vectorize.py --workers=8 --batch=50
```

Flags: `--index` (default `wf_wf`), `--workers` (default `8`), `--batch` (default `50`). The embedding service must already be running.

### `fill_missing.py` — backfill missing ES vectors via OpenAI directly

[`fill_missing.py`](fill_missing.py) streams ES documents without `content_vector` and updates them, calling the OpenAI API **directly** (not through the local service) in batches of 100. Text is `title + " " + strip_tags(message)`, truncated to 8,000 characters. The API key comes from `OPENAI_API_KEY`, falling back to `/web/.env`. There are no command-line flags; just run it:

```bash
python3 fill_missing.py
```

Vectors are L2-normalized before the ES write. On success it also attempts usage metering via `direct_metering` from `/web/fastapi_app`.

### `full_reindex.py` — full destructive rebuild (MySQL → embeddings → ES)

[`full_reindex.py`](full_reindex.py) deletes and recreates the `wf_wf` index, then re-embeds the whole corpus: it reads threads and posts from MySQL (via `pymysql`, unix socket `/run/mysqld/mysqld.sock`, database `wf_wf`, password parsed from `/web/public_html/src/config.php`), gets vectors from the running service's `/embed_batch`, and indexes documents with the mapping defined in the file (`content_vector`: `dense_vector`, 1536 dims, `dot_product`, `bbq_hnsw`). Text uses the same 2,000-character message cap as `bulk_vectorize.py`.

Safety behaviour in the current code:

- It **refuses to run without `--yes`** (argument-parser error: this deletes the live `wf_wf` index).
- Before deleting anything it takes a non-blocking `flock` on `/run/wf-full-reindex.lock` and exits if another run holds it.
- It performs one real embed preflight call and refuses to recreate the index unless the service returns exactly one 1536-dimensional vector.
- If embedding a batch returns the wrong count mid-run, it aborts rather than indexing vectorless documents.

```bash
python3 full_reindex.py --yes
```

The embedding service must already be running.

### `fast_reindex.py` — retired guard (does not reindex)

[`fast_reindex.py`](fast_reindex.py) is deliberately inert. It never touches MySQL or Elasticsearch: a normal invocation prints usage plus an error to stderr and exits with code `64`, directing you to the supported reindexer (`--help` shows the same retirement notice via `argparse`). Do not try to make this copy reindex.

The message printed by the script itself is:

```text
Use: flock -w 60 /tmp/cron_locks/fast_reindex.lock /web/public_html/src/addons/XFES/fast_reindex.py [arguments]
```

Note: [`CLAUDE.md`](CLAUDE.md) describes the same supported reindexer (`/web/public_html/src/addons/XFES/fast_reindex.py`) but with a different lock path (`/tmp/xf_fast_reindex_cron.lock`) and states every manual invocation, including a full destructive rebuild, must run under that lock. The lock path is therefore host/cron configuration, not something this repository defines consistently — follow your host's cron definition.

## Configuration

### Environment variables

Variables actually read by the code in this repository. “Fallback file” means the value is also looked up in that file when not present in the process environment (environment wins).

| Variable | Default | Read by | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | *(empty)* | `server.py`, `generate_all_embeddings.py`, `fill_missing.py` | OpenAI API key for `text-embedding-3-small`. Falls back to parsing `/web/.env` (all three; `generate_all_embeddings.py` also loads `/web/.config` via `dotenv`) |
| `XFAI_EMBED_HOST` | `127.0.0.1` | `server.py` (`__main__`), systemd unit | Bind host |
| `XFAI_EMBED_PORT` | `8765` | `server.py` (`__main__`), systemd unit | Bind port |
| `TOKEN_TRACK_URL` | `http://127.0.0.1:8001/token/track` | `server.py` | Best-effort usage-metering endpoint for OpenAI responses; failures are only logged |
| `XFAI_RERANK_WORKERS` | `4` | `server.py` | Thread-pool workers for cross-encoder inference. Must be `1`, `2`, or `4`; anything else raises `ValueError` at startup. Falls back to `/web/.env`, then `/web/.config` |
| `XFAI_RERANK_TORCH_THREADS` | `4` | `server.py` | `torch.set_num_threads(...)` value. Must be `1`, `2`, or `4`. Same file fallback as above |
| `XFAI_RERANK_PAIR_CACHE` | `0` (disabled) | `server.py` | Set to `1` to enable the 50,000-entry exact ordered-pair cache with coalescing for `pairs` mode. Same file fallback as above. Document-mode caching is separate and always on |
| `HF_HOME` | *(library default)* | systemd unit / Hugging Face libraries | Model cache directory; the unit sets `/opt/xfai-embedding/cache` |
| `TRANSFORMERS_CACHE` | *(library default)* | systemd unit / Hugging Face libraries | Legacy model cache variable; the unit sets the same `/opt/xfai-embedding/cache` |

Systemd-unit-only variables (from [`xfai-embedding.service`](xfai-embedding.service); they tune native libraries, not Python code in this repo): `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `MALLOC_ARENA_MAX=2`.

Rollback for the rerank performance settings (per [`CLAUDE.md`](CLAUDE.md)): set `XFAI_RERANK_PAIR_CACHE` back to `0`, keep workers/threads at `4`/`4`, and restart only this service.

### Fixed values in code (not environment-configurable)

- OpenAI model `text-embedding-3-small` (1536 dimensions), rerank model `cross-encoder/ms-marco-MiniLM-L-12-v2` with `max_length=512` — [`server.py`](server.py).
- Token limit 8,191 tokens and character bound 8,000 chars for `/embed` and `/embed_batch`; batch aggregate limit 200,000 tokens, up to 6 retries on `429`, 2.0 s between sub-batches — [`server.py`](server.py).
- Elasticsearch `http://localhost:9200`, index `wf_wf`; embedding-service URL `http://127.0.0.1:8765/embed_batch` — hardcoded in the batch scripts.

## Architecture

```text
XenForo (PHP)                          This repository
┌─────────────────────────┐           ┌──────────────────────────────┐
│ XFAI add-on             │  loopback │ server.py (FastAPI, :8765)   │
│ XFAI\Service\           │ ────────► │  /embed, /embed_batch ──► OpenAI text-embedding-3-small
│   EmbeddingClient       │   HTTP    │  /rerank ──► local cross-encoder (ms-marco-MiniLM-L-12-v2)
│ /web/public_html/src/   │           │  /health                     │
│   addons/XFAI/          │           └──────────────────────────────┘
└─────────────────────────┘                      ▲
                                                 │ /embed_batch
MySQL (XenForo DB)                               │
┌─────────────────────────┐           ┌──────────┴───────────────────┐
│ xf_post / xf_thread     │ ◄──────── │ generate_all_embeddings.py   │──► MySQL openai_embeddings
│ openai_embeddings       │           │ bulk_vectorize.py            │──► ES wf_wf.content_vector
│  (float32 blob, 1536)   │           │ fill_missing.py (direct API) │──► ES wf_wf.content_vector
└─────────────────────────┘           │ full_reindex.py (destructive)│──► ES wf_wf (recreated)
Elasticsearch :9200                   │ fast_reindex.py (retired)    │──► exits 64, no-op
┌─────────────────────────┐           └──────────────────────────────┘
│ wf_wf.content_vector    │
│  dense_vector 1536,     │
│  dot_product            │
└─────────────────────────┘
```

How the pieces connect:

- **Request path.** The XenForo XFAI add-on calls the service over loopback HTTP for query/document embeddings and reranking. The PHP client lives outside this repository at `/web/public_html/src/addons/XFAI/` (per [`CLAUDE.md`](CLAUDE.md)); only the Python side is in this repo.
- **Dual storage.** Embeddings are stored twice and must stay in sync: MySQL `openai_embeddings` (raw float32 vectors plus `content_hash`/`needs_update` staleness flags) and Elasticsearch `wf_wf.content_vector` (L2-normalized, because the `dot_product` mapping rejects non-unit vectors — see [`es_vector_utils.py`](es_vector_utils.py)). Document IDs are `post-{id}` and `thread-{id}` in both the scripts and the index.
- **Consistent text preparation.** Every producer builds embedding text the same way — `title + " " + strip_tags(message)` truncated to the token limit — so a vector written by a batch script is comparable to one returned by the service at query time. Caps differ by script where noted above (8,191 tokens in the server and `generate_all_embeddings.py`; 2,000 characters in `bulk_vectorize.py`/`full_reindex.py`; 8,000 characters in `fill_missing.py`).
- **Server concurrency.** Cross-encoder inference is blocking, so `server.py` runs it in a `ThreadPoolExecutor` (`XFAI_RERANK_WORKERS` workers) and keeps OpenAI calls on an async `httpx` client. Background tasks keep the OpenAI HTTP/2 connection warm (45 s), trim the glibc heap (300 s), and emit the per-minute `rerank_performance` journal record.
- **Host coupling.** The batch scripts and the `/web/.env` fallbacks assume the production host layout (`/web/...`, `/web/fastapi_app`, `/opt/xfai-embedding/cache`, MySQL socket, ES on localhost). This is a single-host operational toolset, not a portable package; paths above are reproduced exactly as written in the code.

## Development and testing

Tests live beside the code as `test_*.py` files and use `pytest` with offline fakes/mocks (no network, no real OpenAI key, no live model):

| File | Covers |
|---|---|
| [`test_batch_tokenization.py`](test_batch_tokenization.py) | `/embed_batch` tokenization: untruncated texts are encoded once; Unicode truncation retains the exact decoded payload and batch token counts |
| [`test_pair_score_cache.py`](test_pair_score_cache.py) | [`pair_score_cache.py`](pair_score_cache.py): exact ordered-pair identity, LRU bounds, model-identity separation, cancellation-safe coalescing, invalid outputs never cached |
| [`test_rerank_metrics.py`](test_rerank_metrics.py) | [`rerank_metrics.py`](rerank_metrics.py): overlapping inference counts process CPU once; failed predictions release the active window |
| [`test_rerank_performance_contract.py`](test_rerank_performance_contract.py) | `/rerank` contract: pair-cache IDs/order/reverse-direction/`top_k`, document mode, no default truncation in `pairs` mode, `422` on mixed modes |

Run them from the repository root:

```bash
python3 -m pytest test_batch_tokenization.py test_pair_score_cache.py test_rerank_metrics.py test_rerank_performance_contract.py
```

`pytest` is not listed in [`requirements.txt`](requirements.txt); it must be available in your environment. Importing `server.py` in tests also requires the server dependencies above (including `httpx`).

There is no CI configuration, linter configuration, or build step in this repository. [`CLAUDE.md`](CLAUDE.md) contains additional operational notes for contributors working with an AI assistant on this codebase, including the supported fast-reindex workflow.

## Repository layout

| Path | Purpose |
|---|---|
| [`server.py`](server.py) | FastAPI service: `/health`, `/embed`, `/embed_batch`, `/rerank` |
| [`generate_all_embeddings.py`](generate_all_embeddings.py) | Primary 3-phase MySQL + ES embedding sync |
| [`bulk_vectorize.py`](bulk_vectorize.py) | Parallel ES-only vectorization through the running service |
| [`fill_missing.py`](fill_missing.py) | Backfill missing ES vectors via the OpenAI API directly |
| [`full_reindex.py`](full_reindex.py) | Destructive full rebuild of the `wf_wf` index (requires `--yes`) |
| [`fast_reindex.py`](fast_reindex.py) | Retired guard; prints the supported command and exits `64` |
| [`pair_score_cache.py`](pair_score_cache.py) | Bounded ordered-pair score cache with async coalescing (used by `server.py`) |
| [`rerank_metrics.py`](rerank_metrics.py) | Inference metrics for rerank performance logging (used by `server.py`) |
| [`es_vector_utils.py`](es_vector_utils.py) | L2 normalization and bulk-rejection helpers for ES writes (used by the batch scripts) |
| [`requirements.txt`](requirements.txt) | Python dependencies for the service (see dependency notes above) |
| [`xfai-embedding.service`](xfai-embedding.service) | systemd unit for production |
| [`CLAUDE.md`](CLAUDE.md) | Contributor/operator notes, including CPU-tuning history |
| `cache` | Symlink to `/opt/xfai-embedding/cache` (Hugging Face model cache; git-ignored) |
| `test_*.py` | `pytest` suite (see [Development and testing](#development-and-testing)) |

## License

This repository contains no `LICENSE` file. Until a license is added by the copyright holder, no permission to copy, modify, or redistribute this code is granted by this repository; all rights are reserved by default. This README does not add or imply any license terms.
