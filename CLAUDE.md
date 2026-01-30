# XFAI Embedding Service

FastAPI microservice providing AI embedding and reranking capabilities for the XFAI XenForo add-on.

## Overview

This service provides:
- **Text embeddings** via OpenAI `text-embedding-3-small` (1536 dimensions)
- **Cross-encoder reranking** via local `ms-marco-MiniLM-L-12-v2` model

## Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check, returns model status |
| `/embed` | POST | Single text embedding |
| `/embed_batch` | POST | Batch embeddings (up to 500 texts) |
| `/rerank` | POST | Cross-encoder reranking |

## Running

```bash
# Development
python3 server.py

# Production (via systemd)
sudo systemctl start xfai-embedding
sudo systemctl status xfai-embedding
journalctl -u xfai-embedding -f  # View logs
```

## Configuration

Environment variables:
- `XFAI_EMBED_HOST` - Bind address (default: 127.0.0.1)
- `XFAI_EMBED_PORT` - Port (default: 8765)
- `OPENAI_API_KEY` - OpenAI API key (or reads from `/web/.env`)
- `HF_HOME` / `TRANSFORMERS_CACHE` - Model cache directory

## File Structure

```
server.py              # Main FastAPI application
requirements.txt       # Python dependencies
xfai-embedding.service # Systemd service file (copy to /etc/systemd/system/)
cache/                 # Symlink to /opt/xfai-embedding/cache (model files)

# Helper scripts for batch operations:
bulk_vectorize.py      # Vectorize ES documents missing embeddings
fast_reindex.py        # Reindex using cached MySQL embeddings
full_reindex.py        # Complete reindex from scratch
fill_missing.py        # Backfill missing embeddings
migrate_vectors.py     # Convert BLOB to MariaDB VECTOR type
```

## Integration

Called by `XFAI\Service\EmbeddingClient` in the XenForo add-on at `/web/public_html/src/addons/XFAI/`.

## Model Cache

The `cache/` directory is symlinked to `/opt/xfai-embedding/cache` to keep large model files outside the git repository. Models are downloaded automatically on first use from Hugging Face Hub.
