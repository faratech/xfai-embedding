"""L2-normalize embeddings before any Elasticsearch write.

The live wf_wf index maps content_vector as dot_product, and ES rejects
non-unit vectors at index time (#822) — raw text-embedding-3-small output
sits at ~1.0004. The PHP pipeline (XFAI EmbeddingGenerator::normalizeVector)
normalizes before every write; this is the Python twin for the batch tools,
which wrote raw and let streaming_bulk(raise_on_error=False) swallow the
400s (generate_all_progress.json: es_backfilled 1 of 455, failed 0).

MySQL stays the raw store — normalize only at the ES boundary, like PHP.
"""
import math


def l2_normalize(vec):
    """Return vec normalized to unit L2 length (never raises on zero)."""
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if not norm:
        return list(vec)
    return [float(x) / norm for x in vec]


def count_bulk_rejections(ok, info, what):
    """Log the first streaming_bulk rejection detail; return 1 to count it."""
    err = None
    if isinstance(info, dict):
        err = (info.get("update") or {}).get("error") or info.get("error")
    print(f"  {what}: ES rejected a doc: {err}", flush=True)
    return 1
