#!/usr/bin/env python3
"""
Fill missing embeddings using OpenAI text-embedding-3-small.
Streams through ES docs without vectors and updates them.
"""

import os
import time
import httpx
import sys
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan, streaming_bulk

from es_vector_utils import l2_normalize, count_bulk_rejections

ES_HOST = "http://localhost:9200"
ES_INDEX = "wf_wf"
OPENAI_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # OpenAI can handle up to 2048, but smaller is safer
MAX_TEXT_LENGTH = 8000  # text-embedding-3-small limit

# Get OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    with open("/web/.env") as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY="):
                OPENAI_API_KEY = line.split("=", 1)[1].strip()
                break


def strip_tags(html):
    """Remove HTML tags."""
    import re
    if not html:
        return ""
    text = re.sub(r'<[^>]+>', '', html)
    for old, new in [('&nbsp;', ' '), ('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'), ('&quot;', '"'), ('&#39;', "'")]:
        text = text.replace(old, new)
    return text


def get_openai_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings from OpenAI API."""
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENAI_MODEL,
                "input": texts
            }
        )

        if response.status_code != 200:
            print(f"OpenAI error: {response.status_code} - {response.text[:200]}")
            return []

        data = response.json()
        if "/web/fastapi_app" not in sys.path:
            sys.path.insert(0, "/web/fastapi_app")
        from direct_metering import meter_embedding_sync
        meter_embedding_sync(data, model=OPENAI_MODEL, request_source="xfai_embed_fill_missing",
                             n_inputs=len(texts), provider="openai",
                             response_time_ms=int(response.elapsed.total_seconds() * 1000))
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]


def get_docs_without_vectors(es):
    """Stream docs missing content_vector."""
    query = {
        "bool": {
            "must_not": {
                "exists": {"field": "content_vector"}
            }
        }
    }

    return scan(
        es,
        index=ES_INDEX,
        query={"query": query},
        scroll="10m",
        size=500,
        _source=["title", "message"]
    )


def process_batch(batch, es):
    """Process a batch of docs: get embeddings and update ES."""
    # Extract text
    texts = []
    doc_ids = []

    for doc in batch:
        title = doc["_source"].get("title", "") or ""
        message = strip_tags(doc["_source"].get("message", "") or "")
        text = f"{title} {message}"[:MAX_TEXT_LENGTH].strip()

        if text:
            texts.append(text)
            doc_ids.append(doc["_id"])

    if not texts:
        return 0

    # Get embeddings
    embeddings = get_openai_embeddings(texts)
    if not embeddings or len(embeddings) != len(texts):
        print(f"  Warning: Got {len(embeddings) if embeddings else 0} embeddings for {len(texts)} texts")
        return 0

    # Prepare bulk updates
    actions = []
    for doc_id, emb in zip(doc_ids, embeddings):
        actions.append({
            "_op_type": "update",
            "_index": ES_INDEX,
            "_id": doc_id,
            # dot_product mapping rejects non-unit vectors (#822).
            "doc": {"content_vector": l2_normalize(emb)}
        })

    # Execute bulk update
    success = 0
    rejected = 0
    for ok, result in streaming_bulk(es, actions, raise_on_error=False):
        if ok:
            success += 1
        elif rejected < 3:
            rejected += count_bulk_rejections(ok, result, "fill_missing")
        else:
            rejected += 1
    if rejected:
        print(f"fill_missing: {rejected} of {len(actions)} docs rejected by ES", flush=True)

    return success


def main():
    print(f"Fill Missing Embeddings")
    print(f"Model: {OPENAI_MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    if not OPENAI_API_KEY:
        print("ERROR: OpenAI API key not found")
        return

    es = Elasticsearch([ES_HOST])

    # Count missing
    count_query = {"query": {"bool": {"must_not": {"exists": {"field": "content_vector"}}}}}
    total_missing = es.count(index=ES_INDEX, body=count_query)["count"]
    print(f"Documents missing vectors: {total_missing:,}")

    if total_missing == 0:
        print("All documents have vectors!")
        return

    print()

    start_time = time.time()
    processed = 0
    batch = []

    for doc in get_docs_without_vectors(es):
        batch.append(doc)

        if len(batch) >= BATCH_SIZE:
            success = process_batch(batch, es)
            processed += success

            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = total_missing - processed
            eta = remaining / rate / 60 if rate > 0 else 0

            print(f"\rProcessed: {processed:,}/{total_missing:,} ({100*processed/total_missing:.1f}%) | {rate:.1f}/s | ETA: {eta:.1f}m", end="", flush=True)

            batch = []

            # Small delay to respect rate limits
            time.sleep(0.1)

    # Process remaining
    if batch:
        success = process_batch(batch, es)
        processed += success

    elapsed = time.time() - start_time
    print(f"\n\nComplete!")
    print(f"  Processed: {processed:,} documents")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  Rate: {processed/elapsed:.1f} docs/sec")


if __name__ == "__main__":
    main()
