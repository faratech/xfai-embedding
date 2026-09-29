#!/usr/bin/env python3
"""
Fast parallel vectorization of Elasticsearch documents.
Bypasses XenForo job system for maximum throughput.

Usage:
    python3 bulk_vectorize.py [--workers=8] [--batch=50] [--index=wf_wf]
"""

import asyncio
import aiohttp
import argparse
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan

from es_vector_utils import l2_normalize

# Configuration
ES_HOST = "http://localhost:9200"
EMBED_URL = "http://127.0.0.1:8765/embed_batch"
DEFAULT_WORKERS = 8
DEFAULT_BATCH_SIZE = 50
MAX_TEXT_LENGTH = 2000


def get_es_client():
    return Elasticsearch([ES_HOST])


def get_documents_without_vectors(es, index_name, batch_size=1000):
    """Scroll through documents missing content_vector field."""
    query = {
        "bool": {
            "must_not": {
                "exists": {"field": "content_vector"}
            }
        }
    }

    return scan(
        es,
        index=index_name,
        query={"query": query},
        scroll="5m",
        size=batch_size,
        _source=["title", "message"]
    )


def strip_tags(html):
    """Simple HTML tag stripper matching PHP's strip_tags()."""
    import re
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', html or '')
    # Decode common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    return text


def extract_text(doc):
    """
    Extract text content from document for embedding.
    Matches exactly: $title . ' ' . substr(strip_tags($message), 0, 2000)
    """
    source = doc.get("_source", {})
    title = source.get("title", "") or ""
    message = source.get("message", "") or ""

    # Match PHP: strip_tags on message, then substr to 2000
    stripped_message = strip_tags(message)[:MAX_TEXT_LENGTH]

    # Combine: title + space + message (matching PHP concatenation)
    text = f"{title} {stripped_message}"

    if not text.strip():
        return None

    return text


async def embed_batch_async(session, texts):
    """Send batch embedding request."""
    try:
        async with session.post(
            EMBED_URL,
            json={"texts": texts, "type": "document"},
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("embeddings", [])
            else:
                print(f"Embed error: {resp.status}", file=sys.stderr)
                return None
    except Exception as e:
        print(f"Embed request failed: {e}", file=sys.stderr)
        return None


def prepare_bulk_updates(docs_with_vectors, index_name):
    """Prepare bulk update actions for Elasticsearch."""
    actions = []
    for doc_id, vector in docs_with_vectors:
        actions.append({
            "_op_type": "update",
            "_index": index_name,
            "_id": doc_id,
            # dot_product mapping rejects non-unit vectors (#822).
            "doc": {"content_vector": l2_normalize(vector)}
        })
    return actions


async def process_batch(session, batch, index_name, es):
    """Process a batch of documents: embed and update ES."""
    # Extract texts
    texts = []
    valid_docs = []

    for doc in batch:
        text = extract_text(doc)
        if text:
            texts.append(text)
            valid_docs.append(doc)

    if not texts:
        return 0

    # Get embeddings
    embeddings = await embed_batch_async(session, texts)
    if not embeddings or len(embeddings) != len(valid_docs):
        return 0

    # Prepare and execute bulk update
    docs_with_vectors = [
        (doc["_id"], emb)
        for doc, emb in zip(valid_docs, embeddings)
    ]

    actions = prepare_bulk_updates(docs_with_vectors, index_name)

    try:
        success, errors = bulk(es, actions, raise_on_error=False)
        if errors:
            print(f"Bulk errors: {len(errors)}", file=sys.stderr)
        return success
    except Exception as e:
        print(f"Bulk update failed: {e}", file=sys.stderr)
        return 0


async def worker(worker_id, queue, session, index_name, es, stats):
    """Worker that processes batches from queue."""
    while True:
        batch = await queue.get()
        if batch is None:
            queue.task_done()
            break

        count = await process_batch(session, batch, index_name, es)
        stats["processed"] += count
        stats["batches"] += 1
        queue.task_done()


async def main_async(index_name, num_workers, batch_size):
    """Main async loop."""
    es = get_es_client()

    # Get total count
    total_query = {"query": {"bool": {"must_not": {"exists": {"field": "content_vector"}}}}}
    total = es.count(index=index_name, body=total_query)["count"]
    print(f"Documents without vectors: {total:,}")

    if total == 0:
        print("All documents already have vectors!")
        return

    stats = {"processed": 0, "batches": 0}
    queue = asyncio.Queue(maxsize=num_workers * 2)

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        # Start workers
        workers = [
            asyncio.create_task(worker(i, queue, session, index_name, es, stats))
            for i in range(num_workers)
        ]

        # Feed batches to queue
        batch = []
        doc_count = 0

        for doc in get_documents_without_vectors(es, index_name, batch_size=500):
            batch.append(doc)
            doc_count += 1

            if len(batch) >= batch_size:
                await queue.put(batch)
                batch = []

                # Progress update every 10 batches
                if stats["batches"] % 10 == 0 and stats["batches"] > 0:
                    elapsed = time.time() - start_time
                    rate = stats["processed"] / elapsed if elapsed > 0 else 0
                    remaining = total - stats["processed"]
                    eta = remaining / rate if rate > 0 else 0
                    print(f"\rProcessed: {stats['processed']:,}/{total:,} "
                          f"({100*stats['processed']/total:.1f}%) "
                          f"| Rate: {rate:.0f}/s "
                          f"| ETA: {eta/60:.0f}m", end="", flush=True)

        # Send remaining batch
        if batch:
            await queue.put(batch)

        # Signal workers to stop
        for _ in range(num_workers):
            await queue.put(None)

        # Wait for completion
        await asyncio.gather(*workers)

    elapsed = time.time() - start_time
    rate = stats["processed"] / elapsed if elapsed > 0 else 0

    print(f"\n\nCompleted!")
    print(f"  Processed: {stats['processed']:,} documents")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  Rate: {rate:.1f} docs/sec")


def main():
    parser = argparse.ArgumentParser(description="Bulk vectorize ES documents")
    parser.add_argument("--index", default="wf_wf", help="ES index name")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel workers")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    print(f"Bulk Vectorization")
    print(f"  Index: {args.index}")
    print(f"  Workers: {args.workers}")
    print(f"  Batch size: {args.batch}")
    print()

    asyncio.run(main_async(args.index, args.workers, args.batch))


if __name__ == "__main__":
    main()
