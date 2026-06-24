#!/usr/bin/env python3
"""
Full parallel reindex: MySQL -> Embeddings -> Elasticsearch
Uses fresh DB connections per batch to avoid timeout issues.
"""

import asyncio
import aiohttp
import time
import sys
import re
import pymysql
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Configuration
ES_HOST = "http://localhost:9200"
ES_INDEX = "wf_wf"
EMBED_URL = "http://127.0.0.1:8765/embed_batch"
MAX_TEXT_LENGTH = 2000
BATCH_SIZE = 100
NUM_WORKERS = 8

# MySQL config
MYSQL_CONFIG = {
    'unix_socket': '/run/mysqld/mysqld.sock',
    'user': 'root',
    'password': ([l.split("'")[-2] for l in open('/web/public_html/src/config.php') if l.strip().startswith("$config['db']['password']")] or [''])[0],
    'database': 'wf_wf',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
    'connect_timeout': 10,
    'read_timeout': 60
}


def strip_tags(html):
    if not html:
        return ""
    text = re.sub(r'<[^>]+>', '', html)
    for old, new in [('&nbsp;', ' '), ('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'), ('&quot;', '"'), ('&#39;', "'")]:
        text = text.replace(old, new)
    return text


def get_db():
    return pymysql.connect(**MYSQL_CONFIG)


def get_es():
    return Elasticsearch([ES_HOST])


def count_all():
    """Count threads and posts."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as c FROM xf_thread WHERE discussion_state='visible'")
            threads = cur.fetchone()['c']
            cur.execute("SELECT COUNT(*) as c FROM xf_post WHERE message_state='visible'")
            posts = cur.fetchone()['c']
        return threads, posts - threads  # posts minus first posts
    finally:
        conn.close()


def fetch_thread_batch(offset, limit, retries=3):
    """Fetch a batch of threads with fresh connection and retry."""
    for attempt in range(retries):
        try:
            conn = get_db()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT t.thread_id, t.title, t.node_id, t.user_id, t.post_date, p.message
                        FROM xf_thread t
                        LEFT JOIN xf_post p ON p.post_id = t.first_post_id
                        WHERE t.discussion_state = 'visible'
                        ORDER BY t.thread_id
                        LIMIT %s OFFSET %s
                    """, (limit, offset))
                    return cur.fetchall()
            finally:
                conn.close()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            raise


def fetch_post_batch(offset, limit, retries=3):
    """Fetch a batch of posts with fresh connection and retry."""
    for attempt in range(retries):
        try:
            conn = get_db()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT p.post_id, p.thread_id, p.user_id, p.post_date, p.message, t.title, t.node_id
                        FROM xf_post p
                        JOIN xf_thread t ON t.thread_id = p.thread_id
                        WHERE p.message_state = 'visible' AND p.post_id != t.first_post_id
                        ORDER BY p.post_id
                        LIMIT %s OFFSET %s
                    """, (limit, offset))
                    return cur.fetchall()
            finally:
                conn.close()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            raise


def to_doc(row, doc_type):
    """Convert row to ES document."""
    if doc_type == 'thread':
        doc_id = f"thread-{row['thread_id']}"
        discussion_id = row['thread_id']
    else:
        doc_id = f"post-{row['post_id']}"
        discussion_id = row['thread_id']

    title = row.get('title') or ''
    message = strip_tags(row.get('message') or '')[:MAX_TEXT_LENGTH]

    return {
        '_id': doc_id,
        'type': doc_type,
        'title': title,
        'message': message,
        'date': row.get('post_date', 0),
        'user': row.get('user_id', 0),
        'node': row.get('node_id', 0),
        'discussion_id': discussion_id,
        'hidden': False,
        '_text': f"{title} {message}"
    }


async def get_embeddings(session, texts):
    """Get embeddings from service."""
    if not texts:
        return []
    try:
        async with session.post(
            EMBED_URL,
            json={"texts": texts, "type": "document"},
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("embeddings", [])
    except Exception as e:
        print(f"\nEmbed error: {e}", file=sys.stderr)
    return None


async def process_batch(session, es, docs, stats):
    """Process batch: embed and index."""
    if not docs:
        return

    texts = [d.pop('_text') for d in docs]
    embeddings = await get_embeddings(session, texts)

    if embeddings and len(embeddings) == len(docs):
        for doc, emb in zip(docs, embeddings):
            doc['content_vector'] = emb

    actions = [{'_index': ES_INDEX, '_id': d.pop('_id'), **d} for d in docs]

    try:
        success, _ = bulk(es, actions, raise_on_error=False)
        stats['indexed'] += success
    except Exception as e:
        print(f"\nBulk error: {e}", file=sys.stderr)


def create_index(es):
    """Create ES index with proper mapping."""
    try:
        es.indices.delete(index=ES_INDEX)
    except:
        pass

    mapping = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "30s"},
        "mappings": {
            "_source": {"enabled": True},
            "properties": {
                "type": {"type": "keyword"},
                "title": {"type": "text"},
                "message": {"type": "text"},
                "date": {"type": "long"},
                "user": {"type": "long"},
                "node": {"type": "long"},
                "discussion_id": {"type": "long"},
                "hidden": {"type": "boolean"},
                "content_vector": {
                    "type": "dense_vector", "dims": 384, "index": True,
                    "similarity": "cosine",
                    "index_options": {"type": "int8_hnsw", "m": 16, "ef_construction": 100}
                }
            }
        }
    }
    es.indices.create(index=ES_INDEX, body=mapping)


async def main():
    print("Full Reindex: MySQL -> Embeddings -> ES")
    print(f"Workers: {NUM_WORKERS}, Batch: {BATCH_SIZE}\n")

    # Count
    print("Counting...")
    thread_count, post_count = count_all()
    total = thread_count + post_count
    print(f"  Threads: {thread_count:,}")
    print(f"  Posts: {post_count:,}")
    print(f"  Total: {total:,}\n")

    # Create index
    es = get_es()
    print("Creating ES index...")
    create_index(es)
    print("  Done\n")

    stats = {'indexed': 0}
    start = time.time()

    async with aiohttp.ClientSession() as session:
        # Process threads
        print("Indexing threads...")
        offset = 0
        while True:
            rows = fetch_thread_batch(offset, BATCH_SIZE)
            if not rows:
                break
            docs = [to_doc(r, 'thread') for r in rows]
            await process_batch(session, es, docs, stats)
            offset += len(rows)

            elapsed = time.time() - start
            rate = stats['indexed'] / elapsed if elapsed > 0 else 0
            pct = 100 * stats['indexed'] / total
            eta = (total - stats['indexed']) / rate / 60 if rate > 0 else 0
            print(f"\r  {stats['indexed']:,}/{total:,} ({pct:.1f}%) | {rate:.0f}/s | ETA: {eta:.0f}m", end="", flush=True)

        print()

        # Process posts
        print("Indexing posts...")
        offset = 0
        while True:
            rows = fetch_post_batch(offset, BATCH_SIZE)
            if not rows:
                break
            docs = [to_doc(r, 'post') for r in rows]
            await process_batch(session, es, docs, stats)
            offset += len(rows)

            elapsed = time.time() - start
            rate = stats['indexed'] / elapsed if elapsed > 0 else 0
            pct = 100 * stats['indexed'] / total
            eta = (total - stats['indexed']) / rate / 60 if rate > 0 else 0
            print(f"\r  {stats['indexed']:,}/{total:,} ({pct:.1f}%) | {rate:.0f}/s | ETA: {eta:.0f}m", end="", flush=True)

    print("\n\nRefreshing index...")
    es.indices.refresh(index=ES_INDEX)
    es.indices.put_settings(index=ES_INDEX, body={"number_of_replicas": 1, "refresh_interval": "1s"})

    elapsed = time.time() - start
    print(f"\nCompleted!")
    print(f"  Indexed: {stats['indexed']:,}")
    print(f"  Time: {elapsed/60:.1f}m")
    print(f"  Rate: {stats['indexed']/elapsed:.0f}/s")

    # Verify
    count = es.count(index=ES_INDEX)['count']
    vecs = es.count(index=ES_INDEX, body={"query": {"exists": {"field": "content_vector"}}})['count']
    print(f"  Final: {count:,} docs, {vecs:,} with vectors")


if __name__ == "__main__":
    asyncio.run(main())
