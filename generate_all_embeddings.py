#!/usr/bin/env python3
"""
Combined embedding generator: one OpenAI API call, writes to both MySQL and Elasticsearch.

Finds posts/threads missing embeddings in either store, generates vectors once,
and writes them to both MySQL (openai_embeddings table) and ES (wf_wf content_vector).

Also backfills ES-only gaps by copying existing MySQL embeddings without API calls.

Usage:
    python3 generate_all_embeddings.py --yes          # Run with auto-confirm
    python3 generate_all_embeddings.py --dry-run      # Count only
    python3 generate_all_embeddings.py --reset --yes  # Clear progress and run
"""

import asyncio
import argparse
import os
import re
import sys
import time
import json
import signal
import random
import struct
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import tiktoken
import openai
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk, scan

sys.path.insert(0, '/web/fastapi_app')
from dotenv import load_dotenv
load_dotenv('/web/.env')
load_dotenv('/web/.config')

from db_pool import get_async_db_cursor
from embedding_hash_utils import compute_content_hash
from es_vector_utils import l2_normalize, count_bulk_rejections

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100          # Items per processing cycle
API_BATCH_SIZE = 50       # Texts per OpenAI API call
MAX_CONCURRENT = 10       # Concurrent API calls
MAX_TOKENS = 8191         # text-embedding-3-small hard limit is 8192
ES_HOST = "http://localhost:9200"
ES_INDEX = "wf_wf"
PROGRESS_FILE = "/web/services/xfai-embedding/generate_all_progress.json"

openai.api_key = os.getenv("OPENAI_API_KEY")

# cl100k_base is the tokenizer for text-embedding-3-small
_enc = tiktoken.get_encoding("cl100k_base")


def strip_tags(html: str) -> str:
    """Remove HTML tags and decode entities."""
    if not html:
        return ""
    text = re.sub(r'<[^>]+>', '', html)
    for old, new in [('&nbsp;', ' '), ('&amp;', '&'), ('&lt;', '<'),
                     ('&gt;', '>'), ('&quot;', '"'), ('&#39;', "'")]:
        text = text.replace(old, new)
    return text


def truncate_to_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Truncate text to exactly max_tokens using the cl100k_base tokenizer."""
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _enc.decode(tokens[:max_tokens])


@dataclass
class Progress:
    processed_posts: int = 0
    processed_threads: int = 0
    es_backfilled: int = 0
    mysql_backfilled: int = 0
    failed: int = 0
    total_tokens: int = 0
    start_time: float = 0

    def save(self):
        try:
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(asdict(self), f)
        except Exception:
            pass

    @classmethod
    def load(cls):
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE) as f:
                    return cls(**json.load(f))
            except Exception:
                pass
        return cls(start_time=time.time())


class RateLimiter:
    """Adaptive rate limiter with exponential backoff on 429s."""

    def __init__(self):
        self.backoff_until = 0
        self._lock = asyncio.Lock()
        self.consecutive_failures = 0

    async def wait_if_needed(self):
        async with self._lock:
            now = time.time()
            if now < self.backoff_until:
                await asyncio.sleep(self.backoff_until - now)

    async def record_success(self):
        async with self._lock:
            self.consecutive_failures = 0

    async def record_rate_limit(self, retry_after: Optional[float] = None):
        async with self._lock:
            self.consecutive_failures += 1
            wait = retry_after or min(2 ** self.consecutive_failures, 16)
            wait += wait * 0.2 * (random.random() * 2 - 1)  # jitter
            self.backoff_until = time.time() + wait
            return wait


class CombinedEmbeddingGenerator:

    def __init__(self, auto_confirm=False, dry_run=False):
        self.auto_confirm = auto_confirm
        self.dry_run = dry_run
        self.rate_limiter = RateLimiter()
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.progress = Progress.load()
        self.shutdown = False
        self.es = Elasticsearch([ES_HOST])
        signal.signal(signal.SIGINT, self._signal)
        signal.signal(signal.SIGTERM, self._signal)

    def _signal(self, *_):
        print("\nShutdown signal received, finishing current batch...")
        self.shutdown = True

    # ── Find missing items ──────────────────────────────────────────────

    async def fetch_missing_posts(self, offset: int, limit: int) -> List[Dict]:
        """Posts missing from MySQL openai_embeddings or flagged needs_update."""
        async with get_async_db_cursor(dict_cursor=True) as cur:
            await cur.execute("""
                SELECT p.post_id, p.thread_id, p.message, t.title
                FROM xf_post p
                JOIN xf_thread t ON t.thread_id = p.thread_id
                LEFT JOIN openai_embeddings oe ON p.post_id = oe.post_id
                WHERE p.message_state = 'visible'
                AND LENGTH(p.message) > 10
                AND (oe.post_id IS NULL OR oe.needs_update = 1)
                ORDER BY p.post_id
                LIMIT %s OFFSET %s
            """, (limit, offset))
            return await cur.fetchall()

    async def fetch_missing_threads(self, offset: int, limit: int) -> List[Dict]:
        """Threads missing from MySQL openai_embeddings or flagged needs_update."""
        async with get_async_db_cursor(dict_cursor=True) as cur:
            await cur.execute("""
                SELECT t.thread_id, t.title
                FROM xf_thread t
                LEFT JOIN openai_embeddings oe ON t.thread_id = oe.thread_id AND oe.post_id IS NULL
                WHERE t.discussion_state = 'visible'
                AND LENGTH(t.title) > 5
                AND (oe.thread_id IS NULL OR oe.needs_update = 1)
                ORDER BY t.thread_id
                LIMIT %s OFFSET %s
            """, (limit, offset))
            return await cur.fetchall()

    async def count_missing(self) -> Tuple[int, int]:
        async with get_async_db_cursor(dict_cursor=True) as cur:
            await cur.execute("""
                SELECT COUNT(*) as c FROM xf_post p
                LEFT JOIN openai_embeddings oe ON p.post_id = oe.post_id
                WHERE p.message_state = 'visible' AND LENGTH(p.message) > 10
                AND (oe.post_id IS NULL OR oe.needs_update = 1)
            """)
            posts = (await cur.fetchone())['c']

            await cur.execute("""
                SELECT COUNT(*) as c FROM xf_thread t
                LEFT JOIN openai_embeddings oe ON t.thread_id = oe.thread_id AND oe.post_id IS NULL
                WHERE t.discussion_state = 'visible' AND LENGTH(t.title) > 5
                AND (oe.thread_id IS NULL OR oe.needs_update = 1)
            """)
            threads = (await cur.fetchone())['c']
        return posts, threads

    def count_es_missing(self) -> int:
        """Count ES post/thread docs without content_vector (excludes conversations)."""
        r = self.es.count(
            index=ES_INDEX,
            body={"query": {"bool": {
                "must_not": {"exists": {"field": "content_vector"}},
                "filter": {"terms": {"type": ["thread", "post"]}}
            }}}
        )
        return r["count"]

    # ── Text preparation ────────────────────────────────────────────────

    def prepare_text(self, item: Dict, item_type: str) -> Tuple[str, bool]:
        """Consistent text for both stores: stripped HTML, truncated to token limit.
        Returns (text, was_truncated)."""
        if item_type == "post":
            title = item.get("title", "") or ""
            message = strip_tags(item.get("message", "") or "")
            text = f"{title} {message}".strip()
        else:  # thread
            text = (item.get("title", "") or "").strip()
        truncated = truncate_to_tokens(text)
        return truncated, len(truncated) < len(text)

    # ── OpenAI API call ─────────────────────────────────────────────────

    async def call_openai(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Single OpenAI API call, returns list of float arrays or None."""
        if not texts:
            return []

        for attempt in range(5):
            try:
                await self.rate_limiter.wait_if_needed()
                async with self.semaphore:
                    response = await asyncio.to_thread(
                        openai.embeddings.create,
                        input=texts,
                        model=EMBEDDING_MODEL
                    )
                await self.rate_limiter.record_success()

                sorted_data = sorted(response.data, key=lambda x: x.index)
                self.progress.total_tokens += response.usage.total_tokens
                return [item.embedding for item in sorted_data]

            except openai.RateLimitError as e:
                retry_after = None
                if hasattr(e, 'response') and e.response:
                    ra = e.response.headers.get('retry-after')
                    if ra:
                        try:
                            retry_after = float(ra)
                        except ValueError:
                            pass
                wait = await self.rate_limiter.record_rate_limit(retry_after)
                print(f"\n  Rate limit (attempt {attempt+1}/5), backing off {wait:.1f}s...")
                if attempt < 4:
                    continue
                return None

            except openai.BadRequestError as e:
                # Token limit exceeded — fall back to one-at-a-time
                if "context length" in str(e) and len(texts) > 1:
                    return await self._call_openai_individually(texts)
                print(f"\n  Bad request (skipping): {e}")
                return None

            except openai.APIError as e:
                print(f"\n  API error: {e}")
                if attempt < 4:
                    await asyncio.sleep(1)
                    continue
                return None

            except Exception as e:
                print(f"\n  Unexpected error: {e}")
                return None

        return None

    async def _call_openai_individually(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Fallback: embed texts one at a time so only oversized items fail."""
        results: List[Optional[List[float]]] = []
        for text in texts:
            try:
                await self.rate_limiter.wait_if_needed()
                async with self.semaphore:
                    response = await asyncio.to_thread(
                        openai.embeddings.create,
                        input=[text],
                        model=EMBEDDING_MODEL
                    )
                await self.rate_limiter.record_success()
                self.progress.total_tokens += response.usage.total_tokens
                results.append(response.data[0].embedding)
            except openai.BadRequestError:
                # This specific text is too long even after truncation — skip it
                results.append(None)
                self.progress.failed += 1
            except openai.RateLimitError:
                await self.rate_limiter.record_rate_limit()
                results.append(None)
            except Exception:
                results.append(None)
        return results

    # ── Write to MySQL ──────────────────────────────────────────────────

    async def write_mysql(self, items: List[Dict], embeddings: List[List[float]],
                          truncated_flags: List[bool], item_type: str) -> int:
        """Bulk insert embeddings into openai_embeddings table."""
        rows = []
        for item, emb, was_truncated in zip(items, embeddings, truncated_flags):
            if emb is None:
                continue
            blob = np.array(emb, dtype=np.float32).tobytes()
            text, _ = self.prepare_text(item, item_type)
            content_hash = compute_content_hash(text)

            if item_type == "post":
                rows.append((item['post_id'], None, blob, EMBEDDING_DIM, content_hash, was_truncated))
            else:
                rows.append((None, item['thread_id'], blob, EMBEDDING_DIM, content_hash, was_truncated))

        if not rows:
            return 0

        try:
            async with get_async_db_cursor(dict_cursor=False) as cur:
                await cur.executemany("""
                    INSERT INTO openai_embeddings
                    (post_id, thread_id, embedding, embedding_length, content_hash, is_truncated, needs_update, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, 0, NOW())
                    ON DUPLICATE KEY UPDATE
                        embedding = VALUES(embedding),
                        embedding_length = VALUES(embedding_length),
                        content_hash = VALUES(content_hash),
                        is_truncated = VALUES(is_truncated),
                        needs_update = 0,
                        last_updated = NOW()
                """, rows)
            return len(rows)
        except Exception as e:
            print(f"\n  MySQL error: {e}")
            return 0

    # ── Write to Elasticsearch ──────────────────────────────────────────

    def write_es(self, items: List[Dict], embeddings: List[List[float]],
                 item_type: str) -> int:
        """Bulk update content_vector in ES."""
        actions = []
        for item, emb in zip(items, embeddings):
            if emb is None:
                continue
            if item_type == "post":
                doc_id = f"post-{item['post_id']}"
            else:
                doc_id = f"thread-{item['thread_id']}"

            actions.append({
                "_op_type": "update",
                "_index": ES_INDEX,
                "_id": doc_id,
                # dot_product mapping rejects non-unit vectors (#822) — the
                # PHP pipeline normalizes at this same boundary.
                "doc": {"content_vector": l2_normalize(emb)}
            })

        if not actions:
            return 0

        success = 0
        rejected = 0
        for ok, info in streaming_bulk(self.es, actions, raise_on_error=False):
            if ok:
                success += 1
            elif rejected < 3:
                rejected += count_bulk_rejections(ok, info, "write_es")
            else:
                rejected += 1
        if rejected:
            print(f"  write_es: {rejected} of {len(actions)} docs rejected by ES", flush=True)
        return success

    # ── Process a batch ─────────────────────────────────────────────────

    async def process_batch(self, items: List[Dict], item_type: str) -> Tuple[int, int]:
        """Process a batch: prepare text, call OpenAI once, write to both stores."""
        mysql_written = 0
        es_written = 0

        for i in range(0, len(items), API_BATCH_SIZE):
            if self.shutdown:
                break

            chunk = items[i:i + API_BATCH_SIZE]
            prepared = [self.prepare_text(item, item_type) for item in chunk]

            # Filter empty texts, keep truncation flags
            valid = [(item, text, trunc) for item, (text, trunc) in zip(chunk, prepared) if text]
            if not valid:
                continue

            valid_items = [v[0] for v in valid]
            valid_texts = [v[1] for v in valid]
            valid_truncated = [v[2] for v in valid]

            # One API call
            embeddings = await self.call_openai(valid_texts)
            if not embeddings:
                self.progress.failed += len(valid_items)
                continue

            # Write to both stores
            mysql_count = await self.write_mysql(valid_items, embeddings, valid_truncated, item_type)
            es_count = self.write_es(valid_items, embeddings, item_type)

            mysql_written += mysql_count
            es_written += es_count

        return mysql_written, es_written

    # ── Backfill ES from existing MySQL embeddings ──────────────────────

    async def backfill_es_from_mysql(self) -> int:
        """Copy MySQL embeddings to ES where ES is missing content_vector.
        No API calls needed."""
        print("\nPhase 2: Backfilling ES from existing MySQL embeddings...")

        # Find ES docs without vectors
        query = {"bool": {"must_not": {"exists": {"field": "content_vector"}}}}
        missing_docs = []
        for doc in scan(self.es, index=ES_INDEX, query={"query": query},
                        scroll="5m", size=500, _source=False):
            missing_docs.append(doc["_id"])

        if not missing_docs:
            print("  No ES-only gaps found.")
            return 0

        print(f"  Found {len(missing_docs):,} ES docs missing vectors, checking MySQL...")

        # Separate into posts and threads
        post_ids = []
        thread_ids = []
        for doc_id in missing_docs:
            if doc_id.startswith("post-"):
                post_ids.append(int(doc_id.split("-", 1)[1]))
            elif doc_id.startswith("thread-"):
                thread_ids.append(int(doc_id.split("-", 1)[1]))

        filled = 0

        # Backfill posts in batches
        for batch_start in range(0, len(post_ids), 500):
            batch = post_ids[batch_start:batch_start + 500]
            if not batch:
                break
            placeholders = ",".join(["%s"] * len(batch))
            async with get_async_db_cursor(dict_cursor=True) as cur:
                await cur.execute(f"""
                    SELECT post_id, embedding FROM openai_embeddings
                    WHERE post_id IN ({placeholders}) AND embedding IS NOT NULL
                """, batch)
                rows = await cur.fetchall()

            if rows:
                actions = []
                for row in rows:
                    emb = list(struct.unpack(f'{EMBEDDING_DIM}f', row['embedding']))
                    actions.append({
                        "_op_type": "update",
                        "_index": ES_INDEX,
                        "_id": f"post-{row['post_id']}",
                        "doc": {"content_vector": emb}
                    })
                for ok, _ in streaming_bulk(self.es, actions, raise_on_error=False):
                    if ok:
                        filled += 1

            if self.shutdown:
                break

        # Backfill threads in batches
        for batch_start in range(0, len(thread_ids), 500):
            batch = thread_ids[batch_start:batch_start + 500]
            if not batch:
                break
            placeholders = ",".join(["%s"] * len(batch))
            async with get_async_db_cursor(dict_cursor=True) as cur:
                await cur.execute(f"""
                    SELECT thread_id, embedding FROM openai_embeddings
                    WHERE thread_id IN ({placeholders}) AND post_id IS NULL AND embedding IS NOT NULL
                """, batch)
                rows = await cur.fetchall()

            if rows:
                actions = []
                for row in rows:
                    emb = list(struct.unpack(f'{EMBEDDING_DIM}f', row['embedding']))
                    actions.append({
                        "_op_type": "update",
                        "_index": ES_INDEX,
                        "_id": f"thread-{row['thread_id']}",
                        # MySQL is the raw store; ES dot_product needs unit
                        # length (#822).
                        "doc": {"content_vector": l2_normalize(emb)}
                    })
                rejected = 0
                for ok, info in streaming_bulk(self.es, actions, raise_on_error=False):
                    if ok:
                        filled += 1
                    elif rejected < 3:
                        rejected += count_bulk_rejections(ok, info, "backfill_es")
                    else:
                        rejected += 1
                if rejected:
                    print(f"  backfill_es: {rejected} of {len(actions)} docs rejected by ES", flush=True)

            if self.shutdown:
                break

        print(f"  Backfilled {filled:,} ES docs from MySQL (no API calls).")
        self.progress.es_backfilled = filled
        return filled

    # ── Backfill MySQL from existing ES embeddings ──────────────────────

    async def backfill_mysql_from_es(self) -> int:
        """Copy ES content_vectors into MySQL where MySQL is missing the embedding.
        No API calls needed."""
        print("\nPhase 3: Backfilling MySQL from existing ES embeddings...")

        # Find posts missing from MySQL
        async with get_async_db_cursor(dict_cursor=True) as cur:
            await cur.execute("""
                SELECT p.post_id
                FROM xf_post p
                LEFT JOIN openai_embeddings oe ON p.post_id = oe.post_id
                WHERE p.message_state = 'visible' AND LENGTH(p.message) > 10
                AND oe.post_id IS NULL
            """)
            missing_posts = [r['post_id'] for r in await cur.fetchall()]

        async with get_async_db_cursor(dict_cursor=True) as cur:
            await cur.execute("""
                SELECT t.thread_id
                FROM xf_thread t
                LEFT JOIN openai_embeddings oe ON t.thread_id = oe.thread_id AND oe.post_id IS NULL
                WHERE t.discussion_state = 'visible' AND LENGTH(t.title) > 5
                AND oe.thread_id IS NULL
            """)
            missing_threads = [r['thread_id'] for r in await cur.fetchall()]

        if not missing_posts and not missing_threads:
            print("  No MySQL-only gaps found.")
            return 0

        print(f"  MySQL missing: {len(missing_posts):,} posts, {len(missing_threads):,} threads. Checking ES...")

        filled = 0

        # Backfill posts: check ES for vectors, copy to MySQL
        for batch_start in range(0, len(missing_posts), 100):
            if self.shutdown:
                break
            batch = missing_posts[batch_start:batch_start + 100]
            # Audit #837: hash the live text so hash-based drift consumers
            # agree with needs_update=0 — writing '' made every drift pass
            # light these rows up for paid regeneration.
            items_by_id = {}
            async with get_async_db_cursor(dict_cursor=True) as cur:
                placeholders = ",".join(["%s"] * len(batch))
                await cur.execute(f"""
                    SELECT p.post_id, p.message, t.title
                    FROM xf_post p JOIN xf_thread t ON t.thread_id = p.thread_id
                    WHERE p.post_id IN ({placeholders})
                """, batch)
                for r in await cur.fetchall():
                    items_by_id[int(r['post_id'])] = {'post_id': int(r['post_id']),
                                                      'message': r['message'], 'title': r['title']}
            doc_ids = [{"_index": ES_INDEX, "_id": f"post-{pid}"} for pid in batch]

            try:
                result = self.es.mget(body={"docs": doc_ids}, _source=["content_vector"])
            except Exception as e:
                print(f"\n  ES mget error: {e}")
                continue

            rows = []
            for doc in result.get("docs", []):
                if not doc.get("found") or "content_vector" not in doc.get("_source", {}):
                    continue
                emb = doc["_source"]["content_vector"]
                if len(emb) != EMBEDDING_DIM:
                    continue
                pid = int(doc["_id"].split("-", 1)[1])
                item = items_by_id.get(pid)
                if item is None:
                    continue
                text, truncated = self.prepare_text(item, "post")
                blob = np.array(emb, dtype=np.float32).tobytes()
                rows.append((pid, None, blob, EMBEDDING_DIM, compute_content_hash(text), 1 if truncated else 0))

            if rows:
                try:
                    async with get_async_db_cursor(dict_cursor=False) as cur:
                        await cur.executemany("""
                            INSERT INTO openai_embeddings
                            (post_id, thread_id, embedding, embedding_length, content_hash, is_truncated, needs_update, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, 0, NOW())
                            ON DUPLICATE KEY UPDATE
                                embedding = VALUES(embedding),
                                embedding_length = VALUES(embedding_length),
                                needs_update = 0,
                                last_updated = NOW()
                        """, rows)
                    filled += len(rows)
                except Exception as e:
                    print(f"\n  MySQL insert error: {e}")

        # Backfill threads
        for batch_start in range(0, len(missing_threads), 100):
            if self.shutdown:
                break
            batch = missing_threads[batch_start:batch_start + 100]
            items_by_id = {}
            async with get_async_db_cursor(dict_cursor=True) as cur:
                placeholders = ",".join(["%s"] * len(batch))
                await cur.execute(f"""
                    SELECT thread_id, title FROM xf_thread
                    WHERE thread_id IN ({placeholders})
                """, batch)
                for r in await cur.fetchall():
                    items_by_id[int(r['thread_id'])] = {'thread_id': int(r['thread_id']),
                                                        'title': r['title']}
            doc_ids = [{"_index": ES_INDEX, "_id": f"thread-{tid}"} for tid in batch]

            try:
                result = self.es.mget(body={"docs": doc_ids}, _source=["content_vector"])
            except Exception as e:
                print(f"\n  ES mget error: {e}")
                continue

            rows = []
            for doc in result.get("docs", []):
                if not doc.get("found") or "content_vector" not in doc.get("_source", {}):
                    continue
                emb = doc["_source"]["content_vector"]
                if len(emb) != EMBEDDING_DIM:
                    continue
                tid = int(doc["_id"].split("-", 1)[1])
                item = items_by_id.get(tid)
                if item is None:
                    continue
                text, truncated = self.prepare_text(item, "thread")
                blob = np.array(emb, dtype=np.float32).tobytes()
                rows.append((None, tid, blob, EMBEDDING_DIM, compute_content_hash(text), 1 if truncated else 0))

            if rows:
                try:
                    async with get_async_db_cursor(dict_cursor=False) as cur:
                        await cur.executemany("""
                            INSERT INTO openai_embeddings
                            (post_id, thread_id, embedding, embedding_length, content_hash, is_truncated, needs_update, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, 0, NOW())
                            ON DUPLICATE KEY UPDATE
                                embedding = VALUES(embedding),
                                embedding_length = VALUES(embedding_length),
                                needs_update = 0,
                                last_updated = NOW()
                        """, rows)
                    filled += len(rows)
                except Exception as e:
                    print(f"\n  MySQL insert error: {e}")

        print(f"  Backfilled {filled:,} MySQL rows from ES (no API calls).")
        self.progress.mysql_backfilled = filled
        return filled

    # ── Main ────────────────────────────────────────────────────────────

    async def run(self):
        print("=" * 70)
        print("COMBINED EMBEDDING GENERATOR (MySQL + Elasticsearch)")
        print(f"Model: {EMBEDDING_MODEL} | Dim: {EMBEDDING_DIM}")
        print("=" * 70)

        # Count missing
        missing_posts, missing_threads = await self.count_missing()
        es_missing = self.count_es_missing()
        total = missing_posts + missing_threads

        print(f"\nMySQL missing:  {missing_posts:,} posts, {missing_threads:,} threads ({total:,} total)")
        print(f"ES missing:     {es_missing:,} docs without content_vector")

        if total == 0 and es_missing == 0:
            print("\nAll embeddings are in sync!")
            return

        # Cost estimate
        avg_tokens = (missing_posts * 480 + missing_threads * 14)
        est_cost = (avg_tokens / 1_000_000) * 0.02
        print(f"\nEstimated API cost: ~${est_cost:.2f} ({avg_tokens:,} tokens)")

        if self.dry_run:
            print("\n--dry-run: exiting without changes.")
            return

        if not self.auto_confirm:
            input("\nPress Enter to start (Ctrl+C to cancel)...")

        print(f"\nStarted at {datetime.now().strftime('%H:%M:%S')}\n")
        self.progress.start_time = time.time()

        # Phase 1: Generate new embeddings for items missing from MySQL
        if total > 0:
            print("Phase 1: Generating embeddings for MySQL-missing items -> both stores")

            # Posts — always query offset 0 since processed items drop out of the result set
            while not self.shutdown:
                batch = await self.fetch_missing_posts(0, BATCH_SIZE)
                if not batch:
                    break

                mysql_n, es_n = await self.process_batch(batch, "post")
                self.progress.processed_posts += mysql_n

                # If nothing was written, the same batch would loop forever — bail
                if mysql_n == 0:
                    break

                self._print_progress(total)

            # Threads
            while not self.shutdown:
                batch = await self.fetch_missing_threads(0, BATCH_SIZE)
                if not batch:
                    break

                mysql_n, es_n = await self.process_batch(batch, "thread")
                self.progress.processed_threads += mysql_n

                if mysql_n == 0:
                    break

                self._print_progress(total)

        # Phase 2: Backfill ES gaps from MySQL (free, no API calls)
        if not self.shutdown:
            await self.backfill_es_from_mysql()

        # Phase 3: Backfill MySQL gaps from ES (free, no API calls)
        if not self.shutdown:
            await self.backfill_mysql_from_es()

        self.progress.save()
        self._print_summary()

    def _print_progress(self, total: int):
        done = self.progress.processed_posts + self.progress.processed_threads
        elapsed = time.time() - self.progress.start_time
        rate = done / elapsed if elapsed > 0 else 0
        pct = 100 * done / total if total > 0 else 100
        eta = (total - done) / rate / 60 if rate > 0 else 0
        cost = (self.progress.total_tokens / 1_000_000) * 0.02

        print(f"\r  {done:,}/{total:,} ({pct:.1f}%) | "
              f"{rate:.0f}/s | ETA: {eta:.1f}m | "
              f"${cost:.3f}", end="", flush=True)

    def _print_summary(self):
        elapsed = time.time() - self.progress.start_time
        cost = (self.progress.total_tokens / 1_000_000) * 0.02
        total = self.progress.processed_posts + self.progress.processed_threads

        print(f"\n\n{'=' * 70}")
        print("COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Posts embedded:           {self.progress.processed_posts:,}")
        print(f"  Threads embedded:        {self.progress.processed_threads:,}")
        print(f"  ES backfilled from MySQL:  {self.progress.es_backfilled:,}  (free)")
        print(f"  MySQL backfilled from ES:  {self.progress.mysql_backfilled:,}  (free)")
        print(f"  Failed:                    {self.progress.failed:,}")
        print(f"  Tokens:                {self.progress.total_tokens:,}")
        print(f"  Cost:                  ${cost:.3f}")
        print(f"  Time:                  {elapsed/60:.1f}m ({total/elapsed:.0f}/s)" if elapsed > 0 else "")
        print(f"{'=' * 70}\n")


async def main():
    parser = argparse.ArgumentParser(description='Combined embedding generator (MySQL + ES)')
    parser.add_argument('--yes', '-y', action='store_true', help='Auto-confirm')
    parser.add_argument('--dry-run', action='store_true', help='Count only, no changes')
    parser.add_argument('--reset', action='store_true', help='Clear progress file')
    args = parser.parse_args()

    if args.reset and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("Progress reset.\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment or /web/.env")
        sys.exit(1)

    gen = CombinedEmbeddingGenerator(auto_confirm=args.yes, dry_run=args.dry_run)
    await gen.run()

    # Close the aiomysql pool cleanly to suppress "Event loop is closed" warnings
    from db_pool import DatabasePool
    await DatabasePool.close_async_pool()


if __name__ == "__main__":
    asyncio.run(main())
