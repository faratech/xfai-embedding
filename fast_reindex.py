#!/usr/bin/env python3
"""
Fast ES reindex using cached MySQL embeddings.
Uses streaming cursors to avoid memory issues.
"""

import pymysql
from pymysql.cursors import SSCursor
import struct
import sys
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

ES_HOST = "http://localhost:9200"
ES_INDEX = "wf_wf"
BATCH_SIZE = 200

MYSQL_CONFIG = {
    'unix_socket': '/run/mysqld/mysqld.sock',
    'user': 'root',
    'password': 'GqGKjT5vEoIzod/OaB7bgQ==',
    'database': 'wf_wf',
    'charset': 'utf8mb4',
}


def get_es():
    return Elasticsearch([ES_HOST], request_timeout=60)


def decode_embedding(blob):
    """Decode binary blob to float array (1536-dim OpenAI embeddings)."""
    if not blob:
        return None
    try:
        vec = list(struct.unpack(f'{len(blob)//4}f', blob))
        # Verify it's 1536 dimensions (OpenAI)
        if len(vec) != 1536:
            return None
        return vec
    except:
        return None


def create_index(es):
    """Create ES index with vector mapping."""
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "-1",
        },
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "message": {"type": "text"},
                "content_vector": {
                    "type": "dense_vector",
                    "dims": 1536,
                    "index": True,
                    "similarity": "dot_product"
                },
                "type": {"type": "keyword"},
                "discussion_id": {"type": "long"},
                "node_id": {"type": "long"},
                "user_id": {"type": "long"},
                "date": {"type": "date", "format": "epoch_second"},
            }
        }
    }

    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)
    es.indices.create(index=ES_INDEX, body=mapping)


def stream_threads():
    """Stream thread documents one at a time."""
    conn = pymysql.connect(**MYSQL_CONFIG, cursorclass=SSCursor)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            t.thread_id, t.title, t.node_id, t.user_id, t.post_date,
            p.message, e.embedding
        FROM xf_thread t
        LEFT JOIN xf_post p ON p.post_id = t.first_post_id
        LEFT JOIN openai_embeddings e ON e.thread_id = t.thread_id AND e.post_id IS NULL
        WHERE t.discussion_state = 'visible'
    """)

    for row in cursor:
        thread_id, title, node_id, user_id, post_date, message, embedding = row
        vec = decode_embedding(embedding)
        doc = {
            '_index': ES_INDEX,
            '_id': f"thread-{thread_id}",
            'title': title or '',
            'message': message or '',
            'type': 'thread',
            'discussion_id': thread_id,
            'node_id': node_id,
            'user_id': user_id,
            'date': post_date,
        }
        if vec and len(vec) == 1536:
            doc['content_vector'] = vec
        yield doc

    cursor.close()
    conn.close()


def stream_posts():
    """Stream post documents one at a time."""
    conn = pymysql.connect(**MYSQL_CONFIG, cursorclass=SSCursor)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            p.post_id, p.thread_id, t.title, t.node_id, p.user_id,
            p.post_date, p.message, e.embedding
        FROM xf_post p
        JOIN xf_thread t ON t.thread_id = p.thread_id
        LEFT JOIN openai_embeddings e ON e.post_id = p.post_id
        WHERE p.message_state = 'visible' AND t.discussion_state = 'visible'
    """)

    for row in cursor:
        post_id, thread_id, title, node_id, user_id, post_date, message, embedding = row
        vec = decode_embedding(embedding)
        doc = {
            '_index': ES_INDEX,
            '_id': f"post-{post_id}",
            'title': title or '',
            'message': message or '',
            'type': 'post',
            'discussion_id': thread_id,
            'node_id': node_id,
            'user_id': user_id,
            'date': post_date,
        }
        if vec and len(vec) == 1536:
            doc['content_vector'] = vec
        yield doc

    cursor.close()
    conn.close()


def main():
    print("Fast ES Reindex (streaming, low memory)")
    print("=" * 50)

    es = get_es()

    # Get counts
    conn = pymysql.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM xf_thread WHERE discussion_state='visible'")
    thread_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM xf_post p JOIN xf_thread t ON t.thread_id=p.thread_id WHERE p.message_state='visible' AND t.discussion_state='visible'")
    post_count = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    print(f"Threads: {thread_count:,}")
    print(f"Posts: {post_count:,}")
    print()

    print("Creating index...")
    create_index(es)

    # Index threads
    print("Indexing threads...")
    start = time.time()
    count = 0
    for ok, result in streaming_bulk(es, stream_threads(), chunk_size=BATCH_SIZE, raise_on_error=False):
        count += 1
        if count % 5000 == 0:
            rate = count / (time.time() - start)
            print(f"  {count:,}/{thread_count:,} | {rate:.0f}/s")
    print(f"  Done: {count:,} threads")

    # Index posts
    print("Indexing posts...")
    post_start = time.time()
    count = 0
    for ok, result in streaming_bulk(es, stream_posts(), chunk_size=BATCH_SIZE, raise_on_error=False):
        count += 1
        if count % 20000 == 0:
            rate = count / (time.time() - post_start)
            eta = (post_count - count) / rate / 60 if rate > 0 else 0
            print(f"  {count:,}/{post_count:,} | {rate:.0f}/s | ETA: {eta:.1f}m")
    print(f"  Done: {count:,} posts")

    # Finalize
    print("\nFinalizing...")
    es.indices.put_settings(index=ES_INDEX, body={"refresh_interval": "1s"})
    es.indices.refresh(index=ES_INDEX)

    total = time.time() - start
    print(f"\nComplete in {total/60:.1f} minutes ({(thread_count+post_count)/total:.0f}/s)")


if __name__ == "__main__":
    main()
