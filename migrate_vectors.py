#!/usr/bin/env python3
"""
Migrate BLOB embeddings to native MariaDB VECTOR type.

Converts binary float BLOBs to VECTOR(1536) and removes the BLOB to save space.
Run with: python3 /opt/xfai-embedding/migrate_vectors.py
"""

import struct
import pymysql
import time
import sys

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'GqGKjT5vEoIzod/OaB7bgQ==',
    'database': 'wf_wf',
    'autocommit': True,
    'charset': 'utf8mb4'
}

BATCH_SIZE = 2000


def migrate():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Count remaining BLOBs to convert
    cursor.execute("""
        SELECT COUNT(*) FROM openai_embeddings
        WHERE embedding IS NOT NULL AND LENGTH(embedding) = 6144
    """)
    remaining = cursor.fetchone()[0]

    if remaining == 0:
        print("All embeddings already converted!")
        return

    print(f"{remaining:,} embeddings to convert (BLOB -> VECTOR)")

    migrated = 0
    errors = 0
    start_time = time.time()

    while True:
        cursor.execute("""
            SELECT id, embedding FROM openai_embeddings
            WHERE embedding IS NOT NULL AND LENGTH(embedding) = 6144
            ORDER BY id LIMIT %s
        """, (BATCH_SIZE,))

        rows = cursor.fetchall()
        if not rows:
            break

        for row_id, blob in rows:
            try:
                # Unpack 1536 floats from binary blob
                floats = struct.unpack('1536f', blob)
                vector_json = '[' + ','.join(str(f) for f in floats) + ']'

                # Convert to VECTOR and clear BLOB in one update
                cursor.execute("""
                    UPDATE openai_embeddings
                    SET embedding_vec = VEC_FromText(%s), embedding = NULL
                    WHERE id = %s
                """, (vector_json, row_id))
                migrated += 1

            except Exception as e:
                errors += 1
                print(f"\nError on id {row_id}: {e}", file=sys.stderr)

        elapsed = time.time() - start_time
        rate = migrated / elapsed if elapsed > 0 else 0
        pct = (migrated / remaining * 100) if remaining > 0 else 100
        print(f"\rConverted: {migrated:,}/{remaining:,} ({pct:.1f}%) - {rate:.0f}/sec", end='', flush=True)

    elapsed = time.time() - start_time
    print(f"\n\nComplete!")
    print(f"  Converted: {migrated:,}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.1f}s ({migrated/elapsed:.0f}/sec)")

    cursor.close()
    conn.close()


if __name__ == '__main__':
    migrate()
