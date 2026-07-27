#!/usr/bin/env python3
"""Retired guard for the former destructive Elasticsearch reindexer."""

import argparse
import sys


CANONICAL_REINDEXER = "/web/public_html/src/addons/XFES/fast_reindex.py"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "RETIRED: this legacy script is intentionally inert. "
            f"Use {CANONICAL_REINDEXER} under the shared reindex lock."
        )
    )
    parser.parse_args(argv)
    parser.print_usage(sys.stderr)
    print(
        "ERROR: this legacy reindexer is retired and will not access "
        "MySQL or Elasticsearch.\n"
        f"Use: flock -w 60 /tmp/cron_locks/fast_reindex.lock "
        f"{CANONICAL_REINDEXER} [arguments]",
        file=sys.stderr,
    )
    return 64


if __name__ == "__main__":
    sys.exit(main())
