"""Explicit, read-only source migration. Defaults to a non-writing dry run.

Apply requires a paused source, a fresh namespace, and writes a local backup
before the first target mutation. No collection/data deletion or app cutover.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from core.vector_store import ROLES, build_collections


def read_source(collection, batch_size=200):
    rows = []
    count = collection.count()
    for offset in range(0, count, batch_size):
        batch = collection.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])
        for key, document, metadata in zip(batch["ids"], batch["documents"], batch["metadatas"]):
            if not isinstance(document, str) or not document.strip() or not isinstance(metadata, dict):
                raise ValueError("invalid_source_row; inspect old data before migrating")
            rows.append({"id": key, "document": document, "metadata": metadata})
    if len(rows) != count or len({row["id"] for row in rows}) != count:
        raise ValueError("source_changed_or_duplicate_ids")
    return sorted(rows, key=lambda row: row["id"])


def row_hash(rows):
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def migrate_collection(rows, target, batch_size=32):
    # Must be a fresh target or an identical resumable partial migration.
    source_by_id = {row["id"]: row for row in rows}
    existing = target.get()
    for key, document, metadata in zip(existing["ids"], existing["documents"], existing["metadatas"]):
        source = source_by_id.get(key)
        if source is None or source["document"] != document or source["metadata"] != metadata:
            raise ValueError("target_conflict; select a fresh migration prefix")
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        if batch:
            target.upsert(ids=[row["id"] for row in batch], documents=[row["document"] for row in batch],
                          metadatas=[row["metadata"] for row in batch])
    actual = target.get()
    result = sorted([{"id": key, "document": doc, "metadata": meta}
                     for key, doc, meta in zip(actual["ids"], actual["documents"], actual["metadatas"])],
                    key=lambda row: row["id"])
    if row_hash(result) != row_hash(rows):
        raise ValueError("target_verification_failed")
    return {"count": len(rows), "sha256": row_hash(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-kind", choices=["http", "local"], required=True)
    parser.add_argument("--source-host", default="localhost")
    parser.add_argument("--source-port", type=int, default=8001)
    parser.add_argument("--source-path")
    parser.add_argument("--target-prefix", required=True)
    parser.add_argument("--backup-dir", default="data/migration_backups")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--ack-source-paused", action="store_true")
    args = parser.parse_args()
    if args.apply and not args.ack_source_paused:
        parser.error("--apply requires --ack-source-paused; pause writers before export/cutover")
    import chromadb
    settings = chromadb.Settings(anonymized_telemetry=False)
    if args.source_kind == "local":
        if not args.source_path or not (Path(args.source_path) / "chroma.sqlite3").is_file():
            parser.error("local source requires existing chroma.sqlite3 under --source-path")
        source = chromadb.PersistentClient(path=args.source_path, settings=settings)
    else:
        source = chromadb.HttpClient(host=args.source_host, port=args.source_port, settings=settings)
        source.heartbeat()  # No automatic fallback to a different source.
    collections = {name: source.get_collection(name) for name in sorted(ROLES)}
    snapshot = {name: read_source(collection) for name, collection in collections.items()}
    summary = {name: {"count": len(rows), "sha256": row_hash(rows)} for name, rows in snapshot.items()}
    if not args.apply:
        print(json.dumps({"mode": "dry-run", "source": summary}, indent=2))
        return
    backup = Path(args.backup_dir) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex)
    backup.mkdir(parents=True, exist_ok=False)
    # Private text is kept locally, not printed or uploaded. Never overwrite backups.
    with (backup / "source.json").open("x", encoding="utf-8") as handle:
        json.dump(snapshot, handle, ensure_ascii=True, allow_nan=False)
    client, targets = build_collections(sorted(ROLES), backend="milvus", prefix=args.target_prefix)
    try:
        result = {name: migrate_collection(rows, targets[name]) for name, rows in snapshot.items()}
        for name, collection in collections.items():
            if row_hash(read_source(collection)) != summary[name]["sha256"]:
                raise ValueError("source_changed; target is staging only, do not cut over")
        manifest = {"mode": "verified-staging", "target_prefix": args.target_prefix, "source": summary,
                    "target": result, "metric": "COSINE", "embedding": targets["episodic"].embedder.signature,
                    "cutover": False}
        with (backup / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        print(json.dumps(manifest, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
