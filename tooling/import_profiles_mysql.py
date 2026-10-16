"""Explicit, non-destructive import of exported vector profiles into MySQL."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from memory.profile_store import MySQLProfileStore, ProfileConflict, identity, validate_profile


def prepare_profiles(payload):
    rows = payload.get("user_profile") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("expected_exported_profile_rows")
    latest = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
            raise ValueError("invalid_export_row")
        meta = row["metadata"]
        user = identity(meta.get("user_id"))
        conv = identity(meta.get("conv_id") or "legacy-import", "conv_id")
        data = validate_profile(json.loads(row["document"]))
        raw = meta.get("updated_at") or meta.get("ts")
        timestamp = datetime.fromisoformat(raw.replace("Z", "+00:00")) if raw else datetime.min
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timestamp = timestamp.astimezone(timezone.utc)
        previous = latest.get(user)
        if previous and timestamp == previous[0] and data != previous[1]:
            raise ValueError("ambiguous_profile_timestamp")
        if previous is None or timestamp > previous[0]:
            latest[user] = (timestamp, data, conv)
    return [(user, values[1], values[2]) for user, values in sorted(latest.items())]


def import_profiles(rows, store):
    # Preflight every target before starting any write; races are still guarded
    # by each insert's primary key. This is not one cross-user transaction.
    pending = []
    for user, data, conv in rows:
        record = store.load(user)
        if record is not None and record.data != data:
            raise ProfileConflict("existing_mysql_profile_differs")
        if record is None:
            pending.append((user, data, conv))
    for user, data, conv in pending:
        store.save(user, data, conv, expected_version=0)
    for user, data, conv in rows:
        record = store.load(user)
        if record is None or record.data != data:
            raise ProfileConflict("profile_import_verification_failed")
    return {"verified": len(rows), "inserted": len(pending)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--ack-source-paused", action="store_true")
    args = parser.parse_args()
    if args.apply and not args.ack_source_paused:
        parser.error("--apply requires --ack-source-paused")
    source = Path(args.input)
    if source.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("export_too_large_split_before_import")
    rows = prepare_profiles(json.loads(source.read_text(encoding="utf-8")))
    if not args.apply:
        print(json.dumps({"mode": "dry-run", "users": len(rows)}))
        return
    load_dotenv()
    store = MySQLProfileStore()
    try:
        store.check_ready()
        print(json.dumps({"mode": "verified", **import_profiles(rows, store), "cutover": False}))
    finally:
        store.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Profile import failed: {type(exc).__name__}; source unchanged, inspect staging before retry") from None
