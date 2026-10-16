"""Structured user profiles in MySQL; no vectors and no model calls.

Each operation owns a short transaction. Version checks reject stale updates;
the SQL transaction is never held open while waiting for an LLM.
"""
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, JSON, MetaData, String, Table, create_engine, insert, select, update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError


class ProfileConflict(RuntimeError):
    """Another writer committed first; do not blindly retry an old LLM result."""


def identity(value, name="user_id"):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"invalid_{name}")
    value.encode("utf-8")
    return value


def validate_profile(value):
    if not isinstance(value, dict) or set(value) - {"preferences", "entities"}:
        raise ValueError("invalid_profile_fields")
    preferences, entities = value.get("preferences", []), value.get("entities", {})
    def strings(items):
        return (isinstance(items, list) and len(items) <= 32 and
                all(isinstance(item, str) and item.strip() and len(item) <= 256 for item in items))
    if not strings(preferences) or not isinstance(entities, dict) or len(entities) > 32:
        raise ValueError("invalid_profile_shape")
    if any(not isinstance(key, str) or not key.strip() or len(key) > 64 or not strings(items)
           for key, items in entities.items()):
        raise ValueError("invalid_profile_entities")
    clean = {"preferences": list(dict.fromkeys(preferences)),
             "entities": {key: list(dict.fromkeys(items)) for key, items in entities.items()}}
    encoded = json.dumps(clean, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > 32768:
        raise ValueError("profile_too_large")
    return json.loads(encoded)


@dataclass(frozen=True)
class ProfileRecord:
    user_id: str
    data: dict
    version: int
    source_conv_id: str


class MySQLProfileStore:
    def __init__(self, url=None, *, engine=None):
        if engine is None:
            raw = url or os.getenv("MYSQL_PROFILE_URL", "")
            if not raw:
                raise ValueError("MYSQL_PROFILE_URL_required_for_mysql_profiles")
            try:
                parsed = make_url(raw)
            except Exception:
                raise ValueError("invalid_MYSQL_PROFILE_URL") from None
            if parsed.drivername != "mysql+pymysql" or not parsed.database:
                raise ValueError("MYSQL_PROFILE_URL_must_use_mysql_pymysql_and_database")
            # Build the engine lazily; never print a URL containing credentials.
            engine = create_engine(parsed, pool_pre_ping=True, pool_recycle=1800,
                                   connect_args={"connect_timeout": 5, "read_timeout": 10, "write_timeout": 10})
        self.engine = engine
        self.metadata = MetaData()
        self.table = Table("echomind_user_profiles", self.metadata,
            Column("user_key", String(64), primary_key=True),
            Column("user_id", String(512), nullable=False),
            Column("profile", JSON, nullable=False),
            Column("version", Integer, nullable=False),
            Column("source_conv_id", String(512), nullable=False),
            Column("updated_at", DateTime, nullable=False),
            mysql_engine="InnoDB", mysql_charset="utf8mb4")

    @staticmethod
    def key(user_id):
        return hashlib.sha256(identity(user_id).encode("utf-8")).hexdigest()

    def create_schema(self):
        """Explicit operator step, not run during normal requests."""
        self.metadata.create_all(self.engine)

    def check_ready(self):
        with self.engine.connect() as connection:
            connection.execute(select(self.table).limit(0))

    def load(self, user_id):
        key = self.key(user_id)
        with self.engine.connect() as connection:
            row = connection.execute(select(self.table).where(self.table.c.user_key == key)).mappings().first()
        if row is None:
            return None
        if row["user_id"] != user_id:
            raise ValueError("profile_owner_mismatch")
        return ProfileRecord(user_id, validate_profile(row["profile"]), row["version"], row["source_conv_id"])

    def save(self, user_id, profile, source_conv_id, *, expected_version):
        key = self.key(user_id)
        identity(source_conv_id, "conv_id")
        if type(expected_version) is not int or expected_version < 0:
            raise ValueError("invalid_expected_version")
        data = validate_profile(profile)
        values = dict(profile=data, version=expected_version + 1, source_conv_id=source_conv_id,
                      updated_at=datetime.now(timezone.utc).replace(tzinfo=None))
        try:
            with self.engine.begin() as connection:
                if expected_version == 0:
                    connection.execute(insert(self.table).values(user_key=key, user_id=user_id, **values))
                else:
                    result = connection.execute(update(self.table).where(
                        self.table.c.user_key == key, self.table.c.user_id == user_id,
                        self.table.c.version == expected_version).values(**values))
                    if result.rowcount != 1:
                        raise ProfileConflict("profile_version_conflict")
        except IntegrityError:
            raise ProfileConflict("profile_insert_conflict") from None
        return expected_version + 1

    def close(self):
        self.engine.dispose()
