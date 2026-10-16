"""A small Chroma-shaped boundary for the calls already used by EchoMind.

Milvus mode fails closed; it NEVER silently falls back to Chroma/local storage.
The legacy mode is explicit and used during controlled migration only.
"""
import json
import os
import re
import threading

from core.vector_embedding import MiniLMEmbedder, validate_vectors


ROLES = {"knowledge_base", "episodic", "user_profile"}
FILTER_KEYS = {"user_id", "conv_id", "title", "document_id", "version", "chunk_index"}


def metadata_filter(where):
    if where is None:
        return ""
    if not isinstance(where, dict) or not where:
        raise ValueError("invalid_metadata_filter")
    if "$and" in where:
        conditions = where["$and"]
        if len(where) != 1 or not isinstance(conditions, list) or not conditions:
            raise ValueError("invalid_and_filter")
        return " and ".join(f"({metadata_filter(item)})" for item in conditions)
    parts = []
    for key, value in where.items():
        if key not in FILTER_KEYS or type(value) not in (str, int, bool):
            raise ValueError("unsupported_metadata_filter")
        parts.append(f'metadata[{json.dumps(key)}] == {json.dumps(value, ensure_ascii=True)}')
    return " and ".join(parts)


def _bounded_text(value, maximum, name):
    if not isinstance(value, str) or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"invalid_{name}")
    return value


class MilvusCollection:
    def __init__(self, client, name, role, embedder, *, initialize=True):
        if role not in ROLES or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,200}", name):
            raise ValueError("invalid_collection")
        self.client, self.name, self.role, self.embedder = client, name, role, embedder
        self._lock = threading.RLock()
        space = "scalar-profile-unit-placeholder:v1" if role == "user_profile" else embedder.signature
        self.description = json.dumps({"schema": 1, "role": role, "embedding": space}, sort_keys=True)
        if initialize:
            self._initialize()

    def _initialize(self):
        from pymilvus import DataType, MilvusClient
        if not self.client.has_collection(self.name, timeout=10):
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False, description=self.description)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=512)
            schema.add_field("document", DataType.VARCHAR, max_length=65535)
            schema.add_field("metadata", DataType.JSON)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.embedder.dimension)
            indexes = MilvusClient.prepare_index_params()
            indexes.add_index(field_name="embedding", index_name="embedding_cosine", index_type="FLAT", metric_type="COSINE")
            self.client.create_collection(self.name, schema=schema, index_params=indexes,
                                          consistency_level="Strong", timeout=30)
        description = self.client.describe_collection(self.name, timeout=10)
        if description.get("description") != self.description:
            raise ValueError("collection_embedding_or_schema_mismatch; use a new migration prefix")
        fields = {field["name"]: field for field in description.get("fields", [])}
        if set(fields) != {"id", "document", "metadata", "embedding"} or int(fields["embedding"]["params"]["dim"]) != self.embedder.dimension:
            raise ValueError("collection_schema_mismatch")
        index = self.client.describe_index(self.name, "embedding_cosine", timeout=10)
        if index.get("metric_type") != "COSINE":
            raise ValueError("collection_metric_mismatch")
        self.client.load_collection(self.name, timeout=30)

    def upsert(self, ids, documents, metadatas, embeddings=None):
        if not ids or len(set(ids)) != len(ids) or not len(ids) == len(documents) == len(metadatas):
            raise ValueError("invalid_write_batch")
        rows = []
        for key, doc, meta in zip(ids, documents, metadatas):
            _bounded_text(key, 512, "id")
            _bounded_text(doc, 65535, "document")
            if not isinstance(meta, dict) or len(json.dumps(meta, ensure_ascii=False, allow_nan=False).encode()) > 60000:
                raise ValueError("invalid_metadata")
            if self.role != "knowledge_base" and (not isinstance(meta.get("user_id"), str) or not meta["user_id"]):
                raise ValueError("memory_owner_required")
            if self.role == "episodic" and (not isinstance(meta.get("conv_id"), str) or not meta["conv_id"]):
                raise ValueError("conversation_required")
            rows.append({"id": key, "document": doc, "metadata": dict(meta)})
        # Profiles are scalar-only: the required vector field is an unused unit placeholder.
        vectors = ([[1.0] + [0.0] * (self.embedder.dimension - 1) for _ in ids]
                   if self.role == "user_profile" else self.embedder(documents) if embeddings is None else embeddings)
        vectors = validate_vectors(vectors, len(ids), self.embedder.dimension)
        for row, vector in zip(rows, vectors):
            row["embedding"] = vector
        with self._lock:
            return self.client.upsert(self.name, data=rows, timeout=30)

    def add(self, **kwargs):
        # Current callers use unique/deterministic IDs; upsert makes an interrupted
        # compression retry idempotent instead of adding duplicate observations.
        return self.upsert(**kwargs)

    def get(self, ids=None, where=None, include=None):
        expression = metadata_filter(where)
        if ids is not None:
            if any(not isinstance(key, str) for key in ids):
                raise ValueError("invalid_ids")
            if not ids:
                return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
            condition = "id in " + json.dumps(ids, ensure_ascii=True)
            expression = f"({expression}) and ({condition})" if expression else condition
        iterator = self.client.query_iterator(self.name, batch_size=200, filter=expression,
                                              output_fields=["id", "document", "metadata"],
                                              consistency_level="Strong", timeout=30)
        rows = []
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                rows.extend(batch)
        finally:
            iterator.close()
        return {"ids": [row["id"] for row in rows], "documents": [row["document"] for row in rows],
                "metadatas": [row["metadata"] for row in rows]}

    def query(self, query_texts, n_results, where=None):
        if self.role == "user_profile":
            raise ValueError("profiles_support_scalar_reads_only")
        if type(n_results) is not int or not 1 <= n_results <= 16384:
            raise ValueError("invalid_top_k")
        expression = metadata_filter(where)
        if self.role == "episodic" and not _has_user_filter(where):
            raise ValueError("memory_query_owner_required")
        vectors = validate_vectors(self.embedder(query_texts), len(query_texts), self.embedder.dimension)
        results = self.client.search(self.name, data=vectors, anns_field="embedding", filter=expression,
                                     limit=n_results, search_params={"metric_type": "COSINE", "params": {}},
                                     output_fields=["document", "metadata"], consistency_level="Strong", timeout=30)
        return {"ids": [[hit["id"] for hit in hits] for hits in results],
                "documents": [[hit["entity"]["document"] for hit in hits] for hits in results],
                "metadatas": [[hit["entity"]["metadata"] for hit in hits] for hits in results],
                # Milvus COSINE is a similarity; the facade expects a distance.
                "distances": [[1.0 - float(hit["distance"]) for hit in hits] for hits in results]}

    def delete(self, ids):
        if not ids:
            return
        if any(not isinstance(key, str) for key in ids):
            raise ValueError("invalid_ids")
        return self.client.delete(self.name, ids=ids, timeout=30)

    def count(self):
        rows = self.client.query(self.name, filter="", output_fields=["count(*)"],
                                 consistency_level="Strong", timeout=30)
        return int(rows[0]["count(*)"])


def _has_user_filter(where):
    if not isinstance(where, dict):
        return False
    if "user_id" in where:
        return isinstance(where["user_id"], str) and bool(where["user_id"])
    return any(_has_user_filter(item) for item in where.get("$and", []))


def build_collections(names, *, backend=None, chroma_host="localhost", chroma_port=8000,
                      chroma_path="./data/chroma", client=None, embedder=None, prefix=None):
    backend = (backend or os.getenv("VECTOR_BACKEND", "chroma")).strip().lower()
    if not names or backend not in {"chroma", "milvus"} or not set(names) <= ROLES:
        raise ValueError("invalid_vector_backend_or_roles")
    if backend == "chroma":
        import chromadb
        if client is None:
            try:
                client = chromadb.HttpClient(host=chroma_host, port=chroma_port,
                                            settings=chromadb.Settings(anonymized_telemetry=False))
                client.heartbeat()
            except Exception:
                client = chromadb.PersistentClient(path=chroma_path, settings=chromadb.Settings(anonymized_telemetry=False))
        return client, {name: client.get_or_create_collection(name) for name in names}
    prefix = prefix or os.getenv("MILVUS_COLLECTION_PREFIX", "echomind_v1")
    embedder = embedder or MiniLMEmbedder()
    if client is None:
        from pymilvus import MilvusClient
        client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
                              token=os.getenv("MILVUS_TOKEN", ""), timeout=10)
    try:
        return client, {name: MilvusCollection(client, f"{prefix}_{name}", name, embedder) for name in names}
    except Exception:
        client.close()
        raise
