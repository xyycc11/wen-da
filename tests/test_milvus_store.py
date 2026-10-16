import ast
import asyncio
import json
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from core.vector_store import MilvusCollection, build_collections, metadata_filter
from core.vector_embedding import MiniLMEmbedder, validate_vectors
from memory.conversation_memory import MemoryManager
from tooling.knowledge_base import KnowledgeBase
from tooling.migrate_chroma_to_milvus import migrate_collection, read_source, row_hash


class Embedder:
    dimension = 3
    signature = "test-only-three-dimensional"

    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [[1., 0., 0.] if "退款" in text else [0., 1., 0.] for text in texts]


def matches(expression, row):
    if not expression:
        return True
    def value(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.Name):
            return {"id": row["id"], "metadata": row["metadata"], "true": True, "false": False}[node.id]
        if isinstance(node, ast.Subscript): return value(node.value).get(value(node.slice))
        if isinstance(node, ast.List): return [value(item) for item in node.elts]
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And): return all(value(item) for item in node.values)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = value(node.left), value(node.comparators[0])
            if isinstance(node.ops[0], ast.Eq): return left == right
            if isinstance(node.ops[0], ast.In): return left in right
        raise AssertionError("Unexpected filter syntax")
    return value(ast.parse(expression, mode="eval").body)


class Iterator:
    def __init__(self, rows, client):
        self.rows, self.client = list(rows), client

    def next(self):
        batch, self.rows = self.rows[:2], self.rows[2:]
        return batch

    def close(self):
        self.client.iterator_closed += 1


class Client:
    def __init__(self):
        self.rows = {}
        self.iterator_closed = 0
        self.closed = False
        self.last_search = None

    def upsert(self, name, data, **kwargs):
        self.rows.setdefault(name, {}).update({row["id"]: row for row in data})

    def query_iterator(self, name, filter, **kwargs):
        return Iterator([row for row in self.rows.get(name, {}).values() if matches(filter, row)], self)

    def query(self, name, **kwargs):
        return [{"count(*)": len(self.rows.get(name, {}))}]

    def search(self, name, data, filter, limit, **kwargs):
        self.last_search = {"filter": filter, "limit": limit, **kwargs}
        answers = []
        for query in data:
            hits = []
            for row in self.rows.get(name, {}).values():
                if not matches(filter, row): continue
                vector = row["embedding"]
                score = sum(a*b for a, b in zip(query, vector)) / math.sqrt(sum(x*x for x in query) * sum(x*x for x in vector))
                hits.append({"id": row["id"], "distance": score, "entity": row})
            answers.append(sorted(hits, key=lambda hit: hit["distance"], reverse=True)[:limit])
        return answers

    def delete(self, name, ids, **kwargs):
        for key in ids:
            self.rows.get(name, {}).pop(key, None)

    def close(self): self.closed = True


def collection(role="knowledge_base", client=None, embedder=None):
    return MilvusCollection(client or Client(), "test_" + role, role, embedder or Embedder(), initialize=False)


def test_scoring_is_cosine_not_squared_l2():
    store = collection()
    store.upsert(["a", "b"], ["退款", "物流"], [{}, {}])
    result = store.query(["退款"], 2)
    assert result["documents"] == [["退款", "物流"]]
    assert result["distances"] == [[0., 1.]]
    assert store.client.last_search["search_params"]["metric_type"] == "COSINE"
    assert store.client.last_search["consistency_level"] == "Strong"


def test_metadata_and_unicode_roundtrip_with_paginated_get():
    store = collection()
    store.upsert([str(i) for i in range(7)], ["资料"]*7, [{"title": "中文", "chunk_index": i} for i in range(7)])
    result = store.get(where={"title": "中文"})
    assert len(result["ids"]) == store.count() == 7
    assert result["metadatas"][6]["chunk_index"] == 6
    assert store.client.iterator_closed == 1


def test_same_user_conversation_and_cross_user_isolation():
    store = collection("episodic")
    store.upsert(["same", "other-conv", "other-user"], ["退款"]*3,
                 [{"user_id": "u1", "conv_id": "c1"}, {"user_id": "u1", "conv_id": "c2"},
                  {"user_id": "u2", "conv_id": "c1"}])
    current = store.query(["退款"], 5, where={"$and": [{"user_id": "u1"}, {"conv_id": "c1"}]})
    fallback = store.query(["退款"], 5, where={"user_id": "u1"})
    assert current["ids"] == [["same"]]
    assert set(fallback["ids"][0]) == {"same", "other-conv"}


def test_quoted_user_id_cannot_inject_filter():
    malicious = 'u1" or metadata["user_id"] == "u2'
    store = collection("episodic")
    store.upsert(["a", "b"], ["退款"]*2, [{"user_id": malicious, "conv_id": "c"}, {"user_id": "u2", "conv_id": "c"}])
    assert store.query(["退款"], 5, where={"user_id": malicious})["ids"] == [["a"]]


@pytest.mark.parametrize("where", [{}, {"$or": [{"user_id": "u1"}]}, {"user_id": {"$ne": "u1"}},
                                  {"arbitrary_field": "x"}, {"$and": []}, {"$and": [{}]}])
def test_unsupported_filters_fail_closed(where):
    with pytest.raises(ValueError): metadata_filter(where)


@pytest.mark.parametrize("where", [None, {"conv_id": "c1"}, {"user_id": ""}])
def test_memory_query_requires_owner(where):
    with pytest.raises(ValueError): collection("episodic").query(["退款"], 5, where=where)


def test_profile_is_scalar_and_never_embeds():
    embedder = Embedder()
    store = collection("user_profile", embedder=embedder)
    store.upsert(["p"], ['{"style":"brief"}'], [{"user_id": "u1"}])
    assert store.get(ids=["p"])["documents"] == ['{"style":"brief"}']
    assert store.get(where={"user_id": "u2"})["ids"] == []
    assert embedder.calls == []
    with pytest.raises(ValueError): store.query(["question"], 1)


@pytest.mark.parametrize("vector", [[1.], [float('nan'), 0, 0], [0., 0., 0.], [float('inf'), 0, 0]])
def test_invalid_vectors_rejected_before_write(vector):
    store = collection()
    with pytest.raises(ValueError): store.upsert(["a"], ["退款"], [{}], embeddings=[vector])
    assert store.count() == 0


@pytest.mark.parametrize("role,metadata", [("episodic", {"conv_id": "c"}), ("episodic", {"user_id": "u"}),
                                         ("user_profile", {})])
def test_memory_write_requires_owner(role, metadata):
    with pytest.raises(ValueError): collection(role).upsert(["a"], ["x"], [metadata])


def test_invalid_utf8_byte_length_and_empty_batches():
    store = collection()
    for ids, docs, metas in [([], [], []), (["a", "a"], ["x", "y"], [{}, {}]),
                             (["长"*200], ["x"], [{}]), (["a"], ["长"*30000], [{}])]:
        with pytest.raises(ValueError): store.upsert(ids, docs, metas)


@pytest.mark.parametrize("value", [0, -1, True, 16385, "5"])
def test_invalid_top_k(value):
    with pytest.raises(ValueError): collection().query(["x"], value)


def test_knowledge_import_update_and_search_preserve_public_interface():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    import threading
    kb._write_lock = threading.RLock()
    kb._collection = collection()
    kb.add_documents([{"document_id": "a", "title": "退款", "content": "退款"*600},
                      {"document_id": "b", "title": "退款", "content": "物流"}])
    kb.add_documents([{"document_id": "a", "title": "退款", "content": "退款新规则"}])
    assert kb.doc_count == 2
    result = asyncio.run(kb.search_async("退款", 5))
    assert result[0]["content"] == "退款新规则"
    assert set(result[0]) == {"title", "content", "score", "chunk"}
    assert result[0]["score"] == 1.0


def test_idempotent_migration_and_conflicting_target_refusal():
    store = collection()
    rows = [{"id": "a", "document": "退款", "metadata": {"title": "政策"}}]
    assert migrate_collection(rows, store)["count"] == 1
    assert migrate_collection(rows, store)["sha256"] == row_hash(rows)
    assert store.count() == 1
    with pytest.raises(ValueError, match="target_conflict"):
        migrate_collection([{"id": "a", "document": "变了", "metadata": {"title": "政策"}}], store)
    assert store.get()["documents"] == ["退款"]


def test_migration_paginates_source_and_rejects_changed_count():
    class Source:
        def count(self): return 5
        def get(self, limit, offset, include):
            keys = list(range(5))[offset:offset + limit]
            return {"ids": [str(key) for key in keys], "documents": ["退款"]*len(keys), "metadatas": [{}]*len(keys)}
    assert len(read_source(Source(), batch_size=2)) == 5
    source = Source()
    source.count = lambda: 6
    with pytest.raises(ValueError): read_source(source, batch_size=2)


def test_assets_missing_fail_without_download(tmp_path):
    with pytest.raises(RuntimeError, match="model missing"):
        _ = MiniLMEmbedder(tmp_path).signature


def test_asset_fingerprint_changes_with_weights(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"fake-model")
    (tmp_path / "tokenizer.json").write_bytes(b"fake-tokenizer")
    first = MiniLMEmbedder(tmp_path).signature
    (tmp_path / "model.onnx").write_bytes(b"different-fake-model")
    assert first != MiniLMEmbedder(tmp_path).signature


def test_scalar_profile_does_not_require_model_assets(tmp_path):
    store = collection("user_profile", embedder=MiniLMEmbedder(tmp_path))
    store.upsert(["p"], ['{}'], [{"user_id": "u"}])
    assert store.count() == 1


def test_unknown_backend_fails_before_client_use():
    with pytest.raises(ValueError): build_collections(["knowledge_base"], backend="typo")


def test_profile_direct_id_checks_owner_metadata():
    manager = MemoryManager.__new__(MemoryManager)
    manager._profile = SimpleNamespace(get=lambda **kwargs: (
        {"documents": ['{"secret":"other-user"}'], "metadatas": [{"user_id": "other"}]}
        if "ids" in kwargs else {"documents": [], "metadatas": []}))
    assert asyncio.run(manager._get_profile("u1")) == {}


def test_storage_failure_keeps_redis_history_and_summary():
    manager = MemoryManager.__new__(MemoryManager)
    snapshot = [json.dumps({"role": "user", "content": str(i), "ts": "2026-10-14T10:00:00"}) for i in range(20)]
    manager._redis = SimpleNamespace(lrange=AsyncMock(return_value=snapshot), get=AsyncMock(return_value="old-summary"), pipeline=Mock())
    manager._client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(
        content=[{"type": "text", "text": "summary"}], stop_reason="end_turn"))))
    manager._model = "fake"
    manager._merge_summary = AsyncMock(return_value="new-summary")
    manager._store_episodic = AsyncMock(return_value=False)
    asyncio.run(manager._compress("u", "c"))
    manager._redis.pipeline.assert_not_called()


def test_episode_write_failure_reports_false_and_repeat_uses_same_id():
    manager = MemoryManager.__new__(MemoryManager)
    manager._episodic = collection("episodic")
    assert asyncio.run(manager._store_episodic("u", "c", "text", "summary")) is True
    assert asyncio.run(manager._store_episodic("u", "c", "text", "summary")) is True
    assert manager._episodic.count() == 1
    manager._episodic = SimpleNamespace(add=Mock(side_effect=RuntimeError("offline storage failure")))
    assert asyncio.run(manager._store_episodic("u", "c", "text", "summary")) is False


@pytest.mark.parametrize("mismatch", ["description", "dimension", "metric"])
def test_existing_collection_mismatch_fails_without_recreating(mismatch):
    embedder = Embedder()
    expected = collection(embedder=embedder).description
    client = SimpleNamespace(
        has_collection=Mock(return_value=True),
        describe_collection=Mock(return_value={
            "description": "wrong" if mismatch == "description" else expected,
            "fields": [{"name": "id"}, {"name": "document"}, {"name": "metadata"},
                       {"name": "embedding", "params": {"dim": 4 if mismatch == "dimension" else 3}}]}),
        describe_index=Mock(return_value={"metric_type": "L2" if mismatch == "metric" else "COSINE"}),
        load_collection=Mock(), create_collection=Mock())
    with pytest.raises(ValueError, match="mismatch"):
        MilvusCollection(client, "test_knowledge_base", "knowledge_base", embedder)
    client.create_collection.assert_not_called()
    client.load_collection.assert_not_called()


def test_milvus_connection_failure_does_not_fall_back(monkeypatch):
    import pymilvus
    constructor = Mock(side_effect=RuntimeError("offline connection failure"))
    monkeypatch.setattr(pymilvus, "MilvusClient", constructor)
    with pytest.raises(RuntimeError, match="offline connection failure"):
        build_collections(["knowledge_base"], backend="milvus", embedder=Embedder())
    constructor.assert_called_once()
