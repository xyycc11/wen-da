"""Independent local MiniLM embedding; no Chroma client or paid API calls.

Uses the same ONNX artifact and preprocessing as the old default Chroma model.
Model download is an explicit setup step, never a request-time fallback.
"""
import os
import math
import hashlib
import threading
from pathlib import Path


def validate_vectors(vectors, count, dimension):
    if len(vectors) != count:
        raise ValueError("embedding_count_mismatch")
    result = []
    for vector in vectors:
        if len(vector) != dimension:
            raise ValueError("embedding_dimension_mismatch")
        values = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in values) or sum(v*v for v in values) == 0:
            raise ValueError("invalid_embedding")
        result.append(values)
    return result


class MiniLMEmbedder:
    dimension = 384
    signature_base = "onnx-minilm-l6-v2:384:token256:cosine:v1"

    def __init__(self, model_dir=None):
        default = Path.home() / ".cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx"
        self.model_dir = Path(model_dir or os.getenv("VECTOR_MODEL_DIR") or default)
        self._lock = threading.RLock()
        self._session = self._tokenizer = None
        self._signature = None

    def _asset_hash(self):
        digest = hashlib.sha256()
        for name in ("model.onnx", "tokenizer.json"):
            path = self.model_dir / name
            if not path.is_file():
                raise RuntimeError("Local embedding model missing; run python -m tooling.prepare_vector_model")
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        return digest.hexdigest()

    @property
    def signature(self):
        if self._signature is None:
            self._signature = self.signature_base + ":" + self._asset_hash()
        return self._signature

    def __call__(self, texts):
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("embedding_text_must_not_be_empty")
        with self._lock:
            import numpy as np
            if self._session is None:
                expected_signature = self.signature
                if expected_signature != self.signature_base + ":" + self._asset_hash():
                    raise RuntimeError("embedding_assets_changed_before_load")
                if not all((self.model_dir / name).is_file() for name in ("model.onnx", "tokenizer.json")):
                    raise RuntimeError("Local embedding model missing; run python -m tooling.prepare_vector_model")
                import onnxruntime as ort
                from tokenizers import Tokenizer
                tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
                tokenizer.enable_truncation(max_length=256)
                tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
                session = ort.InferenceSession(str(self.model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
                self._tokenizer, self._session = tokenizer, session
            vectors = []
            for start in range(0, len(texts), 32):
                encoded = self._tokenizer.encode_batch(texts[start:start + 32])
                ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
                mask = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
                hidden = self._session.run(None, {"input_ids": ids, "attention_mask": mask,
                                                  "token_type_ids": np.zeros_like(ids)})[0]
                weights = mask[:, :, None]
                pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
                norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                if (norms == 0).any():
                    raise ValueError("zero_embedding")
                vectors.extend((pooled / norms).astype(np.float32).tolist())
            return validate_vectors(vectors, len(texts), self.dimension)
