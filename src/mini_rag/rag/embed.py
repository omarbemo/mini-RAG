"""English sentence embeddings with bge-small-en-v1.5 through onnxruntime (no PyTorch)."""

from typing import Protocol

import numpy as np
from tokenizers import Tokenizer

# Pinned revision of BAAI/bge-small-en-v1.5 (MIT), which ships its own ONNX export.
EMBEDDER = {
    "repo": "BAAI/bge-small-en-v1.5",
    "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "model": "onnx/model.onnx",
    "tokenizer": "tokenizer.json",
}

# Width of the vectors; the `chunks.embedding` column is declared with it.
DIM = 384

_TOKEN_INPUTS = ("input_ids", "attention_mask", "token_type_ids")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Return one unit-length float32 vector per text."""
        ...


class OnnxEmbedder:
    """CLS-pooled, L2-normalised vectors from a session that outputs hidden states.

    Each text is cut to `max_length` tokens. Texts run in batches of `batch_size`.
    """

    def __init__(
        self, session, tokenizer: Tokenizer, max_length: int = 512, batch_size: int = 32
    ):
        pad_id = tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            raise ValueError("tokenizer has no [PAD] token")
        tokenizer.enable_truncation(max_length)
        tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
        self.session = session
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self._input_names = [i.name for i in session.get_inputs()]
        unsupported = [n for n in self._input_names if n not in _TOKEN_INPUTS]
        if unsupported:
            raise ValueError(f"session needs unsupported inputs: {unsupported}")

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            encodings = self.tokenizer.encode_batch(texts[start : start + self.batch_size])
            ids = np.array([e.ids for e in encodings], dtype=np.int64)
            available = {
                "input_ids": ids,
                "attention_mask": np.array(
                    [e.attention_mask for e in encodings], dtype=np.int64
                ),
                "token_type_ids": np.zeros_like(ids),
            }
            feed = {name: available[name] for name in self._input_names}
            hidden = np.asarray(self.session.run(None, feed)[0], dtype=np.float32)
            cls = hidden[:, 0, :]
            cls /= np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12)
            vectors.extend(cls)
        return vectors


def load_embedder() -> OnnxEmbedder:
    """Download (once, then cached) the pinned embedder and open a CPU session."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    def fetch(name: str) -> str:
        return hf_hub_download(
            EMBEDDER["repo"], EMBEDDER[name], revision=EMBEDDER["revision"]
        )

    session = ort.InferenceSession(fetch("model"), providers=["CPUExecutionProvider"])
    return OnnxEmbedder(session, Tokenizer.from_file(fetch("tokenizer")))
