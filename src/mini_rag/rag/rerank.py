"""Cross-encoder reranking with bge-reranker-v2-m3 through onnxruntime (no PyTorch)."""

from typing import Protocol

import numpy as np
from tokenizers import Tokenizer

# Pinned revision of the community ONNX export of BAAI/bge-reranker-v2-m3 (Apache-2.0).
RERANKER = {
    "repo": "onnx-community/bge-reranker-v2-m3-ONNX",
    "revision": "6f5ff65298512715a1e669753bc754d2bc8f367b",
    "model": "onnx/model_int8.onnx",
    "tokenizer": "tokenizer.json",
}

_TOKEN_INPUTS = ("input_ids", "attention_mask", "token_type_ids")


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text; higher is more relevant."""
        ...


class OnnxReranker:
    """Scores (query, text) pairs with an onnxruntime session that outputs logits.

    Each pair is cut to `max_length` tokens, longest side first, so a long query
    cannot fail the call. Pairs run in batches of `batch_size`.
    """

    def __init__(
        self, session, tokenizer: Tokenizer, max_length: int = 512, batch_size: int = 8
    ):
        pad_id = tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise ValueError("tokenizer has no <pad> token")
        tokenizer.enable_truncation(max_length, strategy="longest_first")
        tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")
        self.session = session
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self._input_names = [i.name for i in session.get_inputs()]
        unsupported = [n for n in self._input_names if n not in _TOKEN_INPUTS]
        if unsupported:
            raise ValueError(f"session needs unsupported inputs: {unsupported}")

    def score(self, query: str, texts: list[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encodings = self.tokenizer.encode_batch([(query, t) for t in batch])
            ids = np.array([e.ids for e in encodings], dtype=np.int64)
            available = {
                "input_ids": ids,
                "attention_mask": np.array(
                    [e.attention_mask for e in encodings], dtype=np.int64
                ),
                "token_type_ids": np.zeros_like(ids),
            }
            feed = {name: available[name] for name in self._input_names}
            (logits,) = self.session.run(None, feed)
            scores.extend(
                float(x) for x in np.asarray(logits).reshape(len(batch), -1)[:, 0]
            )
        return scores


def load_reranker() -> OnnxReranker:
    """Download (once, then cached) the pinned reranker and open a CPU session."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    def fetch(name: str) -> str:
        return hf_hub_download(
            RERANKER["repo"], RERANKER[name], revision=RERANKER["revision"]
        )

    session = ort.InferenceSession(fetch("model"), providers=["CPUExecutionProvider"])
    return OnnxReranker(session, Tokenizer.from_file(fetch("tokenizer")))
