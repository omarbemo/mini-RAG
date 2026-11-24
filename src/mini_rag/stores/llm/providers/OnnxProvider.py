"""English sentence embeddings with bge-small-en-v1.5 through onnxruntime (no PyTorch)."""

import logging

import numpy as np
from tokenizers import Tokenizer

from mini_rag.stores.llm.LLMinterface import LLMinterface

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


def _load_embedder() -> OnnxEmbedder:
    """Download (once, then cached) the pinned embedder and open a CPU session."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    def fetch(name: str) -> str:
        return hf_hub_download(
            EMBEDDER["repo"], EMBEDDER[name], revision=EMBEDDER["revision"]
        )

    session = ort.InferenceSession(fetch("model"), providers=["CPUExecutionProvider"])
    return OnnxEmbedder(session, Tokenizer.from_file(fetch("tokenizer")))


class OnnxProvider(LLMinterface):
    """Local English embeddings: bge-small-en-v1.5 through onnxruntime, no API key.

    Embedding only; use OPENAI or COHERE as GENERATION_BACKEND. The model is pinned
    at the top of this file, so EMBEDDING_MODEL_ID is ignored.
    """

    def __init__(self):
        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = DIM
        self.embedder = None
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        if embedding_size and embedding_size != DIM:
            self.logger.warning(
                f"EMBEDDING_MODEL_SIZE={embedding_size} ignored: ONNX embedder outputs {DIM}"
            )
        self.embedding_model_id = EMBEDDER["repo"]
        self.embedder = _load_embedder()  # first run downloads ~130 MB, then cached

    def generate_text(self, prompt: str, chat_history: list=None, max_output_tokens: int=None,
                            temperature: float = None):
        self.logger.error("ONNX provider cannot generate text; set GENERATION_BACKEND to OPENAI or COHERE")
        return None

    def embed_text(self, text: str, document_type: str = None):
        if self.embedder is None:
            self.logger.error("Embedding model for ONNX was not set")
            return None
        return self.embedder.embed([text])[0].tolist()

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": prompt}
