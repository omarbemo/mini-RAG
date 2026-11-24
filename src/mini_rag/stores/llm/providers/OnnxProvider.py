from mini_rag.stores.llm.LLMinterface import LLMinterface
from mini_rag.rag.embed import DIM, EMBEDDER, load_embedder
import logging


class OnnxProvider(LLMinterface):
    """Local English embeddings: bge-small-en-v1.5 through onnxruntime, no API key.

    Embedding only; use OPENAI or COHERE as GENERATION_BACKEND. The model is pinned
    in mini_rag/rag/embed.py, so EMBEDDING_MODEL_ID is ignored.
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
        self.embedder = load_embedder()  # first run downloads ~130 MB, then cached

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
