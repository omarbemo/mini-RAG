from .SparseEncoderEnums import SparseEncoderEnums
from .providers import Bm25Provider


class SparseEncoderProviderFactory:
    def __init__(self, config):
        self.config = config

    def create(self, provider: str):
        if provider == SparseEncoderEnums.BM25.value:
            return Bm25Provider()

        return None
