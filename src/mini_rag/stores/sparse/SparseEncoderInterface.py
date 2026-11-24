from abc import ABC, abstractmethod
from typing import List

from qdrant_client import models


class SparseEncoderInterface(ABC):

    @abstractmethod
    def encode_docs(self, texts: List[str]) -> List[models.SparseVector]:
        pass

    @abstractmethod
    def encode_query(self, text: str) -> models.SparseVector:
        pass
