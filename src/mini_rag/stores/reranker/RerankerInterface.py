from abc import ABC, abstractmethod
from typing import List


class RerankerInterface(ABC):

    @abstractmethod
    def score(self, query: str, texts: List[str]) -> List[float]:
        """Return one relevance score per text; higher is more relevant."""
        pass
