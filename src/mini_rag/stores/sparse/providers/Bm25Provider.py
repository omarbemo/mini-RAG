"""English BM25 sparse vectors for Qdrant keyword search.

Documents carry the BM25 term-frequency part; Qdrant adds IDF itself (the sparse
vector is created with `Modifier.IDF`), so scores stay right as the corpus grows.
"""

import re
import zlib
from collections import Counter

import snowballstemmer
from qdrant_client import models

from ..SparseEncoderInterface import SparseEncoderInterface

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Common English function words; they would match nearly every chunk.
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers herself him
    himself his how i if in into is it its itself just me more most my myself no nor
    not now of off on once only or other our ours ourselves out over own same she
    should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where
    which while who whom why will with would you your yours yourself yourselves
    """.split()
)


class Bm25Provider(SparseEncoderInterface):
    """Lowercase, drop stop words, Snowball-stem, hash each stem to a uint32 index.

    `avg_len` stands in for the corpus's mean document length (in kept tokens);
    it only shapes length normalisation, so a fixed estimate is good enough.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_len: float = 60.0):
        self.k1, self.b, self.avg_len = k1, b, avg_len
        self._stemmer = snowballstemmer.stemmer("english")

    def terms(self, text: str) -> list[str]:
        words = [
            w.replace("'", "")
            for w in _WORD.findall(text.lower())
            if w not in STOPWORDS
        ]
        return self._stemmer.stemWords(words)

    def encode_docs(self, texts: list[str]) -> list[models.SparseVector]:
        out = []
        for text in texts:
            terms = self.terms(text)
            norm = self.k1 * (1 - self.b + self.b * len(terms) / self.avg_len)
            weights = {
                _index(t): tf * (self.k1 + 1) / (tf + norm)
                for t, tf in Counter(terms).items()
            }
            out.append(_vector(weights))
        return out

    def encode_query(self, text: str) -> models.SparseVector:
        return _vector({_index(t): 1.0 for t in set(self.terms(text))})


def _index(term: str) -> int:
    return zlib.crc32(term.encode())


def _vector(weights: dict[int, float]) -> models.SparseVector:
    indices = sorted(weights)
    return models.SparseVector(indices=indices, values=[weights[i] for i in indices])
