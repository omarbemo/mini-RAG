from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from mini_rag.stores.reranker.providers import OnnxRerankerProvider

VOCAB = {"<pad>": 0, "<unk>": 1, "cats": 2, "dogs": 3, "purr": 4, "bark": 5}


def tokenizer():
    tok = Tokenizer(models.WordLevel(VOCAB, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


class CountSession:
    """Fake cross-encoder: logit = count of token 4 ("purr") in the pair."""

    def __init__(self, inputs=("input_ids", "attention_mask")):
        self.inputs = inputs
        self.feeds = []

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in self.inputs]

    def run(self, _, feed):
        self.feeds.append(feed)
        return [(feed["input_ids"] == 4).sum(axis=1, keepdims=True).astype(np.float32)]


def test_scores_follow_logits_across_batches():
    session = CountSession()
    r = OnnxRerankerProvider(session, tokenizer(), batch_size=2)
    scores = r.score("cats", ["dogs bark", "cats purr purr", "purr"])
    assert scores == [0.0, 2.0, 1.0]
    assert len(session.feeds) == 2
    assert set(session.feeds[0]) == {"input_ids", "attention_mask"}


def test_long_pairs_are_truncated_not_failed():
    session = CountSession()
    r = OnnxRerankerProvider(session, tokenizer(), max_length=4)
    r.score("cats " * 50, ["purr " * 50])
    assert session.feeds[0]["input_ids"].shape[1] <= 4


def test_rejects_session_with_unknown_inputs():
    with pytest.raises(ValueError, match="unsupported"):
        OnnxRerankerProvider(CountSession(inputs=("input_ids", "pixel_values")), tokenizer())


def test_rejects_tokenizer_without_pad():
    tok = Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
    with pytest.raises(ValueError, match="<pad>"):
        OnnxRerankerProvider(CountSession(), tok)
