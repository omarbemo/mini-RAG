from mini_rag.rag.fusion import rrf_fuse
from mini_rag.rag.sparse import Bm25Encoder


def test_stems_and_drops_stop_words():
    assert Bm25Encoder().terms("The invoices were PAID, isn't it?") == ["invoic", "paid", "isnt"]


def test_query_matches_doc_after_stemming():
    enc = Bm25Encoder()
    (doc,) = enc.encode_docs(["Invoice payment terms"])
    assert set(enc.encode_query("invoices").indices) <= set(doc.indices)


def test_repeats_saturate_and_indices_are_sorted():
    enc = Bm25Encoder()
    (once,) = enc.encode_docs(["cat"])
    (many,) = enc.encode_docs(["cat " * 20])
    assert once.indices == many.indices
    assert once.values[0] < many.values[0] < 1 + enc.k1
    (mixed,) = enc.encode_docs(["zebra apple mango"])
    assert mixed.indices == sorted(mixed.indices)


def test_stop_word_only_query_is_empty():
    assert Bm25Encoder().encode_query("the of and").indices == []


def test_rrf_rewards_agreement_and_keeps_first_seen_on_ties():
    # 1 and 3 tie (1/61 + 1/63) just above 2 (2/62); the tie keeps first-seen order.
    assert [i for i, _ in rrf_fuse([[1, 2, 3], [3, 2, 1]])] == [1, 3, 2]
    assert [i for i, _ in rrf_fuse([[1, 2], [2, 3]])][0] == 2
