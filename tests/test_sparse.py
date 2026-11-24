from mini_rag.stores.sparse.providers import Bm25Provider


def test_stems_and_drops_stop_words():
    assert Bm25Provider().terms("The invoices were PAID, isn't it?") == ["invoic", "paid", "isnt"]


def test_query_matches_doc_after_stemming():
    enc = Bm25Provider()
    (doc,) = enc.encode_docs(["Invoice payment terms"])
    assert set(enc.encode_query("invoices").indices) <= set(doc.indices)


def test_repeats_saturate_and_indices_are_sorted():
    enc = Bm25Provider()
    (once,) = enc.encode_docs(["cat"])
    (many,) = enc.encode_docs(["cat " * 20])
    assert once.indices == many.indices
    assert once.values[0] < many.values[0] < 1 + enc.k1
    (mixed,) = enc.encode_docs(["zebra apple mango"])
    assert mixed.indices == sorted(mixed.indices)


def test_stop_word_only_query_is_empty():
    assert Bm25Provider().encode_query("the of and").indices == []

