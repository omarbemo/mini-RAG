import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from mini_rag.api.schemas.data import ProcessRequest
from mini_rag.controllers import ProcessController

PAGE = Document(
    page_content="INTRO\n●\n" + "Qdrant stores vectors and payloads. " * 60,
    metadata={"page": 2, "source": "/abs/server/path.pdf", "author": "x"},
)


def test_chunks_are_sized_clean_and_carry_only_name_and_page():
    chunks = ProcessController("t").process_file_content([PAGE], "doc.pdf")
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 800 for c in chunks)
    assert all("●" not in c.page_content for c in chunks)
    assert all(c.metadata == {"source": "doc.pdf", "page": 2} for c in chunks)


def test_metadata_without_page_is_just_source():
    doc = Document(page_content="hello world " * 5, metadata={"author": "x"})
    (chunk,) = ProcessController("t").process_file_content([doc], "a.txt")
    assert chunk.metadata == {"source": "a.txt"}


def test_request_rejects_bad_sizes():
    assert ProcessRequest().chunk_size == 800
    for bad in ({"overlap": 800}, {"chunk_size": 5000}, {"chunk_size": 100, "overlap": 100}):
        with pytest.raises(ValidationError):
            ProcessRequest(**bad)
