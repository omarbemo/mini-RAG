from pydantic import BaseModel, Field, model_validator
from typing import Optional

class ProcessRequest(BaseModel):
    file_id : str = None
    # Characters. Upper bound matches INPUT_DAFAULT_MAX_CHARACTERS: embedding
    # input is cut there, so bigger chunks would be partly ignored.
    chunk_size : Optional[int] = Field(800, ge=100, le=1024)
    overlap : Optional[int] = Field(100, ge=0)
    do_reset : Optional[int] = 0
    do_push : Optional[int] = 1      # also embed + store in Qdrant now

    @model_validator(mode="after")
    def overlap_below_chunk_size(self):
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self
