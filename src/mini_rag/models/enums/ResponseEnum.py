from enum import Enum

class ResponseEnum(Enum):
    FILE_TYPE_NOT_SUPPORTED = "file_type_not_supported"
    FILE_SIZE_EXCEEDED = "file_size_exceeded"
    FILE_UPLOADED_SUCCESS = "file_uploaded_success"
    FILE_UPLOAD_FAILED = "file_upload_failed"
    FILE_VALIDATED_SUCCESSFUL = "file_validated_successfuly"
    FILE_CHUNKS_SUCCESS = "file_chunks_successfuly"
    FILE_CHUNKS_FAILED = "file_chunks_failed"
    PROCESSING_SUCCESS = "processing_success"
    PROCESSING_FAILED = "processing_failed"
    NO_FILES_ERROR = "no_files_retrived"
    FILE_ID_ERROR = "no_file_id_with_this_id"