from .RerankerEnums import RerankerEnums
from .providers import OnnxRerankerProvider


class RerankerProviderFactory:
    def __init__(self, config):
        self.config = config

    def create(self, provider: str):
        if provider == RerankerEnums.ONNX.value:
            return OnnxRerankerProvider.from_pretrained()  # first run downloads ~570 MB

        return None
