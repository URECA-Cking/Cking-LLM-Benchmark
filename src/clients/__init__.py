"""필요한 제공자만 로드해 고정 모델 E2E에서 로컬 모델 런타임을 초기화하지 않는다."""

from importlib import import_module

__all__ = ["LocalEmbeddingClient", "OpenAIEmbeddingClient", "OpenAIJudge", "OpenAITagger", "RerankerClient"]

_MODULES = {
    "LocalEmbeddingClient": "local_embedding",
    "OpenAIEmbeddingClient": "openai_embedding",
    "OpenAIJudge": "openai_judge",
    "OpenAITagger": "openai_tagger",
    "RerankerClient": "reranker",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f"src.clients.{_MODULES[name]}"), name)
    globals()[name] = value
    return value
