"""모델 제공자를 선택하기 전에는 로컬 모델 런타임을 초기화하지 않는다."""

import subprocess
import sys


def test_http_import_does_not_load_local_model_runtime():
    result = subprocess.run([sys.executable, "-c", "import sys; from src.recommendation.http import CkingBackendClient; "
                             "assert 'torch' not in sys.modules; assert 'sentence_transformers' not in sys.modules"],
                            capture_output=True, timeout=30)
    assert result.returncode == 0


def test_public_provider_export_still_resolves_to_same_class():
    result = subprocess.run([sys.executable, "-c", "from src.clients import OpenAIEmbeddingClient; "
                             "from src.clients.openai_embedding import OpenAIEmbeddingClient as Direct; "
                             "assert OpenAIEmbeddingClient is Direct; import src.clients; "
                             "assert src.clients.OpenAIEmbeddingClient is Direct; "
                             "assert not hasattr(src.clients, 'unknown_provider')"], capture_output=True, timeout=30)
    assert result.returncode == 0
