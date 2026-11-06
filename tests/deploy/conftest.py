import pytest

from tests.deploy.harness import Sandbox


@pytest.fixture
def sb(tmp_path):
    return Sandbox(tmp_path)
