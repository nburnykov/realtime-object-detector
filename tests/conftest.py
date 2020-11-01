import pytest
from os import environ


@pytest.fixture()
def test_environment_vars():
    environ['APP_CONFIG'] = 'test.json'
    yield
    environ['APP_CONFIG'] = ''
