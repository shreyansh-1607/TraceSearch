"""Fast deterministic tests plus opt-in real MySQL and local-model tests."""
from dataclasses import replace
import hashlib
import os
import re

import numpy as np
import pytest

from tracesearch.config import Settings
from tracesearch.storage import Database
from tracesearch.engine import Engine


class WordTokenizer:
    def __call__(self, text, **kwargs):
        offsets = [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]
        return {'offset_mapping': offsets, 'input_ids': list(range(len(offsets)))}


class FakeEmbedder:
    """A test double, never exposed as the production semantic-search model."""
    tokenizer = WordTokenizer()
    dimension = 64
    signature = 'f' * 64
    identity = {'test_double': True, 'dimension': 64}

    def __init__(self):
        self.embedded = 0

    def encode(self, texts):
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r'\w+', text.lower()):
                result[row, int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % self.dimension] += 1
            if not result[row].any():
                result[row, 0] = 1
        return result / np.linalg.norm(result, axis=1, keepdims=True)

    def documents(self, texts):
        self.embedded += len(texts)
        return self.encode(texts)

    def query(self, text):
        return self.encode([text])


@pytest.fixture
def fake():
    return FakeEmbedder()


@pytest.fixture
def engine(tmp_path, fake):
    if os.environ.get('TRACE_RUN_INTEGRATION') != '1':
        pytest.skip('Set TRACE_RUN_INTEGRATION=1 and select a dedicated database ending in _test.')
    settings = Settings.load(os.environ.get('TRACE_TEST_ENV', '.env'))
    if not settings.database.endswith('_test'):
        pytest.fail('Integration tests require a disposable database name ending in _test.')
    settings = replace(settings, workspace=tmp_path / 'workspace', chunk_tokens=32, overlap_tokens=4)
    database = Database(settings)
    database.initialize()
    with database.connect() as connection:
        cursor = connection.cursor()
        for table in ('ts_chunks', 'ts_documents', 'ts_generations'):
            cursor.execute(f'DELETE FROM {table}')
        cursor.execute('UPDATE ts_meta SET active_generation=NULL, workspace_id=NULL WHERE singleton=1')
        cursor.close()
    return Engine(settings, fake)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / 'docs'
    root.mkdir()
    (root/'alpha.txt').write_text('Alpha telescope astronomy planets orbit the distant stars.', encoding='utf-8')
    (root/'beta.md').write_text('Beta database transactions prevent partial commits and preserve atomicity.', encoding='utf-8')
    return root


@pytest.fixture(scope='session')
def real_embedder():
    if os.environ.get('TRACE_RUN_MODEL') != '1':
        pytest.skip('Set TRACE_RUN_MODEL=1 to run the downloaded pretrained model.')
    from tracesearch.embeddings import LocalEmbedder
    return LocalEmbedder(Settings.load(os.environ.get('TRACE_TEST_ENV', '.env')))
