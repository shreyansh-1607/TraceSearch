"""Unit, MySQL integration and real-model checks for the complete search workflow."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import mysql.connector
import numpy as np
import pytest

from tracesearch.config import Settings, SearchError, read_env
from tracesearch.documents import scan, extract, chunk_pages, Source
from tracesearch.embeddings import normalized
from tracesearch.engine import Engine
from tracesearch.evaluation import evaluate
from tracesearch.storage import save_index, load_index, workspace_id



def test_env_quotes_and_equals(tmp_path):
    path = tmp_path/'settings.env'
    path.write_text('# comment\nTRACE_DB_PASSWORD="a=b#c"\n')
    assert read_env(path)['TRACE_DB_PASSWORD'] == 'a=b#c'


def test_env_overrides_file(tmp_path, monkeypatch):
    path = tmp_path/'settings.env'
    path.write_text('TRACE_DB_PORT=3307\n')
    monkeypatch.setenv('TRACE_DB_PORT', '3308')
    assert Settings.load(path).port == 3308


@pytest.mark.parametrize('options', [dict(overlap_tokens=180), dict(chunk_tokens=0),
    dict(revision='main'), dict(database='bad-name'), dict(port=0), dict(batch_size=0)])
def test_invalid_config(options):
    with pytest.raises(ValueError):
        replace(Settings(), **options).validate()


def test_scan_ignores_hidden_and_unsupported(tmp_path):
    (tmp_path/'a.txt').write_text('hello')
    (tmp_path/'private.bin').write_bytes(b'abc')
    (tmp_path/'.hidden.txt').write_text('secret')
    sources, _ = scan(tmp_path, 1)
    assert [s.relative_path for s in sources] == ['a.txt']


def test_file_limit(tmp_path):
    (tmp_path/'large.txt').write_bytes(b'x' * (1024**2 + 1))
    with pytest.raises(SearchError, match='per-file limit'):
        scan(tmp_path, 1)


def test_chunk_offsets_preserve_text_and_overlap(fake):
    text = 'one two\nthree four five six seven'
    chunks = chunk_pages([(text,None)], fake.tokenizer, 4, 1)
    assert [c['passage'] for c in chunks] == ['one two\nthree four', 'four five six seven']
    assert chunks[0]['line_start'] == 1 and chunks[0]['line_end'] == 2
    assert chunks[1]['line_start'] == 2


def test_pdf_page_locations(fake):
    result = chunk_pages([('first page',1),('second page',2)],fake.tokenizer,4,1)
    assert [c['page_number'] for c in result] == [1,2]
    assert all(c['line_start'] is None for c in result)


def test_extract_rejects_changed_bytes(tmp_path):
    file = tmp_path/'text.txt'
    file.write_text('before')
    sources, _ = scan(tmp_path,1)
    file.write_text('after')
    with pytest.raises(SearchError, match='changed after scanning'):
        extract(sources[0])


def test_invalid_utf8_is_explicit(tmp_path):
    (tmp_path/'text.txt').write_bytes(b'\xff\xfe')
    sources, _ = scan(tmp_path,1)
    with pytest.raises(SearchError, match='Cannot extract'):
        extract(sources[0])


@pytest.mark.parametrize('data', [np.array([[0,0]]), np.array([[float('nan'),1]]), np.array([[1,2,3]])])
def test_bad_vectors(data):
    with pytest.raises(SearchError):
        normalized(data,2)


def test_real_faiss_roundtrip_and_checksum(tmp_path):
    identity = workspace_id(tmp_path,create=True)
    meta = dict(generation='a'*32, signature='b'*64, dimension=2, chunk_count=2,workspace_id=identity)
    meta['index_sha256'] = save_index(tmp_path, meta['generation'], np.eye(2,dtype='float32'),meta)
    index = load_index(tmp_path,meta)
    scores, ids = index.search(np.array([[1,0]],dtype='float32'),1)
    assert ids[0,0] == 0 and scores[0,0] == 1
    path = tmp_path/'generations'/meta['generation']/'index.faiss'
    with open(path,'ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(SearchError,match='checksum'):
        load_index(tmp_path,meta)


def test_original_sample_pdf_extracts():
    path = Path(__file__).resolve().parents[1]/'samples/documents/lab_handbook.pdf'
    data = path.read_bytes()
    source = Source(path,path.name,hashlib.sha256(data).hexdigest(),len(data))
    pages = extract(source)
    assert len(pages) == 2 and '5 PM' in pages[0][0]
    assert 'dataset version' in pages[1][0]


@pytest.mark.integration
def test_first_index_search_and_persistence(engine, corpus):
    report = engine.update(corpus)
    assert report['added'] == 2 and report['embedded_chunks'] == 2
    fresh = Engine(engine.settings,engine.embedder)
    result = fresh.search('database transactions atomicity',1)[0]
    assert result['document'] == 'beta.md' and 'transactions' in result['text']
    assert fresh.status()['index_verified']


@pytest.mark.integration
def test_unchanged_update_embeds_nothing(engine, corpus):
    first = engine.update(corpus)
    count = engine.embedder.embedded
    result = engine.update()
    assert result['no_op'] and result['generation'] == first['generation']
    assert result['embedded_chunks'] == 0 and engine.embedder.embedded == count


@pytest.mark.integration
def test_added_modified_deleted_and_reused(engine, corpus):
    engine.update(corpus)
    (corpus/'alpha.txt').write_text('New astronomy fact: a sapphire comet arrives.')
    (corpus/'new.txt').write_text('Bananas are yellow fruit.')
    result = engine.update()
    assert (result['added'],result['modified'],result['unchanged']) == (1,1,1)
    assert result['reused_chunks'] == 1 and result['embedded_chunks'] == 2
    (corpus/'beta.md').unlink()
    result = engine.update()
    assert result['deleted'] == 1 and result['embedded_chunks'] == 0
    assert all(r['document'] != 'beta.md' for r in engine.search('database',100))
    assert not any('distant stars' in r['text'] for r in engine.search('astronomy',100))


@pytest.mark.integration
def test_delete_all_and_empty_search(engine, corpus):
    engine.update(corpus)
    for path in corpus.iterdir():
        path.unlink()
    result = engine.update()
    assert result['deleted'] == 2 and result['chunks'] == 0
    assert engine.search('anything') == []


@pytest.mark.integration
def test_empty_text_is_tracked(engine, corpus):
    (corpus/'empty.txt').write_text('')
    result = engine.update(corpus)
    assert result['documents'] == 3 and result['chunks'] == 2
    assert engine.update()['no_op']


@pytest.mark.integration
def test_failed_extraction_keeps_old_generation(engine, corpus):
    before = engine.update(corpus)
    (corpus/'bad.pdf').write_bytes(b'not a PDF')
    with pytest.raises(SearchError,match='Cannot extract'):
        engine.update()
    assert engine.status()['generation'] == before['generation']


@pytest.mark.integration
def test_failure_after_index_write_before_publication(engine, corpus):
    before = engine.update(corpus)
    (corpus/'alpha.txt').write_text('replaced information')
    def fail():
        raise RuntimeError('simulated interruption')
    with pytest.raises(RuntimeError,match='simulated interruption'):
        engine.update(before_publish=fail)
    assert engine.status()['generation'] == before['generation']
    assert any('distant stars' in r['text'] for r in engine.search('stars',100))
    assert engine.update()['generation'] != before['generation']


@pytest.mark.integration
def test_sql_transaction_rolls_back_on_insert_error(engine, corpus, monkeypatch):
    before = engine.update(corpus)
    (corpus/'alpha.txt').write_text('new content')
    original = engine.db.publish
    def broken(connection,meta,documents,chunks,identity):
        chunks[1]['vector_id'] = chunks[0]['vector_id']
        original(connection,meta,documents,chunks,identity)
    monkeypatch.setattr(engine.db,'publish',broken)
    with pytest.raises(mysql.connector.IntegrityError):
        engine.update()
    assert engine.status()['generation'] == before['generation']


@pytest.mark.integration
def test_reader_pins_old_generation_during_update(engine, corpus):
    engine.update(corpus)
    session = engine.session()
    (corpus/'alpha.txt').write_text('Different contents only')
    engine.update()
    assert any('distant stars' in r['text'] for r in session.search('stars',100))
    assert not any('distant stars' in r['text'] for r in engine.search('stars',100))


@pytest.mark.integration
def test_concurrent_writer_rejected(engine, corpus):
    with engine.db.writer():
        with pytest.raises(SearchError,match='Another update'):
            engine.update(corpus)


@pytest.mark.integration
def test_config_change_requires_rebuild(engine, corpus):
    engine.update(corpus)
    engine.embedder.signature = 'e'*64
    with pytest.raises(SearchError,match='rebuild'):
        engine.update()
    with pytest.raises(SearchError,match='configuration differs'):
        engine.search('anything')
    result = engine.update(rebuild=True)
    assert result['embedded_chunks'] == 2 and result['reused_chunks'] == 0


@pytest.mark.integration
def test_workspace_mismatch(engine, corpus, tmp_path):
    engine.update(corpus)
    other = Engine(replace(engine.settings, workspace=tmp_path/'other'),engine.embedder)
    with pytest.raises(SearchError,match='another workspace'):
        other.update(corpus)


@pytest.mark.integration
def test_root_cannot_silently_change(engine, corpus, tmp_path):
    engine.update(corpus)
    other = tmp_path/'other-source'
    other.mkdir()
    with pytest.raises(SearchError,match='another source'):
        engine.update(other)


@pytest.mark.integration
def test_source_change_during_embedding_aborts(engine, corpus, monkeypatch):
    before = engine.update(corpus)
    (corpus/'alpha.txt').write_text('start changing')
    original = engine.embedder.documents
    def changed(texts):
        vectors = original(texts)
        (corpus/'alpha.txt').write_text('changed during embedding')
        return vectors
    monkeypatch.setattr(engine.embedder,'documents',changed)
    with pytest.raises(SearchError,match='changed during indexing'):
        engine.update()
    assert engine.status()['generation'] == before['generation']


@pytest.mark.integration
def test_invalid_query_and_parameters(engine, corpus):
    engine.update(corpus)
    for query, k, score in [('',1,None),('x',0,None),('x',101,None),('x',1,1.1)]:
        with pytest.raises(SearchError):
            engine.search(query,k,score)


@pytest.mark.integration
def test_evaluation_requires_evidence_not_only_filename(engine, corpus, tmp_path):
    engine.update(corpus)
    dataset = tmp_path/'questions.json'
    dataset.write_text(json.dumps([
        dict(id='yes',query='database atomicity',evidence=[dict(document='beta.md',contains='transactions')]),
        dict(id='wrong',query='database atomicity',evidence=[dict(document='beta.md',contains='absent content')]),
        dict(id='no',query='weather',evidence=[])]))
    result = evaluate(engine,dataset,top_k=2)
    assert result['hit_rate_at_k'] == 0.5
    assert result['unsupported_query_return_rate'] == 1.0


@pytest.mark.integration
@pytest.mark.model
def test_real_model_corpus_and_incremental_reuse(engine, real_embedder):
    engine._embedder = real_embedder
    engine.settings = replace(engine.settings, chunk_tokens=180, overlap_tokens=30)
    root = Path(__file__).resolve().parents[1]
    first = engine.update(root/'samples/documents')
    assert first['documents'] == 2 and first['embedded_chunks'] > 0
    results = engine.search('When do I need to give back a borrowed lab adapter?',3)
    assert any(r['document']=='lab_handbook.pdf' and r['page']==1 for r in results)
    assert engine.update()['embedded_chunks'] == 0
    report = evaluate(engine,root/'samples/evaluation.json',top_k=3)
    assert report['hit_rate_at_k'] >= 0.75


def run_cli(arguments, environment=None):
    return subprocess.run([sys.executable,'-m','tracesearch',*arguments],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, encoding='utf-8', timeout=90)


def test_help_without_database():
    result = run_cli(['--help'])
    assert result.returncode == 0 and 'evaluate' in result.stdout


def test_config_error_has_nonzero_exit_without_secret():
    environment = dict(os.environ, TRACE_DB_PORT='invalid', TRACE_DB_PASSWORD='must-not-be-printed')
    result = run_cli(['status'],environment)
    assert result.returncode == 2 and 'Invalid configuration' in result.stderr
    assert 'must-not-be-printed' not in result.stdout + result.stderr


@pytest.mark.integration
@pytest.mark.model
def test_real_cli_loads_persisted_index(engine, real_embedder):
    engine._embedder = real_embedder
    engine.settings = replace(engine.settings,chunk_tokens=180,overlap_tokens=30)
    root = Path(__file__).resolve().parents[1]
    engine.update(root/'samples/documents')
    environment = dict(os.environ, TRACE_WORKSPACE=str(engine.settings.workspace),
        TRACE_DB_HOST=engine.settings.host, TRACE_DB_PORT=str(engine.settings.port),
        TRACE_DB_USER=engine.settings.user, TRACE_DB_PASSWORD=engine.settings.password,
        TRACE_DB_NAME=engine.settings.database, TRACE_MODEL=real_embedder.settings.model,
        TRACE_MODEL_REVISION=real_embedder.settings.revision,
        TRACE_OFFLINE=str(real_embedder.settings.offline).lower(),
        TRACE_CHUNK_TOKENS='180',TRACE_OVERLAP_TOKENS='30')
    result = run_cli(['search','How do I prevent circular wait deadlocks?','--json','--top-k','3'],environment)
    assert result.returncode == 0, result.stderr
    results = json.loads(result.stdout)
    assert any(r['document']=='notes.md' and 'circular wait' in r['text'] for r in results)
