"""MySQL metadata and immutable FAISS snapshots, with one SQL publication point."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import faiss
import mysql.connector

from .config import SearchError
from .documents import digest_file

SCHEMA = """
CREATE TABLE IF NOT EXISTS ts_meta (
    singleton TINYINT PRIMARY KEY,
    schema_version INT NOT NULL,
    workspace_id CHAR(36) NULL,
    active_generation CHAR(32) NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS ts_generations (
    generation CHAR(32) PRIMARY KEY,
    root_path TEXT NOT NULL,
    signature CHAR(64) NOT NULL,
    index_sha256 CHAR(64) NOT NULL,
    dimension INT NOT NULL,
    document_count INT NOT NULL,
    chunk_count INT NOT NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    report_json JSON NOT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS ts_documents (
    generation CHAR(32) NOT NULL,
    document_id CHAR(64) NOT NULL,
    relative_path TEXT NOT NULL,
    content_sha256 CHAR(64) NOT NULL,
    byte_size BIGINT NOT NULL,
    PRIMARY KEY (generation, document_id),
    FOREIGN KEY (generation) REFERENCES ts_generations(generation)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS ts_chunks (
    generation CHAR(32) NOT NULL,
    vector_id BIGINT NOT NULL,
    document_id CHAR(64) NOT NULL,
    ordinal INT NOT NULL,
    page_number INT NULL,
    line_start INT NULL,
    line_end INT NULL,
    passage MEDIUMTEXT NOT NULL,
    PRIMARY KEY (generation, vector_id),
    FOREIGN KEY (generation, document_id) REFERENCES ts_documents(generation, document_id)
) ENGINE=InnoDB;
"""


class Database:
    def __init__(self, settings):
        self.settings = settings

    @contextmanager
    def connect(self):
        connection = mysql.connector.connect(**self.settings.connection())
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self):
        with self.connect() as connection:
            cursor = connection.cursor()
            try:
                for statement in SCHEMA.split(';'):
                    if statement.strip():
                        cursor.execute(statement)
                cursor.execute('INSERT IGNORE INTO ts_meta (singleton, schema_version) VALUES (1, 1)')
                cursor.execute('SELECT schema_version FROM ts_meta WHERE singleton=1')
                if cursor.fetchone()[0] != 1:
                    raise SearchError('Unsupported database schema version.')
            finally:
                cursor.close()

    @contextmanager
    def writer(self):
        # GET_LOCK spans the entire update, not just the final SQL transaction.
        with self.connect() as connection:
            cursor = connection.cursor()
            name = 'tracesearch:' + hashlib.sha256(self.settings.database.encode()).hexdigest()[:48]
            cursor.execute('SELECT GET_LOCK(%s, 0)', (name,))
            if cursor.fetchone()[0] != 1:
                cursor.close()
                raise SearchError('Another update is running for this database. Try again later.')
            try:
                yield connection
            finally:
                if connection.is_connected():
                    connection.rollback()
                    cursor.execute('SELECT RELEASE_LOCK(%s)', (name,))
                    cursor.fetchone()
                cursor.close()

    def bind_workspace(self, connection, workspace_id):
        cursor = connection.cursor()
        try:
            cursor.execute('SELECT workspace_id FROM ts_meta WHERE singleton=1')
            row = cursor.fetchone()
            if not row:
                raise SearchError('Run tracesearch init-db first.')
            if row[0] and row[0] != workspace_id:
                raise SearchError('This database is bound to another workspace. Restore its workspace or use a new database.')
            if not row[0]:
                cursor.execute('UPDATE ts_meta SET workspace_id=%s WHERE singleton=1', (workspace_id,))
        finally:
            cursor.close()

    def active(self):
        with self.connect() as connection:
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute('SELECT g.*, m.workspace_id FROM ts_meta m LEFT JOIN ts_generations g '
                               'ON g.generation=m.active_generation WHERE m.singleton=1')
                row = cursor.fetchone()
                if not row:
                    raise SearchError('Run tracesearch init-db first.')
                return row if row['generation'] else None
            finally:
                cursor.close()

    def documents(self, generation):
        with self.connect() as connection:
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute('SELECT * FROM ts_documents WHERE generation=%s ORDER BY relative_path', (generation,))
                return cursor.fetchall()
            finally:
                cursor.close()

    def chunks(self, generation):
        with self.connect() as connection:
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute('SELECT * FROM ts_chunks WHERE generation=%s ORDER BY vector_id', (generation,))
                return cursor.fetchall()
            finally:
                cursor.close()

    def publish(self, connection, meta, documents, chunks, workspace_id):
        cursor = connection.cursor()
        connection.start_transaction()
        try:
            cursor.execute('INSERT INTO ts_generations '
                '(generation,root_path,signature,index_sha256,dimension,document_count,chunk_count,report_json) '
                'VALUES (%s,%s,%s,%s,%s,%s,%s,%s)',
                (meta['generation'], meta['root_path'], meta['signature'], meta['index_sha256'],
                 meta['dimension'], len(documents), len(chunks), json.dumps(meta['report'])))
            if documents:
                cursor.executemany('INSERT INTO ts_documents '
                    '(generation,document_id,relative_path,content_sha256,byte_size) VALUES (%s,%s,%s,%s,%s)',
                    [(meta['generation'], d['document_id'], d['relative_path'], d['content_sha256'], d['byte_size']) for d in documents])
            # Bounded SQL batches avoid constructing one giant max_allowed_packet.
            for start in range(0, len(chunks), 100):
                cursor.executemany('INSERT INTO ts_chunks '
                    '(generation,vector_id,document_id,ordinal,page_number,line_start,line_end,passage) '
                    'VALUES (%s,%s,%s,%s,%s,%s,%s,%s)',
                    [(meta['generation'], c['vector_id'], c['document_id'], c['ordinal'],
                      c['page_number'], c['line_start'], c['line_end'], c['passage']) for c in chunks[start:start+100]])
            cursor.execute('UPDATE ts_meta SET active_generation=%s WHERE singleton=1 AND workspace_id=%s',
                           (meta['generation'], workspace_id))
            if cursor.rowcount != 1:
                raise SearchError('Workspace binding changed during publication.')
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            cursor.close()

    def passages(self, generation, ids):
        if not ids:
            return {}
        with self.connect() as connection:
            cursor = connection.cursor(dictionary=True)
            try:
                placeholders = ','.join(['%s'] * len(ids))
                cursor.execute('SELECT c.*, d.relative_path FROM ts_chunks c JOIN ts_documents d '
                    'ON c.generation=d.generation AND c.document_id=d.document_id '
                    f'WHERE c.generation=%s AND c.vector_id IN ({placeholders})', (generation, *ids))
                return {int(row['vector_id']): row for row in cursor.fetchall()}
            finally:
                cursor.close()


# Local index files are fully written before Database.publish changes the active pointer.

def sync_file(path):
    with open(path, 'r+b') as stream:
        os.fsync(stream.fileno())


def workspace_id(workspace, create=False):
    workspace = Path(workspace)
    marker = workspace / 'workspace.json'
    if create:
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            with open(marker, 'x', encoding='utf-8') as stream:
                json.dump({'workspace_id': str(uuid.uuid4())}, stream)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            pass
    try:
        identity = json.loads(marker.read_text(encoding='utf-8'))['workspace_id']
        return str(uuid.UUID(identity))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SearchError('Workspace marker missing or invalid. Restore the matching workspace; '
                          'do not reuse a populated database with a new workspace.') from exc


def generation_dir(workspace, generation):
    if not re.fullmatch(r'[0-9a-f]{32}', generation):
        raise SearchError('Invalid generation identifier.')
    return Path(workspace) / 'generations' / generation


def save_index(workspace, generation, vectors, manifest):
    target = generation_dir(workspace, generation)
    target.mkdir(parents=True, exist_ok=False)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    path = target / 'index.faiss'
    faiss.write_index(index, str(path))
    sync_file(path)
    checksum = digest_file(path)
    manifest = {**manifest, 'index_sha256': checksum}
    with open(target / 'manifest.json', 'x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    return checksum


def load_index(workspace, meta):
    if workspace_id(workspace) != meta['workspace_id']:
        raise SearchError('Database and local workspace do not match.')
    target = generation_dir(workspace, meta['generation'])
    try:
        manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
        for field in ('generation', 'signature', 'dimension', 'chunk_count', 'index_sha256', 'workspace_id'):
            if manifest.get(field) != meta[field]:
                raise SearchError(f'Generation manifest mismatch: {field}.')
        path = target / 'index.faiss'
        if digest_file(path) != meta['index_sha256']:
            raise SearchError('FAISS index checksum mismatch. Restore the workspace from a trusted backup.')
        index = faiss.read_index(str(path))
        if index.d != meta['dimension'] or index.ntotal != meta['chunk_count']:
            raise SearchError('FAISS dimensions/count do not match MySQL metadata.')
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise SearchError('Unexpected FAISS distance metric.')
        return index
    except SearchError:
        raise
    except Exception as exc:
        raise SearchError(f'Cannot load generation {meta["generation"]}: {exc}') from exc
