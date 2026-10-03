"""Incremental ingestion and exact cosine passage retrieval."""
from collections import defaultdict
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .storage import Database, load_index, save_index, workspace_id
from .config import SearchError
from .documents import scan, extract, chunk_pages
from .embeddings import LocalEmbedder, normalized


class Engine:
    def __init__(self, settings, embedder=None):
        self.settings = settings
        self.db = Database(settings)
        self._embedder = embedder

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = LocalEmbedder(self.settings)
        return self._embedder

    def update(self, root=None, rebuild=False, before_publish=None):
        started = time.perf_counter()
        with self.db.writer() as connection:
            identity = workspace_id(self.settings.workspace, create=True)
            self.db.bind_workspace(connection, identity)
            old = self.db.active()
            if root is None:
                if not old:
                    raise SearchError('First run: tracesearch index PATH_TO_DOCUMENTS')
                root = old['root_path']
            root = Path(root).resolve()
            if old and root != Path(old['root_path']).resolve():
                raise SearchError('This collection is bound to another source directory. Use a new database/workspace.')
            sources, warnings = scan(root, self.settings.max_file_mib)
            # Do not accidentally include index artifacts inside the source collection.
            if root == self.settings.workspace.resolve() or root.is_relative_to(self.settings.workspace.resolve()):
                raise SearchError('Source directory cannot be the workspace or a directory inside it.')
            embedder = self.embedder
            if old and old['signature'] != embedder.signature and not rebuild:
                raise SearchError('Model, dependency versions, or chunk settings changed. Run update --rebuild.')
            previous = {d['relative_path']: d for d in self.db.documents(old['generation'])} if old else {}
            current = {source.relative_path: source for source in sources}
            added = sorted(current.keys() - previous.keys())
            deleted = sorted(previous.keys() - current.keys())
            modified = sorted(name for name in current.keys() & previous.keys()
                              if current[name].content_sha256 != previous[name]['content_sha256'])
            unchanged = sorted(current.keys() & previous.keys() - set(modified))
            report = dict(added=len(added), modified=len(modified), deleted=len(deleted),
                unchanged=len(unchanged), embedded_chunks=0, reused_chunks=0,
                documents=len(sources), warnings=warnings, rebuilt=bool(rebuild),
                changes=dict(added=added, modified=modified, deleted=deleted))
            old_index = load_index(self.settings.workspace, old) if old and not rebuild else None
            if old and not rebuild and not (added or modified or deleted):
                report.update(generation=old['generation'], no_op=True, chunks=old['chunk_count'],
                              reused_chunks=old['chunk_count'], seconds=round(time.perf_counter()-started, 4))
                return report
            old_chunks = defaultdict(list)
            if old_index is not None:
                for chunk in self.db.chunks(old['generation']):
                    old_chunks[chunk['document_id']].append(chunk)
            documents, chunks, vector_parts = [], [], []
            for source in sources:
                documents.append(source.record())
                reuse = source.relative_path in unchanged and old_index is not None
                if reuse:
                    document_chunks = [dict(c) for c in old_chunks[source.document_id]]
                    vectors = np.asarray([old_index.reconstruct(int(c['vector_id'])) for c in document_chunks],
                                         dtype=np.float32).reshape(-1, embedder.dimension)
                    report['reused_chunks'] += len(document_chunks)
                else:
                    document_chunks = chunk_pages(extract(source), embedder.tokenizer,
                        self.settings.chunk_tokens, self.settings.overlap_tokens)
                    if len(chunks) + len(document_chunks) > self.settings.max_chunks:
                        raise SearchError('Collection exceeds TRACE_MAX_CHUNKS; reduce its size or raise the limit.')
                    vectors = embedder.documents([c['passage'] for c in document_chunks])
                    if len(vectors) != len(document_chunks):
                        raise SearchError('Embedding count differs from chunk count.')
                    vectors = normalized(vectors, embedder.dimension)
                    report['embedded_chunks'] += len(document_chunks)
                    if not document_chunks:
                        warnings.append(f'Empty document: {source.relative_path}')
                if len(chunks) + len(document_chunks) > self.settings.max_chunks:
                    raise SearchError('Collection exceeds TRACE_MAX_CHUNKS.')
                for chunk in document_chunks:
                    chunk.update(vector_id=len(chunks), document_id=source.document_id)
                    chunks.append(chunk)
                if len(vectors):
                    vector_parts.append(vectors)
            # A second snapshot catches additions, deletions and content edits during the build.
            after, _ = scan(root, self.settings.max_file_mib)
            if [(s.relative_path,s.content_sha256) for s in after] != [(s.relative_path,s.content_sha256) for s in sources]:
                raise SearchError('Source collection changed during indexing. Old generation is retained; retry.')
            vectors = np.concatenate(vector_parts) if vector_parts else np.empty((0, embedder.dimension), dtype=np.float32)
            generation = uuid.uuid4().hex
            report.update(generation=generation, no_op=False, chunks=len(chunks),
                          seconds=round(time.perf_counter()-started, 4))
            meta = dict(generation=generation, root_path=str(root), signature=embedder.signature,
                dimension=embedder.dimension, document_count=len(documents), chunk_count=len(chunks),
                workspace_id=identity, model=embedder.identity, report=report)
            checksum = save_index(self.settings.workspace, generation, vectors, meta)
            meta['index_sha256'] = checksum
            # Reopen and validate before any SQL pointer can refer to this generation.
            load_index(self.settings.workspace, meta)
            if before_publish is not None:
                before_publish()  # Used by tests to inject a failure at the publication boundary.
            self.db.publish(connection, meta, documents, chunks, identity)
            report['seconds'] = round(time.perf_counter()-started, 4)
            return report

    def session(self):
        return SearchSession(self)

    def search(self, query, top_k=5, min_score=None):
        return self.session().search(query, top_k, min_score)

    def status(self):
        meta = self.db.active()
        if not meta:
            return dict(indexed=False, message='No active index. Run tracesearch index PATH.')
        load_index(self.settings.workspace, meta)
        return dict(indexed=True, generation=meta['generation'], source=meta['root_path'],
            documents=meta['document_count'], chunks=meta['chunk_count'], dimension=meta['dimension'],
            created_at=str(meta['created_at']), index_verified=True,
            last_update=json.loads(meta['report_json']) if isinstance(meta['report_json'],str) else meta['report_json'])


class SearchSession:
    """Pins a generation so multiple searches/evaluation cannot mix snapshots."""
    def __init__(self, engine):
        self.engine = engine
        self.meta = engine.db.active()
        if not self.meta:
            raise SearchError('No active index. Run tracesearch index PATH first.')
        self.index = load_index(engine.settings.workspace, self.meta)
        if self.meta['signature'] != engine.embedder.signature:
            raise SearchError('Embedding configuration differs from the active index. Restore it or run update --rebuild.')

    def search(self, query, top_k=5, min_score=None):
        if not query.strip():
            raise SearchError('Search query cannot be empty.')
        if not 1 <= top_k <= 100:
            raise SearchError('top-k must be between 1 and 100.')
        if min_score is not None and not -1 <= min_score <= 1:
            raise SearchError('min-score must be between -1 and 1; it is not a probability.')
        if not self.index.ntotal:
            return []
        vectors = self.engine.embedder.query(query)
        scores, ids = self.index.search(vectors, min(top_k, self.index.ntotal))
        selected = [(int(i), float(score)) for i, score in zip(ids[0], scores[0])
                    if i >= 0 and (min_score is None or score >= min_score)]
        passages = self.engine.db.passages(self.meta['generation'], [i for i, _ in selected])
        if len(passages) != len(selected):
            raise SearchError('Index/metadata inconsistency: a retrieved passage is missing.')
        return [dict(rank=rank, score=round(score, 6), generation=self.meta['generation'],
            chunk_id=i, document=passages[i]['relative_path'], page=passages[i]['page_number'],
            line_start=passages[i]['line_start'], line_end=passages[i]['line_end'], text=passages[i]['passage'])
            for rank, (i, score) in enumerate(selected, 1)]
