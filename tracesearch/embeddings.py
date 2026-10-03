"""One local, pinned pretrained model; no answer-generating LLM or API calls."""
import hashlib
from importlib.metadata import version
import json

import numpy as np

from .config import SearchError


def normalized(values, dimension):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != dimension or not np.isfinite(values).all():
        raise SearchError('Embedding model returned invalid vectors.')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise SearchError('Embedding model returned a zero vector.')
    return np.ascontiguousarray(values / norms, dtype=np.float32)


class LocalEmbedder:
    def __init__(self, settings):
        from sentence_transformers import SentenceTransformer
        self.settings = settings
        try:
            self.model = SentenceTransformer(settings.model, revision=settings.revision,
                local_files_only=settings.offline, device='cpu', trust_remote_code=False)
        except Exception as exc:
            raise SearchError('Could not load the pinned embedding model. Check its name/revision and '
                'network access, or download it before setting TRACE_OFFLINE=true. '
                f'Detail: {type(exc).__name__}: {exc}') from exc
        self.tokenizer = self.model.tokenizer
        if not self.tokenizer.is_fast:
            raise SearchError('A fast tokenizer with character offsets is required.')
        overhead = self.tokenizer.num_special_tokens_to_add(pair=False)
        if settings.chunk_tokens + overhead > self.model.max_seq_length:
            raise SearchError(f'Chunk size is too large for this model (limit {self.model.max_seq_length} '
                              f'including {overhead} special tokens).')
        dimension_method = getattr(self.model, 'get_embedding_dimension', None)
        if dimension_method is None:
            dimension_method = self.model.get_sentence_embedding_dimension
        self.dimension = int(dimension_method())
        self.identity = dict(model=settings.model, revision=settings.revision, dimension=self.dimension,
            chunk_tokens=settings.chunk_tokens, overlap_tokens=settings.overlap_tokens,
            chunker='page-offset-token-v1', loader='utf8-pypdf-v1', encoder='query-document-normalized-v1',
            libraries={name: version(name) for name in ('sentence-transformers', 'transformers', 'pypdf')})
        self.signature = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()

    def documents(self, texts):
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return normalized(self.model.encode_document(texts, batch_size=self.settings.batch_size,
            show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True), self.dimension)

    def query(self, text):
        count = len(self.tokenizer(text, add_special_tokens=True)['input_ids'])
        if count > self.model.max_seq_length:
            raise SearchError(f'Query is too long: {count} tokens; model limit is {self.model.max_seq_length}.')
        return normalized(self.model.encode_query([text], show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True), self.dimension)
