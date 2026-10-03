"""Strict source snapshots and source-preserving, token-aware chunks."""
from dataclasses import dataclass
import hashlib
from io import BytesIO
import os
from pathlib import Path

from pypdf import PdfReader

from .config import SearchError

SUPPORTED = {'.txt', '.md', '.pdf'}
EXCLUDED = {'.git', '.venv', '__pycache__', 'node_modules'}


@dataclass
class Source:
    path: Path
    relative_path: str
    content_sha256: str
    byte_size: int

    @property
    def document_id(self):
        return hashlib.sha256(self.relative_path.encode('utf-8')).hexdigest()

    def record(self):
        return dict(document_id=self.document_id, relative_path=self.relative_path,
                    content_sha256=self.content_sha256, byte_size=self.byte_size)


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scan(root, max_file_mib):
    root = Path(root).resolve()
    if not root.is_dir():
        raise SearchError(f'Document directory does not exist: {root}')
    sources, warnings = [], []
    def fail(error):
        raise SearchError(f'Cannot scan directory: {error}')
    for parent, directories, files in os.walk(root, followlinks=False, onerror=fail):
        kept = []
        for name in sorted(directories):
            directory = Path(parent) / name
            if name.startswith('.') or name in EXCLUDED:
                continue
            if directory.is_symlink():
                warnings.append(f'Skipped symlink directory: {directory.relative_to(root)}')
            else:
                kept.append(name)
        directories[:] = kept
        for name in sorted(files):
            path = Path(parent) / name
            if name.startswith('.') or path.suffix.lower() not in SUPPORTED:
                continue
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                warnings.append(f'Skipped symlink file: {relative}')
                continue
            try:
                if path.stat().st_size > max_file_mib * 1024**2:
                    raise SearchError(f'{relative} exceeds the {max_file_mib} MiB per-file limit.')
                before = path.stat()
                digest = digest_file(path)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise SearchError(f'{relative} changed during scanning. Retry with stable source files.')
                sources.append(Source(path, relative, digest, after.st_size))
                if len(sources) > 5000:
                    raise SearchError('This edition supports at most 5000 documents per collection.')
            except OSError as exc:
                raise SearchError(f'Cannot read {relative}: {exc}') from exc
    return sorted(sources, key=lambda item: item.relative_path), warnings


def extract(source):
    try:
        data = source.path.read_bytes()
        if hashlib.sha256(data).hexdigest() != source.content_sha256:
            raise SearchError(f'{source.relative_path} changed after scanning; retry the update.')
        if source.path.suffix.lower() == '.pdf':
            reader = PdfReader(BytesIO(data), strict=False)
            if reader.is_encrypted:
                raise SearchError(f'{source.relative_path}: encrypted PDFs are unsupported.')
            pages = [(page.extract_text() or '', number) for number, page in enumerate(reader.pages, 1)]
            if not any(text.strip() for text, _ in pages):
                raise SearchError(f'{source.relative_path}: no extractable text. Scanned PDFs need OCR (not included).')
        else:
            pages = [(data.decode('utf-8-sig'), None)]
        if sum(len(text) for text, _ in pages) > 2_000_000:
            raise SearchError(f'{source.relative_path}: extracted text exceeds 2 million characters.')
        return pages
    except SearchError:
        raise
    except Exception as exc:
        raise SearchError(f'Cannot extract {source.relative_path}: {type(exc).__name__}: {exc}') from exc


def chunk_pages(pages, tokenizer, chunk_tokens, overlap_tokens):
    chunks = []
    step = chunk_tokens - overlap_tokens
    for text, page in pages:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                            truncation=False, verbose=False)
        offsets = [(int(a), int(b)) for a, b in encoded['offset_mapping'] if b > a]
        for start in range(0, len(offsets), step):
            end = min(start + chunk_tokens, len(offsets))
            char_start, char_end = offsets[start][0], offsets[end - 1][1]
            passage = text[char_start:char_end]
            if not passage.strip():
                continue
            chunks.append(dict(ordinal=len(chunks), passage=passage, page_number=page,
                line_start=None if page else text.count('\n', 0, char_start) + 1,
                line_end=None if page else text.count('\n', 0, char_end) + 1))
            if end == len(offsets):
                break
    return chunks
