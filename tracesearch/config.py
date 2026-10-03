"""Explicit configuration, with secrets kept outside source control."""
from dataclasses import dataclass
import os
from pathlib import Path
import re


class SearchError(Exception):
    """An actionable application error suitable for the terminal."""


def read_env(path):
    values = {}
    if not Path(path).exists():
        return values
    for number, raw in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            raise SearchError(f'Invalid configuration line {number}: expected KEY=value.')
        key, value = line.split('=', 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


@dataclass(frozen=True)
class Settings:
    host: str = '127.0.0.1'
    port: int = 3306
    user: str = 'tracesearch_app'
    password: str = ''
    database: str = 'tracesearch'
    workspace: Path = Path('.tracesearch')
    model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    revision: str = 'c9745ed1d9f207416be6d2e6f8de32d1f16199bf'
    offline: bool = False
    chunk_tokens: int = 180
    overlap_tokens: int = 30
    batch_size: int = 32
    max_file_mib: int = 50
    max_chunks: int = 50000

    @classmethod
    def load(cls, env_file='.env'):
        values = {**read_env(env_file), **os.environ}
        def get(name, default):
            return values.get('TRACE_' + name, default)
        try:
            offline = str(get('OFFLINE', 'false')).lower()
            if offline not in {'true', 'false'}:
                raise ValueError('TRACE_OFFLINE must be true or false')
            result = cls(host=get('DB_HOST', '127.0.0.1'), port=int(get('DB_PORT', 3306)),
                user=get('DB_USER', 'tracesearch_app'), password=get('DB_PASSWORD', ''),
                database=get('DB_NAME', 'tracesearch'), workspace=Path(get('WORKSPACE', '.tracesearch')).resolve(),
                model=get('MODEL', cls.model), revision=get('MODEL_REVISION', cls.revision),
                offline=offline == 'true', chunk_tokens=int(get('CHUNK_TOKENS', 180)),
                overlap_tokens=int(get('OVERLAP_TOKENS', 30)), batch_size=int(get('BATCH_SIZE', 32)),
                max_file_mib=int(get('MAX_FILE_MIB', 50)), max_chunks=int(get('MAX_CHUNKS', 50000)))
            result.validate()
            return result
        except ValueError as exc:
            raise SearchError(f'Invalid configuration: {exc}') from exc

    def validate(self):
        if not re.fullmatch(r'[A-Za-z0-9_]{1,64}', self.database):
            raise ValueError('Database name must contain only letters, digits, and underscores')
        if not 1 <= self.port <= 65535:
            raise ValueError('Invalid database port')
        if not 16 <= self.chunk_tokens <= 4096 or not 0 <= self.overlap_tokens < self.chunk_tokens:
            raise ValueError('Chunk tokens must be 16..4096 and overlap smaller than chunk size')
        if min(self.batch_size, self.max_file_mib, self.max_chunks) < 1:
            raise ValueError('Batch size and limits must be positive')
        if not re.fullmatch(r'[0-9a-f]{40}', self.revision):
            raise ValueError('TRACE_MODEL_REVISION must be a 40-character model commit SHA')

    def connection(self):
        return dict(host=self.host, port=self.port, user=self.user, password=self.password,
                    database=self.database, charset='utf8mb4', connection_timeout=10, autocommit=True)
