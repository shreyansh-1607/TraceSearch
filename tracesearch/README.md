# TraceSearch

A command-line tool for searching local PDFs and text files by meaning, using **Python, FAISS, a local embedding model, and MySQL**.

When documents change, TraceSearch processes the changed files and reuses embeddings for unchanged ones. Results show the matching passage and its source page or line numbers. It retrieves evidence; it does not generate chatbot answers.

## Features

- Text-based PDF, UTF-8 TXT and Markdown ingestion.
- Semantic retrieval with source references.
- Added/modified/deleted document detection using content hashes.
- Embedding reuse and zero new embeddings on unchanged updates.
- Complete saved index versions, transactional publication and recovery from failed updates.
- CLI, evaluation questions and 39 automated tests.

This compact edition combines storage code, tests and sample notes. The search algorithm and recovery design are unchanged.

## Setup

Use 64-bit Python 3.11 or 3.12 and a running MySQL 8.0+ server. The default model runs on CPU. Internet is needed initially for packages and model files; no hosted LLM or API key is required.

**Windows PowerShell**, from this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

For Python 3.11, substitute `py -3.11`. The explicit interpreter path works without environment activation.

**Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Below, `python` means the environment's interpreter. On Windows use `.\.venv\Scripts\python.exe` unless you have activated the environment.

### MySQL

Open MySQL as your database administrator, for example `mysql -u root -p`. Run these statements with your own application password. Use a **new dedicated database**, not your library project's database.

```sql
CREATE DATABASE tracesearch CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;
CREATE USER 'tracesearch_app'@'localhost' IDENTIFIED BY 'YOUR_LOCAL_PASSWORD';
CREATE USER 'tracesearch_app'@'127.0.0.1' IDENTIFIED BY 'YOUR_LOCAL_PASSWORD';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, REFERENCES ON tracesearch.* TO 'tracesearch_app'@'localhost';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, REFERENCES ON tracesearch.* TO 'tracesearch_app'@'127.0.0.1';
```

Both loopback account variants accommodate MySQL host matching. Edit `.env` with the matching host, port, username and password. Environment variables override the file. Keep credentials out of Git.

```bash
python -m tracesearch init-db
python -m tracesearch model-info
python -m tracesearch index samples/documents
python -m tracesearch search "How can a scheduler prevent starvation?" --top-k 3
```

`init-db` creates tables in the existing database. `model-info` downloads/loads the pinned model. After downloading it successfully, set `TRACE_OFFLINE=true` to require cached model files.

## Commands

Prefix these with `python -m tracesearch`:

| Command | Use |
|---|---|
| `index FOLDER` | Index a collection for the first time |
| `update` | Rescan that folder and process changes |
| `update --rebuild` | Reprocess everything after changing model/chunk settings |
| `search "QUERY" --top-k 3` | Show ranked passages |
| `search "QUERY" --json` | Return results as JSON |
| `status` | Verify the saved index and show counts/version |
| `documents` | List indexed files |
| `evaluate samples/evaluation.json --top-k 3` | Run labelled retrieval checks |

For another config file: `python -m tracesearch --env-file PATH COMMAND`.

Try an unchanged `update`: expect `embedded_chunks: 0` and `no_op: true`. Add a temporary TXT file to the indexed folder, update and search for its contents. Delete that demo file and update again to demonstrate removal. Restore sample edits before comparing evaluation results.

## How it works

```text
Source files → hashes → extract text → token chunks → local embeddings
                                                        ↓
Query → query embedding → FAISS search → MySQL passage lookup → results
```

MySQL stores hashes, passage text, source locations and index versions. FAISS stores vectors. Normalized vectors and a flat inner-product index give exact cosine-similarity ranking.

Updates re-embed whole added/modified documents and reuse unchanged vectors. The FAISS index is rebuilt from active vectors: "incremental" saves embedding work, not all scanning or rebuilding. Renames count as deletion plus addition.

Before publication, the program rescans sources, writes/checks a new immutable index, then inserts metadata and changes the active pointer in one MySQL transaction. One writer is allowed per database. Readers use a fixed generation; old versions remain available for active readers.

The source folder, database and workspace form one collection. Keep their pairing intact. Changing the model revision, chunk settings or relevant dependency versions requires a rebuild.

## Tests and evaluation

```bash
python -m pytest -q
python -m tracesearch evaluate samples/evaluation.json --top-k 3 --output reports/local-evaluation.json
```

By default, database/model integration tests are skipped. For all tests, create a **disposable database ending in `_test`**, grant the application account access to its tables, and configure it in `.env.test`. Tests clear that database's `ts_*` rows before each integration case.

PowerShell:

```powershell
$env:TRACE_TEST_ENV = ".env.test"
$env:TRACE_DB_NAME = "tracesearch_test"
$env:TRACE_RUN_INTEGRATION = "1"
$env:TRACE_RUN_MODEL = "1"
python -m pytest -q
```

Linux uses `export NAME=value`. Remove overrides before normal use. Model tests need cached weights or download access. GitHub Actions includes Windows/Linux unit checks and a MySQL/model integration job.

The original sample material is MIT-licensed: combined study notes and a two-page fictional lab handbook. The evaluation has 16 answerable and 4 unsupported questions. A hit requires both the expected source and supporting text. This is a small demo set, not a general accuracy benchmark.

**Compact-edition verification:** all 39 tests passed on Windows with Python 3.12.14, MySQL 8.0.45, real FAISS and the pinned embedding model running offline. Tests cover additions, edits, deletions, embedding reuse, failed publication, corruption, concurrent writers and a separate CLI process. Wheel and source-distribution builds also passed. Linux/GitHub CI is configured but has not been run locally. All original tests are retained.

## Limits and troubleshooting

- Similarity is not confidence. Unsupported questions still return nearest passages by default. `--min-score` is an optional filter to calibrate yourself, not reliable abstention.
- No OCR, generated answers, web UI, automatic watching or user-level access control. Use trusted local documents and local MySQL.
- Memory grows with collection size. Default guards: 5,000 files, 50,000 chunks, 50 MiB per file and 2 million extracted characters per file. These are not benchmarked capacity claims.
- A malformed supported file aborts the update and leaves the prior version active. That version may be stale; `status` does not rescan sources.
- Old versions retain text and use storage. There is no automatic cleanup or secure erasure. Back up the database and matching workspace together while updates are stopped.
- Source roots are fixed per collection. Another folder needs a new database/workspace. PDF page references are physical positions, not printed page labels.
- Only load your trusted workspace. Checksums detect corruption; they cannot protect against someone who can alter files and SQL. No exhaustive power-loss or production security guarantee is claimed.

For connection errors, check MySQL's service, credentials and port. Create the database before `init-db`. Restore indexed settings or rebuild after a configuration mismatch. A scanned PDF needs OCR outside this version. Restore a matching backup for damaged index files, or explicitly rebuild from stable source documents.

## Files (21 total)

| File | Purpose |
|---|---|
| `README.md` | Setup, usage, design, limitations and file guide |
| `.env.example` | Configuration template without real credentials |
| `.gitignore` | Excludes secrets, indexes, caches and local data |
| `LICENSE` | MIT reuse terms |
| `pyproject.toml` | Dependencies, installation, CLI entry point and test settings |
| `MANIFEST.in` | Includes samples/tests in source distributions |
| `tracesearch/__init__.py` | Package identity and version |
| `tracesearch/__main__.py` | Supports `python -m tracesearch` |
| `tracesearch/cli.py` | Commands, output and error handling |
| `tracesearch/config.py` | Loads and validates settings |
| `tracesearch/documents.py` | Scanning, hashing, extraction and chunking |
| `tracesearch/embeddings.py` | Local model and normalized vectors |
| `tracesearch/storage.py` | SQL schema/queries, writer lock and saved FAISS versions |
| `tracesearch/engine.py` | Incremental updates, retrieval and consistency |
| `tracesearch/evaluation.py` | Retrieval metrics against labelled evidence |
| `tests/conftest.py` | Shared test fixtures and isolated database setup |
| `tests/test_search.py` | All 39 tests, including integration and CLI checks |
| `samples/documents/notes.md` | OS, networking, database and search sample text |
| `samples/documents/lab_handbook.pdf` | Two-page PDF sample |
| `samples/evaluation.json` | Questions and supporting-passage labels |
| `.github/workflows/tests.yml` | Automated Windows/Linux checks |

Generated credentials, indexes and caches are local runtime files, not repository files.

## GitHub and your CV

Use this folder as the repository root. If updating an older checkout, remove the retired docs/modules too; copying files over the old version will leave extras. Preserve your `.git` directory and private `.env`.

Review `git status` and staged filenames before committing. For a new repository, initialize Git, commit the reviewed files, connect your actual remote URL and push. This package does not upload anything to GitHub.

A fair description is: "Implemented incremental document search using local embeddings, FAISS and MySQL, with source references and versioned index publication." Add measured results only after reproducing them. Be ready to explain embedding reuse, SQL/file consistency, failure tests and your own contributions.
