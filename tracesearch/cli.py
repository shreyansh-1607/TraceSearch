"""Terminal interface; JSON output makes demos and evaluation reproducible."""
import argparse
import json
from pathlib import Path
import sys

import mysql.connector

from . import __version__
from .config import Settings, SearchError
from .storage import Database
from .engine import Engine
from .evaluation import evaluate


def parser():
    root = argparse.ArgumentParser(description='Local incremental document search with FAISS and MySQL.')
    root.add_argument('--version', action='version', version=__version__)
    root.add_argument('--env-file', default='.env', help='Configuration file (environment variables take precedence).')
    sub = root.add_subparsers(dest='command', required=True)
    sub.add_parser('init-db', help='Create application tables inside your existing dedicated database.')
    sub.add_parser('model-info', help='Download/load the pinned local model and show its configuration.')
    sub.add_parser('status', help='Verify the saved index and show active collection metadata.')
    sub.add_parser('documents', help='List indexed files, not a fresh scan of the source folder.')
    index = sub.add_parser('index', help='Index a source folder, or update that same collection.')
    index.add_argument('folder')
    index.add_argument('--rebuild', action='store_true', help='Re-extract and re-embed every document.')
    update = sub.add_parser('update', help='Rescan the previously indexed folder.')
    update.add_argument('--rebuild', action='store_true', help='Required after model or chunk configuration changes.')
    search = sub.add_parser('search', help='Return source passages; does not generate an answer.')
    search.add_argument('query')
    search.add_argument('--top-k', type=int, default=5)
    search.add_argument('--min-score', type=float, help='Optional tested cosine threshold, not answer confidence.')
    search.add_argument('--json', action='store_true')
    evaluation = sub.add_parser('evaluate', help='Measure retrieval against labelled evidence.')
    evaluation.add_argument('dataset')
    evaluation.add_argument('--top-k', type=int, default=3)
    evaluation.add_argument('--min-score', type=float)
    evaluation.add_argument('--output', help='Save the report JSON (parent directories are created).')
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        settings = Settings.load(args.env_file)
        engine = Engine(settings)
        if args.command == 'init-db':
            Database(settings).initialize()
            result = {'initialized': True, 'database': settings.database,
                      'next': 'Run model-info, then index PATH_TO_DOCUMENTS.'}
        elif args.command == 'model-info':
            result = engine.embedder.identity
        elif args.command == 'status':
            result = engine.status()
        elif args.command == 'documents':
            meta = engine.db.active()
            result = engine.db.documents(meta['generation']) if meta else []
        elif args.command in {'index','update'}:
            result = engine.update(getattr(args, 'folder', None), rebuild=args.rebuild)
        elif args.command == 'search':
            result = engine.search(args.query, args.top_k, args.min_score)
            if not args.json:
                if not result:
                    print('No passages returned. The collection may be empty or all scores below your threshold.')
                for item in result:
                    location = f'PDF page {item["page"]}' if item['page'] else f'lines {item["line_start"]}-{item["line_end"]}'
                    print(f'\n{item["rank"]}. {item["document"]} | {location} | cosine {item["score"]:.4f}')
                    print(item['text'])
                print('\nResults are retrieved passages, not generated answers. Similarity is not confidence.')
                return 0
        else:
            result = evaluate(engine, args.dataset, args.top_k, args.min_score)
            if args.output:
                destination = Path(args.output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0
    except mysql.connector.Error as exc:
        print(f'MySQL error {exc.errno}: check your .env, database permissions, and that MySQL is running. '
              'Use init-db before indexing. Your password is not printed.', file=sys.stderr)
        return 2
    except (SearchError, OSError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('\nInterrupted. Any previously published generation remains available. Check status before retrying.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
