"""Passage-level retrieval evaluation against explicit evidence labels."""
import json
from pathlib import Path
import time

from .config import SearchError


def evaluate(engine, dataset, top_k=3, min_score=None):
    try:
        questions = json.loads(Path(dataset).read_text(encoding='utf-8'))
        if not isinstance(questions, list) or not questions:
            raise ValueError('Expected a nonempty list of questions')
        seen = set()
        for item in questions:
            if not isinstance(item['id'], str) or item['id'] in seen:
                raise ValueError('Question IDs must be unique strings')
            seen.add(item['id'])
            if not isinstance(item['query'], str) or not item['query'].strip():
                raise ValueError('Every question needs a nonempty query')
            if not isinstance(item['evidence'], list):
                raise ValueError('Evidence must be a list; use [] for unsupported questions')
            for evidence in item['evidence']:
                if not all(isinstance(evidence[key], str) and evidence[key] for key in ('document', 'contains')):
                    raise ValueError('Evidence requires document and a nonempty supporting substring')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SearchError(f'Invalid evaluation dataset: {exc}') from exc
    session = engine.session()
    details, hits, reciprocal_ranks, recalls, latencies = [], [], [], [], []
    negative_returns, false_abstentions = [], []
    for item in questions:
        started = time.perf_counter()
        results = session.search(item['query'], top_k, min_score)
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        def matches(result, evidence):
            return result['document'] == evidence['document'] and evidence['contains'].casefold() in result['text'].casefold()
        evidence = item['evidence']
        ranks = [r['rank'] for r in results if any(matches(r, e) for e in evidence)]
        detail = dict(id=item['id'], query=item['query'], answerable=bool(evidence),
                      result_count=len(results), seconds=round(elapsed, 6),
                      results=[{key: r[key] for key in ('rank','document','page','score','text')} for r in results])
        if evidence:
            hit = bool(ranks)
            recall = sum(any(matches(r, e) for r in results) for e in evidence) / len(evidence)
            rr = 1/min(ranks) if ranks else 0.0
            hits.append(hit); recalls.append(recall); reciprocal_ranks.append(rr)
            false_abstentions.append(not results)
            detail.update(hit=hit, evidence_recall=recall, reciprocal_rank=rr)
        else:
            negative_returns.append(bool(results))
            detail['returned_results_for_unsupported_query'] = bool(results)
        details.append(detail)
    mean = lambda values: round(sum(values)/len(values), 6) if values else None
    return dict(generation=session.meta['generation'], top_k=top_k, min_score=min_score,
        questions=len(questions), answerable_questions=len(hits), unsupported_questions=len(negative_returns),
        hit_rate_at_k=mean(hits), mean_reciprocal_rank_at_k=mean(reciprocal_ranks),
        mean_evidence_recall_at_k=mean(recalls), mean_query_seconds=mean(latencies),
        unsupported_query_return_rate=mean(negative_returns), answerable_query_empty_rate=mean(false_abstentions),
        metric_note='Passage matching requires both source filename and labelled supporting substring. '
                    'Latency excludes initial model/index loading. Scores are not probabilities. '
                    'Unsupported return rate measures results, not generated answers; no LLM is used.',
        details=details)
