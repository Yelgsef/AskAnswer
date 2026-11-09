"""Penalized Accuracy as the Evaluation tab describes it (a local reproduction, not the organizers' hidden scorer).

Per question: +1 correct, 0 abstain or missing, -1 wrong. An answer is correct only if its type equals the reference type and
its value matches under that type's rule. Strings, list items and table name cells also match any alias of the reference.
Inference never imports this module: gold answers are read only by evaluation commands.

    python -m datamate.scoring --data data/train --pred RUN/submission.jsonl --output RUN/score.json [--questions IDS.jsonl]
"""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
import math
from pathlib import Path
import random
import unicodedata


def norm(v):
    s = unicodedata.normalize('NFC', str(v)).lower()
    return ' '.join(''.join(c if c.isalnum() or c.isspace() else ' ' for c in s).split())


def key(v):
    if isinstance(v, bool): return ('bool', v)
    if isinstance(v, (int, float)):
        if not math.isfinite(v): raise ValueError('nonfinite')
        return ('num', Decimal(str(v)))
    if v is None: return ('null',)
    if isinstance(v, str): return ('str', norm(v))
    raise ValueError('invalid answer cell')


def score(pred, gold):
    if not pred or pred.get('type') == 'abstain': return 0
    if 'reason' in pred and not isinstance(pred['reason'], str): return -1
    if pred.get('type') != gold.get('type'): return -1
    t = gold['type']; p = pred.get('answer'); g = gold.get('answer')
    try:
        if t == 'unanswerable': return 1
        if t == 'string': ok = isinstance(p, str) and any(key(p) == key(x) for x in [g] + (gold.get('alias') or []))
        elif t == 'percent':
            ok = not isinstance(p, bool) and isinstance(p, (int, float)) and Decimal(str(p)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP) == Decimal(str(g)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
        elif t == 'date': ok = isinstance(p, str) and p == g
        elif t in ('number', 'boolean'): ok = key(p) == key(g)
        elif t == 'list':  # alias of a list/table holds the accepted forms of each item / of each row's name
            al = gold.get('alias') or [[] for _ in g]
            ok = isinstance(p, list) and len(p) == len(g) and all(key(x) in {key(y) for y in [z, *a]} for x, z, a in zip(p, g, al))
        elif t == 'table':
            pc, gc = p['columns'], g['columns']; pr, gr = p['rows'], g['rows']
            if not all(isinstance(x, list) for x in (pc, gc, pr, gr)) or len(pc) != len(gc): return -1
            pn, gn = list(map(norm, pc)), list(map(norm, gc))
            indices = [pn.index(c) for c in gn] if len(set(pn)) == len(pn) and set(pn) == set(gn) else list(range(len(gc)))
            if any(not isinstance(r, list) or len(r) != len(gc) for r in pr + gr): return -1
            left = {tuple(key(r[i]) for i in indices) for r in pr}
            ok = len(left) == len({tuple(map(key, r)) for r in gr})
            for r, a in zip(gr, gold.get('alias') or [[] for _ in gr]):
                hit = next((x for x in left if x[1:] == tuple(map(key, r[1:])) and x[0] in {key(y) for y in [r[0], *a]}), None)
                ok = ok and hit is not None
                left.discard(hit)
        else: ok = False
        return 1 if ok else -1
    except (ValueError, TypeError, KeyError, IndexError, InvalidOperation): return -1


def penalized_accuracy(preds, refs):
    """preds/refs keyed by question id. Questions missing from preds count as abstain."""
    s = {qid: score(preds.get(qid), ref) for qid, ref in refs.items()}
    c = Counter(s.values())
    n = len(refs)
    return {"PA": (c[1] - c[-1]) / n if n else 0.0, "n": n, "correct": c[1], "wrong": c[-1], "abstain": c[0],
            "precision": c[1] / (c[1] + c[-1]) if c[1] + c[-1] else float("nan"), "per_question": s}


def read_jsonl(path):
    out = {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip(): continue
        x = json.loads(line)
        if x['id'] in out: raise ValueError('duplicate ID ' + str(x['id']))
        out[x['id']] = x
    return out


def metrics(rows):
    n = len(rows); counts = Counter(r['score'] for r in rows); answered = counts[1] + counts[-1]
    return {'N': n, 'correct': counts[1], 'wrong': counts[-1], 'abstain': counts[0], 'PA': (counts[1] - counts[-1]) / n if n else None, 'precision': counts[1] / answered if answered else None, 'coverage': answered / n if n else None, 'answer_type_accuracy': sum(r.get('type_correct', False) for r in rows) / n if n else None}


def cluster_interval(rows, field='score', samples=2000):
    groups = defaultdict(list)
    for r in rows: groups[r['business_id']].append(r[field])
    keys = list(groups)
    if not keys: return [None, None]
    rng = random.Random(0); values = []
    for _ in range(samples):
        chosen = [v for b in rng.choices(keys, k=len(keys)) for v in groups[b]]
        values.append(sum(chosen) / len(chosen))
    values.sort(); return [values[int(.025 * samples)], values[min(samples - 1, int(.975 * samples))]]


def evaluate(gold, pred, questions):
    unknown = set(pred) - set(gold)
    if unknown: raise ValueError(f'{len(unknown)} predictions have unknown IDs')
    rows = []
    for id, g in gold.items():
        p = pred.get(id, {'type': 'abstain'}); q = questions.get(id, {})
        rows.append({'id': id, 'business_id': q.get('business_id', 'unknown'), 'score': score(p, g), 'type_correct': p.get('type') == g['type'], 'gold_type': g['type'], 'pred_type': p.get('type')})
    report = {**metrics(rows), 'missing_predictions': len(set(gold) - set(pred)), 'PA_cluster_bootstrap_95': cluster_interval(rows), 'scorer': 'published-rule reproduction; not official code', 'breakdowns': {}}
    for axis in ('business_id', 'gold_type', 'pred_type'):
        groups = defaultdict(list)
        for r in rows: groups[r[axis]].append(r)
        report['breakdowns'][axis] = {str(k): metrics(v) for k, v in groups.items()}
    report['semantic_correctness'] = 'Not inferred from successful execution. Requires manual plan audit, plus reference-answer correctness.'
    return report, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True); p.add_argument('--pred', type=Path, required=True); p.add_argument('--compare', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--businesses', nargs='+')
    p.add_argument('--questions', type=Path, help='Explicit benchmark questions JSONL; denominator is exactly this set')
    a = p.parse_args()
    questions = read_jsonl(a.data / 'questions.jsonl'); a.output.parent.mkdir(parents=True, exist_ok=True)
    gold = read_jsonl(a.data / 'answers.jsonl'); pred = read_jsonl(a.pred)
    if a.questions:
        selected = read_jsonl(a.questions)
        if not selected or set(selected) - set(questions): p.error('question subset must be nonempty and contain only known IDs')
        if any(selected[k] != questions[k] for k in selected): p.error('subset question records differ from the source')
        missing_gold = set(selected) - set(gold)
        if missing_gold: p.error(f'{len(missing_gold)} selected questions have no reference answers')
        gold = {k: gold[k] for k in selected}; pred = {k: v for k, v in pred.items() if k in selected}
    if a.businesses:
        ids = {k for k, q in questions.items() if q['business_id'] in a.businesses}; gold = {k: v for k, v in gold.items() if k in ids}; pred = {k: v for k, v in pred.items() if k in ids}
    if not gold: p.error('selected evaluation question set is empty after applying filters')
    report, rows = evaluate(gold, pred, questions)
    if a.compare:
        old = read_jsonl(a.compare); deltas = [{'business_id': r['business_id'], 'score': r['score'] - score(old.get(r['id']), gold[r['id']])} for r in rows]
        report['paired_comparison'] = {'delta_PA': sum(r['score'] for r in deltas) / len(deltas), 'cluster_bootstrap_95': cluster_interval(deltas), 'prediction_disagreements': sum(pred.get(id) != old.get(id) for id in gold), 'warning': 'Use identical question sets/checkpoint/seed/environment. This command cannot establish matching runtime metadata.'}
    a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)); print(json.dumps({k: v for k, v in report.items() if k != 'breakdowns'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
