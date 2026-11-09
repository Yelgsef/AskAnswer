"""The inference pipeline, stage by stage.

    questions -> shops (loaded into DuckDB while the model starts)
              -> per shop, once: canonical sales views (views.py) and a debt layer (debts.py), drafted by the model and tested
                 on the data, or reused from a saved run (--views-from)
              -> per question: K plain SQL samples over two prompts, executed, failed ones repaired. Thinking samples for
                 low-agreement and debt questions. Majority vote, answer or abstain (voting.py)

Pipeline holds the run's settings and state. Each stage is one method, so a stage can run without the model (setup_shop and
build_chats only read the data, which is how tests/replays rebuild the exact prompts of a saved run).
"""
import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import debts, fewshot, views
from .hints import DEBT_WORDS, data_span, outside, question_hints
from .ingest import entity_hits, load_business, norm, profile, sample_text, schema_text
from .prompts import NOTES, SYSTEM, SYSTEM_ALT, chat, repair_hint
from .sqlexec import execute, parse
from .submission import merge_shards, write_submission
from .voting import debt_pool, final_answer, top_votes, vote


def load_questions(a):
    """The questions of this run: --ids / --limit filters, then this process's share of the shops (--shard i/n)."""
    qs = [json.loads(l) for l in (a.data / "questions.jsonl").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    if a.ids:
        keep = set(re.findall(r"q_[0-9a-f]+", a.ids.read_text()))
        qs = [q for q in qs if q["id"] in keep]
    qs = qs[:a.limit] if a.limit else qs
    i, n = map(int, a.shard.split("/"))
    load = Counter(q["business_id"] for q in qs)
    bins = [[] for _ in range(n)]
    for b in sorted(load, key=lambda b: (-load[b], b)):  # greedy balance of question counts over shards
        min(bins, key=lambda x: sum(load[y] for y in x)).append(b)
    shops = sorted(bins[i])
    return [q for q in qs if q["business_id"] in shops], shops


@dataclass
class Shop:
    """One shop ready for questions: its database (views and debt macros built in), what the prompt shows of it."""
    name: str
    db: object
    meta: dict
    shown: dict
    view: str = ""
    debt: str = ""
    ctx: str = ""
    vinfo: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, a, engine=None):
        self.a, self.engine = a, engine
        self.rulebook = (a.data / "rulebook.md").read_text(encoding="utf-8-sig")
        base_prompts = (SYSTEM, SYSTEM) if a.fewshot else (SYSTEM, SYSTEM_ALT)
        self.systems = [s + "\n\nRULEBOOK:\n" + self.rulebook for s in base_prompts]
        self.bank = fewshot.Bank(a.fewshot) if a.fewshot else None
        self.half = a.k // 2
        self.vtexts, self.vchecked, self.vfixed = {}, {}, {}
        self.dchecked, self.dfixed = {}, {}
        # one log record per question: --log-dir (Docker: /result/logs), else the run's own logs/
        self.log_dir = getattr(a, "log_dir", None) or (a.output / "logs" if getattr(a, "output", None) else None)
        self._calls, self._runs = [], []

    # ---- shops -------------------------------------------------------------------------------------------------------
    def load_shops(self, shops):
        loaded = {}
        for b in shops:
            try:
                db, meta = load_business(self.a.data, b, canon=self.a.canon_names)
                db.execute("SET enable_external_access = false")  # model SQL cannot read files or the network
                db.execute("SET memory_limit = '6GB'")  # a runaway model join fails (-> repair) instead of exhausting host RAM
                loaded[b] = db, meta
            except Exception as e:
                print(f"[load error] {b}: {e}", flush=True)
        return loaded

    def reuse_layers(self, loaded):
        """Views and debt layers an earlier run chose (its logs/_views_<shop>.json, _debt_<shop>.json)."""
        a = self.a
        if a.views == "llm" and a.views_from:  # same views as that run, so an A/B run differs only in what it changes
            self.vfixed = {b: json.loads(f.read_text(encoding="utf-8")) for b in loaded if (f := a.views_from / "logs" / f"_views_{b}.json").exists()}
            print(f"views of {len(self.vfixed)}/{len(loaded)} shops reused from {a.views_from}", flush=True)
        if a.debt_views and a.views_from:
            self.dfixed = {b: json.loads(f.read_text(encoding="utf-8")) for b in loaded if (f := a.views_from / "logs" / f"_debt_{b}.json").exists()}

    def draft_views(self, loaded, t0):
        """Per shop: a.k drafts of one mapping SELECT, data tests, up to 2 repairs. A second set when there is no majority."""
        a, eng = self.a, self.engine
        drafted = [b for b in loaded if b not in self.vfixed]  # shops without reused views get fresh drafts
        if a.views != "llm" or not drafted:
            return
        samples = {b: sample_text(loaded[b][0], loaded[b][1], max_rows=a.view_sample_rows)
                   for b in drafted} if getattr(a, "view_sample_rows", 0) > 0 else {}

        def view_round(shops, seed):  # all shops in one batch (GPU stays full)
            vchats = [[{"role": "system", "content": views.VIEW_SYSTEM + "\n\nRULEBOOK:\n" + self.rulebook},
                       {"role": "user", "content": "README:\n" + loaded[b][1]["readme"] + "\n\nSCHEMA:\n" + schema_text(loaded[b][1])
                           + ("\n\nREPRESENTATIVE ROWS:\n" + samples[b] if samples.get(b) else "")}] for b in shops]
            vsp = eng.params(n=a.k, temperature=0.6 if a.view_think else a.temperature, top_p=0.95, top_k=20,
                             max_tokens=a.view_think + 2048, seed=seed)
            vouts = eng.chat(vchats, vsp, thinking=bool(a.view_think))
            texts = {b: [x.text for x in o.outputs] for b, o in zip(shops, vouts)}
            checked = {b: views.check(loaded[b][0], ts) for b, ts in texts.items()}
            # drafts failing the data tests get up to 2 repairs, each with the latest error and the turns so far, all shops in one batch
            vchat_of = dict(zip(shops, vchats))
            convo = {(b, j): vchat_of[b] + [{"role": "assistant", "content": texts[b][j].split("</think>")[-1].strip()}] for b in shops for j in range(len(texts[b]))}
            n_fix = 0
            for _ in range(2):
                fail = [(b, j) for b, c in checked.items() for j, (fp, _) in enumerate(c) if fp is None]
                if not fail:
                    break
                n_fix += len(fail)
                rch = [convo[b, j] + [{"role": "user", "content": f"ERROR: {checked[b][j][1]}\nFix the SELECT. Answer with the SQL only, in one ```sql``` block."}] for b, j in fail]
                routs = eng.chat(rch, eng.params(n=1, temperature=0, max_tokens=2048))
                for (b, j), c, o in zip(fail, rch, routs):
                    texts[b][j] += "\n#REPAIR\n" + o.outputs[0].text
                    convo[b, j] = c + [{"role": "assistant", "content": o.outputs[0].text}]
                    checked[b][j] = views.check(loaded[b][0], [o.outputs[0].text])[0]
            print(f"view drafts in {time.time() - t0:.0f}s ({n_fix} repairs)", flush=True)
            return texts, checked
        self.vtexts, self.vchecked = view_round(drafted, a.seed)
        # no majority: a second set of drafts and one vote over both (holdout seeds 0-4: a pick without a majority moved with the seed)
        if weak := [b for b in drafted if 2 * top_votes(self.vchecked[b]) < a.k]:
            t2, c2 = view_round(weak, a.seed + 1)
            for b in weak:
                self.vtexts[b] += t2[b]
                self.vchecked[b] += c2[b]

    def draft_debts(self, loaded, t0):
        """Same recipe for the debt layer: a.k drafts per shop, data tests, one repair turn, most common fingerprint."""
        a, eng = self.a, self.engine
        ddraft = [b for b in loaded if b not in self.dfixed]
        if not a.debt_views or not ddraft:
            return
        dround = lambda shops, seed: debts.draft(lambda cs, sp, th: eng.chat(cs, sp, thinking=th), eng.SamplingParams,
                                                 {b: loaded[b] for b in shops}, self.rulebook, schema_text, k=a.k, think=a.view_think, temperature=0.6, seed=seed)
        self.dchecked = {b: r[1] for b, r in dround(ddraft, a.seed).items()}
        if weak := [b for b in ddraft if 2 * top_votes(self.dchecked[b]) < a.k]:  # no majority (NONE counts): second set, one vote over both
            for b, r in dround(weak, a.seed + 1).items():
                self.dchecked[b] += r[1]
        print(f"debt drafts in {time.time() - t0:.0f}s", flush=True)

    def setup_shop(self, b, db, meta):
        """Pick and build the shop's views and debt layer, write their logs, and compose the shop part of the prompt."""
        a, vfixed, vtexts, vchecked = self.a, self.vfixed, self.vtexts, self.vchecked
        view, vinfo = "", {}
        if b in vfixed:
            sql, nv, verr = vfixed[b]["sql"], vfixed[b]["agree"], vfixed[b]["errors"]
            vtexts[b] = vfixed[b]["drafts"]
            if sql:
                views.build(db, sql)
        elif b in vtexts:
            sql, nv = views.pick(db, vchecked[b])
            verr = [e for fp, e in vchecked[b] if fp is None]
            if not sql or nv * 4 < a.k:  # too few drafts pass the catalog test: the most common one that only fails that test
                soft, sn = views.pick_soft(db, vtexts[b])
                if sn > nv:
                    sql, nv = soft, sn
                elif sql:
                    views.build(db, sql)
            if sql and len(bad := views.mixed_price_items(db, "v_sales")) >= 2:  # qty is still not in one base unit
                fixed = views.qty_by_price(sql)
                views.build(db, fixed)
                left = views.mixed_price_items(db, "v_sales")
                if len(left) < len(bad):  # factors read from the prices only help when unit holds the line's own unit
                    sql = fixed
                else:
                    views.build(db, sql)
                print(f"[views] {b}: price per qty differs over 3x on {len(bad)} items, factors read from the prices -> {len(left)} left, kept={sql == fixed}", flush=True)
        if b in vtexts:
            # drafts behind the pick: a saved layer keeps no drafts, so its own count (prompts of the 0.683 run said "7/0" there)
            nk = len(vtexts[b]) or (vfixed[b].get("k") if b in vfixed else 0) or a.k
            if sql and nv * 4 >= a.k:  # too little agreement = the mapping is a guess. Better no view than a wrong one (counted against one set of a.k drafts)
                view = views.describe(db, sql, nv, nk)
                meta["tables"] += [profile(db, v, "canonical view") for v in ("v_sales", "v_orders")]
            vinfo = {"agree": nv, "k": nk, "used": bool(view), "sql": sql, "errors": verr, "drafts": vtexts[b]}
            print(f"[views] {b}: {nv}/{nk} agree, used={bool(view)}, {len(verr)} failed drafts", flush=True)
        debt = ""
        if a.debt_views:
            dfixed, dchecked = self.dfixed, self.dchecked
            sqls, dn = (dfixed[b]["sqls"], dfixed[b]["agree"]) if b in dfixed else debts.pick(dchecked[b])
            dfps = dfixed[b].get("checked", []) if b in dfixed else [str(fp) for fp, _ in dchecked.get(b, [])]
            dk = dfixed[b].get("k", a.k) if b in dfixed else len(dchecked[b])  # a reused file keeps its own verdict and draft count: re-judging it at another k, or from a reuse that lost "checked", dropped a working layer
            if sqls and (dfixed[b]["used"] if b in dfixed else debts.accept(dfps, dn, a.k)):
                try:
                    debts.build(db, sqls)  # the tables stay out of SCHEMA: the prompt offers the macros and their recipes
                    debt = debts.describe(dn, dk)
                except ValueError as e:  # a reused layer that fails a data test added after it was saved: no layer rather than a crash
                    print(f"[debts] {b}: reused layer rejected: {str(e)[:200]}", flush=True)
            if a.output:
                (a.output / "logs" / f"_debt_{b}.json").write_text(json.dumps({"agree": dn, "k": dk, "used": bool(debt), "sqls": sqls, "checked": dfps,
                                                                             "errors": [e for fp, e in dchecked.get(b, []) if fp is None]}, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[debts] {b}: {dn}/{dk} agree, used={bool(debt)}", flush=True)
        shown, raw_note = meta, ""
        if a.hide_raw and view:  # the views become the interface. The mapping SQL would only point back at the raw tables
            shown, raw_note = views.hide_raw(db, meta, sql)
            view = view.split("\nBuilt from the raw tables")[0]
        ctx = "README:\n" + meta["readme"] + "\n\nSCHEMA:\n" + schema_text(shown) + raw_note + ("\n\nCANONICAL VIEWS:\n" + view if view else "") + ("\n\n" + debt if debt else "")
        return Shop(b, db, meta, shown, view, debt, ctx, vinfo)

    def build_chats(self, shop, todo):
        """Per question: values matching it, its hints, whether its period lies outside the data. Then one chat per prompt."""
        a, bank = self.a, self.bank
        for q in todo:
            q["_hits"] = entity_hits(shop.db, shop.shown, q["question"])
            q["_periods"] = question_hints(q["question"], q["as_of_date"], shop.db, shop.shown)
            q["_oor"] = outside(q["_periods"], data_span(shop.meta))
        # examples read the canonical views, so a shop without them gets none
        return [[chat(s, shop.ctx, q, fewshot.block(bank.retrieve(q["question"], shop.name, a.fewshot_k, debt=bool(shop.debt))) if bank and shop.view and j == 1 else "")
                 for j, s in enumerate(self.systems)] for q in todo]

    # ---- sampling ----------------------------------------------------------------------------------------------------
    def _call(self, i, kind, j, text, thinking_mode=False, lora=True):
        """Log one model generation for question i (call order kept for the per-question log)."""
        self._calls[i].append({"kind": kind, "sample": j, "thinking_mode": thinking_mode, "lora": bool(lora and self.a.lora), "text": text})

    def _exec(self, shop, i, j, text, question):
        """Execute sample j of question i and log the SQL as run, its output or its error."""
        tr = {}
        try:
            p = execute(shop.db, text, question, trace=tr)
        except Exception as e:
            self._runs[i].append({"sample": j, "sql": tr.get("sql"), "executed": "sql" in tr, "output": None, "error": str(e)[:400]})
            raise
        self._runs[i].append({"sample": j, "sql": tr.get("sql"), "executed": "sql" in tr, "output": p, "error": None})
        return p

    def _repair(self, shop, todo, chats, texts, preds, errs, pairs, prompt_of):
        """Greedy repair turns for failed samples (i, j): the error plus a hint for its kind, up to --repairs turns."""
        a, eng = self.a, self.engine
        conv = {(i, j): chats[i][prompt_of(j)] + [{"role": "assistant", "content": texts[i][j].split("</think>")[-1].strip()}]
                for i, j in pairs}
        turns = 0
        for _ in range(a.repairs):
            pairs = [(i, j) for i, j in pairs if preds[i][j] is None]
            if not pairs:
                break
            for i, j in pairs:
                e = errs[i][j].split(" | ")[-1]
                conv[i, j].append({"role": "user", "content": f"ERROR: {e}\n{repair_hint(e)}Fix the query. Answer again in the same format."})
            routs = eng.chat([conv[ij] for ij in pairs], eng.params(n=1, temperature=0, max_tokens=a.max_tokens), lora=True)
            for (i, j), o in zip(pairs, routs):
                out = o.outputs[0].text
                self._call(i, "repair", j, out)
                texts[i][j] += "\n#REPAIR\n" + out
                conv[i, j].append({"role": "assistant", "content": out})
                try:
                    preds[i][j] = self._exec(shop, i, j, out, todo[i]["question"])
                except Exception as e:
                    errs[i][j] += " | " + str(e)[:400]
            turns += len(pairs)
        return turns

    def sample(self, shop, todo, chats, is_behind):
        """Plain samples (half per prompt), executed. TYPE fills and repairs. Thinking samples for hard and debt questions.
        is_behind() is asked after the plain samples: a run behind its --deadline pace skips the thinking samples."""
        a, eng, half, systems, bank = self.a, self.engine, self.half, self.systems, self.bank
        budget = a.context - a.max_tokens - 64
        fits = [max(eng.n_tokens(c) for c in cs) <= budget for cs in chats]
        sp = eng.params(n=half, temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens, seed=a.seed)
        batch = [c for cs, f in zip(chats, fits) if f for c in cs]
        if a.notes:
            batch = [c + [{"role": "assistant", "content": NOTES}] for c in batch]
        outs = eng.chat(batch, sp, lora=True, prefill=a.notes)
        it = iter(outs)
        texts = [[(NOTES if a.notes else "") + o.text for _ in systems for o in next(it).outputs] if f else [] for f in fits]
        self._calls, self._runs = [[] for _ in todo], [[] for _ in todo]  # per question: model calls and SQL runs, in order
        for i, ts in enumerate(texts):
            for j, t in enumerate(ts):
                self._call(i, "sample", j, t)

        # Execute. Failed samples get greedy repair turns with the error message.
        preds = [[None] * len(t) for t in texts]
        errs, repair = [[None] * len(t) for t in texts], []
        for i, ts in enumerate(texts):
            for j, t in enumerate(ts):
                try:
                    preds[i][j] = self._exec(shop, i, j, t, todo[i]["question"])
                except Exception as e:
                    errs[i][j] = str(e)[:400]
                    repair.append((i, j))
        # A sample with SQL but no TYPE line gets the model's own type: prefill "TYPE:" and read <=4 tokens.
        notype = [(i, j) for i, j in repair if errs[i][j].startswith("missing or invalid TYPE") and parse(texts[i][j])[1]]
        if notype:
            tchats = [chats[i][j // half] + [{"role": "assistant", "content": texts[i][j].rstrip() + "\nTYPE:"}] for i, j in notype]
            touts = eng.chat(tchats, eng.params(n=1, temperature=0, max_tokens=4), lora=True, prefill=True)
            for (i, j), o in zip(notype, touts):
                self._call(i, "type_fill", j, o.outputs[0].text)
                texts[i][j] = texts[i][j].rstrip() + "\nTYPE:" + o.outputs[0].text
                try:
                    preds[i][j], errs[i][j] = self._exec(shop, i, j, texts[i][j], todo[i]["question"]), None
                    repair.remove((i, j))
                except Exception as e:
                    errs[i][j] = str(e)[:400]

        prompt_idx = lambda j: j // half if j < 2 * half else (1 if bank else (j - 2 * half) % len(systems))  # which prompt asked sample j
        n_repair = self._repair(shop, todo, chats, texts, preds, errs, repair, lambda j: j // half)
        behind = is_behind()
        debt_q = [bool(shop.debt) and a.debt_route and bool(DEBT_WORDS.search(norm(q["question"]))) for q in todo]
        if a.think_k and not behind:  # adaptive compute: low-agreement questions, and debt questions (thinking samples are the ones calling the macros), get thinking-mode samples
            hard = [i for i, ps in enumerate(preds) if texts[i] and (debt_q[i] or vote(ps, len(ps))[1] < a.think_below)]
            if hard:
                # with examples available all thinking samples get them: right 86% vs 71.5% on 873 hard train questions
                hchats = [chats[i][1 if bank else s % len(systems)] for i in hard for s in range(a.think_k)]
                houts = eng.chat(hchats, eng.params(n=1, temperature=0.6, top_p=0.95, top_k=20, max_tokens=a.think_budget + a.max_tokens, seed=a.seed),
                                 thinking=True, lora=a.lora_thinking)
                for n, o in enumerate(houts):
                    i, t = hard[n // a.think_k], o.outputs[0].text
                    if "<think>" not in t:  # Qwen3.5's template opens <think> in the prompt and parse() keys on it to spot a cut-off thought
                        t = "<think>\n" + t
                    texts[i].append(t)
                    self._call(i, "thinking_sample", len(texts[i]) - 1, t, thinking_mode=True, lora=a.lora_thinking)
                    try:
                        preds[i].append(self._exec(shop, i, len(texts[i]) - 1, t, todo[i]["question"])); errs[i].append(None)
                    except Exception as e:
                        preds[i].append(None); errs[i].append(str(e)[:400])
                # failed thinking samples get the same repair turns (non-thinking). A cut-off thought has no query to fix
                n_repair += self._repair(shop, todo, chats, texts, preds, errs,
                                         [(i, j) for i in hard for j in range(2 * half, len(texts[i])) if preds[i][j] is None and "</think>" in texts[i][j]], prompt_idx)
        pools = [debt_pool(ts, 2 * half, dq) if a.debt_route else list(range(len(ts))) for ts, dq in zip(texts, debt_q)]
        return texts, preds, errs, pools, len(repair), n_repair, behind

    def decide(self, todo, preds, pools):
        """(final, vote, confidence) per question."""
        out = []
        for q, ps, pool in zip(todo, preds, pools):
            voters = [ps[j] for j in pool]
            ans, conf = vote(voters, len(voters))
            out.append((final_answer(ans, conf, self.a.threshold, q["_oor"], self.a.answer_unanswerable), ans, conf))
        return out

    # ---- the run -----------------------------------------------------------------------------------------------------
    def run(self, qs, shops, loaded, t0):
        a = self.a
        by_shop = defaultdict(list)
        for q in qs:
            by_shop[q["business_id"]].append(q)
        self.reuse_layers(loaded)
        self.draft_views(loaded, t0)
        self.draft_debts(loaded, t0)
        results, n_repair = {}, 0
        tr = (a.output / "trace.jsonl").open("w", encoding="utf-8")
        for b, (db, meta) in loaded.items():  # one shop at a time: answer all its questions, write results (checkpoint), close
            tb = time.time()
            shop = self.setup_shop(b, db, meta)
            todo = by_shop[b]
            chats = self.build_chats(shop, todo)
            tg = time.time()
            is_behind = lambda: a.deadline and time.time() - t0 > a.deadline * (len(results) + len(todo)) / max(1, len(qs))
            texts, preds, errs, pools, n_first, n_rep, behind = self.sample(shop, todo, chats, is_behind)
            n_repair += n_rep
            decisions = self.decide(todo, preds, pools)
            t_ans = time.time()
            # Batched generation: every question of the shop enters the batch when the shop starts (its data reads, views and debt
            # layer included) and gets its answer when the batch ends, so they share one elapsed time. The per-shop view and debt
            # drafts made before the loop are shared preparation, like loading the model.
            elapsed_ms = int(round((t_ans - tb) * 1000))
            per_q = (t_ans - tb) / max(1, len(todo))
            layers = {"views": str(a.output / "logs" / f"_views_{b}.json"), **({"debt": str(a.output / "logs" / f"_debt_{b}.json")} if a.debt_views else {})}
            for n, (q, (r, ans, conf), ts, ps, es, cs, pool) in enumerate(zip(todo, decisions, texts, preds, errs, chats, pools)):
                results[q["id"]] = r
                rec = {"id": q["id"], "business_id": b, "question": q["question"], "final": r, "vote": ans, "conf": conf,
                       "elapsed_seconds": round(per_q, 2), "samples": [{"text": t, "pred": pr, "error": e} for t, pr, e in zip(ts, ps, es)]}
                if a.save_prompts:  # the system message is SYSTEM / SYSTEM_ALT + rulebook (samples 0..K/2-1 / the rest), the user message is shared
                    rec["prompt_user"] = cs[0][1]["content"]
                tr.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                log = question_log(q, r, elapsed_ms, self._calls[n], self._runs[n],
                                   {"code": "src/datamate/voting.py::vote, final_answer",
                                    "inputs": {"answers": [ps[j] for j in pool], "samples": pool, "threshold": a.threshold,
                                               "out_of_range": q["_oor"], "answer_unanswerable": a.answer_unanswerable},
                                    "vote": ans, "share": conf}, layers)
                (self.log_dir / f"{q['id']}.json").write_text(json.dumps(log, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
            tr.flush()
            (a.output / "logs" / f"_views_{b}.json").write_text(json.dumps(shop.vinfo, ensure_ascii=False, indent=1), encoding="utf-8")
            write_submission(a.output / "submission.jsonl", qs, results)
            if getattr(a, "merge_into", None):  # the shared /result/submission.jsonl holds every answer saved so far
                merge_shards(a.merge_into, a.data / "questions.jsonl", shards=a.output.parent, copy_logs=False, quiet=True)
            db.close()
            print(f"[{b}] {len(todo)} q, {sum(r['type'] != 'abstain' for r in (results[q['id']] for q in todo))} answered, "
                  f"{n_first} repairs, gen {time.time() - tg:.0f}s, total {time.time() - tb:.0f}s{', behind pace: no thinking' if behind else ''}", flush=True)
        tr.close()
        return results, n_repair


def reasoning(call):
    """The model's reasoning in one generation, as generated: the text inside <think>…</think> (to the end when the thought was cut
    off), else the notes a plain sample writes before its TYPE line and SQL. A TYPE fill (a few tokens of the type) has none."""
    text = call["text"]
    if call["kind"] == "type_fill":
        return ""
    if "<think>" in text:
        return text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    cut = min([k for k in (text.find("TYPE:"), text.find("```")) if k >= 0], default=len(text))
    return text[:cut].strip()


def question_log(q, answer, elapsed_ms, calls, runs, decision, layers):
    """One log record per question (organizers' Docker guide, section 3.3): id, answer, elapsed_ms, thinking (one string per
    model call, in call order; null when no call reasoned), code and output (every SQL run against the database, in order),
    then the full generations, every execution attempt, the vote and the shop layers the SQL reads (views, debt macros)."""
    thinking = [reasoning(c) for c in calls]
    ran = [x for x in runs if x["executed"]]
    return {"id": q["id"], "business_id": q["business_id"], "question": q["question"], "as_of_date": q.get("as_of_date"),
            "answer": answer, "elapsed_ms": elapsed_ms,
            "thinking": thinking if any(thinking) else None,
            "code": [x["sql"] for x in ran] or None,
            "output": [x["output"] if x["error"] is None else {"error": x["error"]} for x in ran] or None,
            "calls": calls, "executions": runs, "decision": decision, "shop_layers": layers}


def run_in_background(fn, *args):
    """Start fn(*args) in a thread (shop loading overlaps the model start). Returns a future."""
    return ThreadPoolExecutor(1).submit(fn, *args)
