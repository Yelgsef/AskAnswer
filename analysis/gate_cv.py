"""Answer-selection variants replayed offline on holdout traces (train gold only, no LLM).
Usage: python analysis/gate_cv.py RUN
Each variant re-votes the trace's samples with predict.vote / debt_pool and a decision rule. The threshold is also picked
leave-one-shop-out (fit on 5 shops, score the 6th), so the reported CV PA is an honest held-out number."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from datamate import hints, ingest, voting  # noqa: E402
from err import GOLD, load  # noqa: E402
from datamate.scoring import score  # noqa: E402

TAUS = [0.25, 0.3, 0.375, 0.4, 0.45, 0.5, 0.55, 0.6, 0.625, 0.65, 0.7, 0.75, 0.8]


def debt_shops(run):
    """Shops whose debt layer the run used (predict.main offers the macros only there)."""
    return {f.stem.removeprefix("_debt_") for f in Path(run).glob("**/logs/_debt_*.json") if json.loads(f.read_text(encoding="utf-8")).get("used")}


def candidates(r, debt=frozenset(), route=True):
    """(vote, conf) over the voting pool of one traced question, as predict.main computes it."""
    texts = [s["text"] for s in r["samples"]]
    preds = [s["pred"] for s in r["samples"]]
    dq = route and r["business_id"] in debt and bool(hints.DEBT_WORDS.search(ingest.norm(r["question"])))
    pool = voting.debt_pool(texts, 8, dq) if texts else []
    voters = [preds[j] for j in pool]
    return voting.vote(voters, len(voters)) if voters else ({"type": "abstain"}, 0.0)


def final(r, ans, conf, tau, decide=voting.decide):
    if r["final"].get("reason") == "period outside the shop's data dates":
        return r["final"]
    if ans.get("type") == "unanswerable" and conf >= tau:
        return ans
    return ans if decide(ans, conf, tau) else {"type": "abstain"}


def pa(rows, tau, decide):
    return sum(score(final(r, a, c, tau, decide), GOLD[r["id"]]) for r, a, c in rows)


def main():
    rs = load(sys.argv[1])
    debt = debt_shops(sys.argv[1])
    rows = [(r, *candidates(r, debt, "--old-route" not in sys.argv)) for r in rs.values() if r["id"] in GOLD]
    n = len(rows)
    same = sum(final(r, a, c, 0.5) == r["final"] for r, a, c in rows)
    print(f"replay reproduces {same}/{n} finals at tau 0.5")
    variants = {"current": voting.decide}
    if hasattr(voting, "decide_old"):
        variants["old"] = voting.decide_old
    shops = sorted({r["business_id"] for r, _, _ in rows})
    for name, dec in variants.items():
        print(f"[{name}] fixed tau: " + " ".join(f"{t}:{pa(rows, t, dec) / n:.3f}" for t in TAUS))
        tot, picked = 0, []
        for b in shops:
            fit = [x for x in rows if x[0]["business_id"] != b]
            test = [x for x in rows if x[0]["business_id"] == b]
            t = max(TAUS, key=lambda t: (pa(fit, t, dec), -abs(t - 0.5)))
            picked.append(t)
            tot += pa(test, t, dec)
        print(f"[{name}] leave-one-shop-out PA {tot / n:.4f}, taus {picked}")
        c = Counter(score(final(r, a, cf, 0.5, dec), GOLD[r["id"]]) for r, a, cf in rows)
        print(f"[{name}] tau .5: correct {c[1]} wrong {c[-1]} abstain {c[0]}")


if __name__ == "__main__":
    main()
