"""See what your Part 1 pipelines produce on real data (course-provided).

    uv run python -m cs329z_hw1.show priority em-12167 [em-...]   # label one or more emails
    uv run python -m cs329z_hw1.show digest 2001-06-22 [--labels] [--limit N]  # the digest for one day
    uv run python -m cs329z_hw1.show email_qa "Who leads the Basin Analytics move?"
    uv run python -m cs329z_hw1.show search_docs "parental leave" [--k 5]

Each command calls your adapters the way the tests do, on the grading model,
and prints the result with the cost and time of the calls. Repeating a
command with unchanged code is free, because the LM caches every reply.

``digest --labels`` also prints the label your run_priority gives each email
of the day, so you can see which pile each one landed in. Those calls are
the ones your digest already made, so they come from the cache. A day of
the archive holds about 280 emails (the live tests use about 30 per day);
``--limit N`` runs the digest on the first N emails of the day.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs329z_hw1 import data  # noqa: E402
from cs329z_hw1.llm import LM  # noqa: E402


def _email_header(email: dict) -> str:
    return (
        f"{email['id']}  {email['date'][:16]}  from {email.get('from', '')}\n"
        f"  subject: {email.get('subject') or '(no subject)'}"
    )


def _finish(lm: LM, started: float) -> None:
    u = lm.usage()
    cached = f", {u['cached_calls']} from the cache" if u["cached_calls"] else ""
    print(f"\n[{u['calls']} model calls{cached}; ${u['cost_usd']:.4f}; {time.time() - started:.1f} s]")


def show_priority(args) -> None:
    from cs329z_hw1 import adapters

    by_id = {e["id"]: e for e in data.load_emails()}
    lm = LM(tag="show/priority")
    started = time.time()
    for email_id in args.ids:
        email = by_id.get(email_id)
        if email is None:
            print(f"{email_id}: no such email")
            continue
        result = adapters.run_priority(email, lm)
        print(_email_header(email))
        body = " ".join(email["body"].split())
        print(f"  body: {body[:300]}{'...' if len(body) > 300 else ''}")
        print(f"  -> {result['category']}: {result['reason']}\n")
    _finish(lm, started)


def show_digest(args) -> None:
    from cs329z_hw1 import adapters

    emails = data.emails_on(args.date)
    if not emails:
        sys.exit(f"No emails on {args.date}. The archive covers 2001-06-01 to 2001-08-31.")
    total = len(emails)
    if args.limit:
        emails = emails[: args.limit]
    print(f"Running your digest on {len(emails)} of the {total} emails of {args.date}...", flush=True)
    lm = LM(tag="show/digest")
    started = time.time()
    digest = adapters.run_daily_digest(emails, lm)
    print(f"\nDigest for {args.date} ({len(emails)} emails, {len(digest.split())} words):\n")
    print(digest)
    if args.labels:
        print(f"\nLabels from your run_priority:")
        for email in emails:
            result = adapters.run_priority(email, lm)
            print(f"  {result['category']:<8} {email['id']}  {email.get('subject') or '(no subject)'}")
    _finish(lm, started)


def show_email_qa(args) -> None:
    from cs329z_hw1 import adapters

    archive = data.load_emails()
    lm = LM(tag="show/email_qa")
    started = time.time()
    index = adapters.run_bm25_build([data.email_text(e) for e in archive])
    returned: dict[str, dict] = {}

    def search(query: str, k: int = 5) -> list[dict]:
        hits = [archive[doc_id] for doc_id, _ in adapters.run_bm25_search(index, query, k)]
        print(f"  search({query!r}, {k}) -> {[e['id'] for e in hits]}")
        returned.update((e["id"], e) for e in hits)
        return hits

    print(f"Question: {args.question}\n")
    result = adapters.run_email_qa(args.question, search, lm)
    print(f"\nAnswer: {result['answer']}")
    print("Support:")
    for email_id in result["support"]:
        email = returned.get(email_id)
        print(f"  {_email_header(email) if email else email_id + '  (not returned by search)'}")
    _finish(lm, started)


def show_search_docs(args) -> None:
    from cs329z_hw1 import adapters

    index = adapters.run_build_doc_index(data.load_docs())
    results = adapters.run_search_docs(index, args.query, args.k)
    print(f"search_docs({args.query!r}, {args.k}) -> {len(results)} results\n")
    for r in results:
        print(f"{r['doc_id']}  ({r['title']})\n  {r['snippet']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one of your Part 1 pipelines and print what it produces.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("priority", help="label one or more emails by id")
    p.add_argument("ids", nargs="+")
    p.set_defaults(fn=show_priority)
    p = sub.add_parser("digest", help="the digest for one day, YYYY-MM-DD")
    p.add_argument("date")
    p.add_argument("--labels", action="store_true", help="also print each email's priority label")
    p.add_argument("--limit", type=int, default=None, metavar="N", help="use only the first N emails of the day")
    p.set_defaults(fn=show_digest)
    p = sub.add_parser("email_qa", help="answer a question over the archive")
    p.add_argument("question")
    p.set_defaults(fn=show_email_qa)
    p = sub.add_parser("search_docs", help="search the company documents")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=5)
    p.set_defaults(fn=show_search_docs)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
