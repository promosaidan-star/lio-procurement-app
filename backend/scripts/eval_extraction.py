"""Run the five customer quotes through the real extraction pipeline and score them.

Usage (needs OPENAI_API_KEY in backend/.env or the environment):

    cd backend && python scripts/eval_extraction.py [--json out.json]

Uses an in-memory SQLite database seeded exactly like production (same code path
as the test suite), so no Postgres or Docker is required. Scores each quote
against tests/fixtures/quotes/expected.json and prints a table; the exit code is
the number of failed checks, so it can gate CI once a key is available there.
"""
from __future__ import annotations  # modern type hints on 3.11

import argparse  # --json flag
import json  # read expected.json, write raw results
import sys  # sys.path tweak and exit code
import time  # per-quote timing
from pathlib import Path  # file paths

ROOT = Path(__file__).resolve().parents[1]  # backend/ directory
sys.path.insert(0, str(ROOT))  # make "import app" work when run as a script

from sqlalchemy import create_engine  # noqa: E402  (imports after the path tweak on purpose)
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402  one shared in-memory connection

from app.core.config import settings  # noqa: E402  reads OPENAI_API_KEY from .env
from app.db.base import Base  # noqa: E402  table metadata
from app.seed import seed  # noqa: E402  same seed production runs
from app.services.extraction import extract_vendor_data  # noqa: E402  the pipeline under test
from app.services.pdf import extract_text  # noqa: E402  the parser under test

FIXTURES = ROOT / "tests" / "fixtures" / "quotes"  # the five PDFs + expected.json


def _money_ok(got: float, want: float) -> bool:
    return abs(got - want) <= 0.01  # to the cent


def _name_ok(got: str, want: str) -> bool:
    return want.lower() in got.lower() if want else got.strip() == ""  # substring match, or both empty


def score(got: dict, want: dict) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []  # (label, passed, what we got)
    checks.append(("vendor", _name_ok(got["vendorName"], want["vendorName"]), got["vendorName"]))
    checks.append(("tax id", got["vatId"] == want["vatId"], got["vatId"] or "(none)"))  # exact after normalisation
    checks.append(("customer", _name_ok(got["department"], want["department"]), got["department"]))
    checks.append(
        ("total", _money_ok(got["totalCost"], want["totalCost"]), f"{got['totalCost']:.2f}")
    )
    n_got, n_want = len(got["orderLines"]), len(want["orderLines"])  # catches included alternatives
    checks.append(("line count", n_got == n_want, f"{n_got} (want {n_want})"))
    got_sum = round(sum(l["totalPrice"] for l in got["orderLines"]), 2)  # lines should add to the net subtotal
    checks.append(("lines sum", _money_ok(got_sum, want["netSubtotal"]), f"{got_sum:.2f}"))
    checks.append(
        (
            "group",
            (got["commodityGroupName"] or "") == want["commodityGroupName"],  # classification
            got["commodityGroupName"] or "(none)",
        )
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="write raw results here")  # optional raw dump for the PR
    args = parser.parse_args()

    if not settings.openai_api_key:  # fail fast with a clear message
        print("OPENAI_API_KEY is not set; put it in backend/.env first.")
        return 1

    engine = create_engine(  # throwaway SQLite in memory
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)  # create every table
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)  # session factory
    with Session() as db:
        seed(db)  # commodity groups, suppliers, articles, demo orgs

    expected = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))  # ground truth
    results: dict[str, dict] = {}  # filename -> raw response, for --json
    failures = 0  # becomes the exit code
    for filename, want in expected.items():
        if filename.startswith("_"):  # skip the "_comment" key
            continue
        pdf_path = FIXTURES / filename
        print(f"\n=== {filename}")
        t0 = time.perf_counter()  # time parse + model call together
        text = extract_text(pdf_path.read_bytes())  # step 1: PDF -> text
        with Session() as db:
            resp = extract_vendor_data(text, db)  # step 2: text -> structured data
        elapsed = time.perf_counter() - t0
        payload = resp.model_dump(by_alias=True)  # camelCase, exactly what the frontend sees
        results[filename] = payload
        if not resp.success or resp.data is None:  # hard failure: count every check as failed
            print(f"  FAILED: {resp.error}")
            failures += 7
            continue
        got = payload["data"]
        for label, ok, shown in score(got, want):  # print one line per check
            failures += 0 if ok else 1
            print(f"  [{'ok' if ok else 'XX'}] {label:<10} {shown}")
        if resp.warnings:  # arithmetic warnings the UI would show
            print("  warnings:", " | ".join(resp.warnings))
        if resp.missing_fields:  # fields the user would have to fill in
            print("  missing: ", ", ".join(resp.missing_fields))
        print(f"  {elapsed:.1f}s, title={got['title']!r}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")  # raw dump
    print(f"\n{failures} failed checks")
    return failures  # 0 = all good


if __name__ == "__main__":
    sys.exit(main())  # exit code = number of failed checks
