"""Offline evaluation of the assistant against a fixed scenario set.

The harness is deliberately simple: for each scenario it runs one turn through
the real graph and checks three things.

1. Did the supervisor pick the expected lane?
2. Does the answer contain the facts it must contain (``beklenen_ifadeler``)?
3. Does it avoid what it must not say (``yasak_ifadeler``)?

The third check is the important one. It is what catches the failure mode that
matters in an operational assistant: inventing a ticket number or a device for a
request that should have been refused or clarified.

Scenarios never approve a write, so the evaluation cannot change the database.

    python eval/run_eval.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph import build_graph  # noqa: E402
from tools import ops_toolset  # noqa: E402

EVAL_SET = Path(__file__).resolve().parent / "eval_set.json"


def _answer_of(result: dict[str, Any]) -> str:
    messages = result.get("messages") or []

    if not messages:
        return ""

    return str(getattr(messages[-1], "content", ""))


async def _run_case(graph: Any, case: dict[str, Any]) -> dict[str, Any]:
    config = {"configurable": {"thread_id": f"eval-{case['id']}"}}
    started = time.monotonic()

    try:
        result = await graph.ainvoke(
            {"messages": [("user", case["soru"])]}, config=config
        )

        # Reaching the approval gate means the agent wanted to write. The
        # evaluation declines, which both keeps the data clean and exercises
        # the rejection path.
        if result.get("__interrupt__"):
            from langgraph.types import Command

            result = await graph.ainvoke(Command(resume="iptal"), config=config)

        answer = _answer_of(result)
        route = result.get("route")
        error: str | None = None

    except Exception as exception:  # noqa: BLE001 - recorded as a failure
        answer, route, error = "", None, str(exception)

    elapsed = time.monotonic() - started

    lowered = answer.lower()

    missing = [
        needle for needle in case["beklenen_ifadeler"] if needle.lower() not in lowered
    ]
    forbidden = [
        needle for needle in case["yasak_ifadeler"] if needle.lower() in lowered
    ]

    expected_route = case.get("beklenen_kulvar")
    route_ok = expected_route is None or route == expected_route

    return {
        "id": case["id"],
        "gecti": not missing and not forbidden and route_ok and error is None,
        "kulvar": route,
        "kulvar_ok": route_ok,
        "eksik": missing,
        "yasakli": forbidden,
        "hata": error,
        "sure": elapsed,
        "cevap": answer,
    }


async def main() -> int:
    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))

    async with ops_toolset() as toolset:
        graph = build_graph(toolset)

        print(f"\n{len(cases)} senaryo calistiriliyor ({toolset.transport})\n")

        results = []

        for case in cases:
            print(f"-> {case['id']}")
            results.append(await _run_case(graph, case))

    print("\n" + "=" * 78)
    print(f"{'SENARYO':<28}{'SONUC':<8}{'KULVAR':<12}{'SURE':<8}NOT")
    print("=" * 78)

    for result in results:
        status = "GECTI" if result["gecti"] else "BASARISIZ"

        notes = []
        if not result["kulvar_ok"]:
            notes.append("yanlis kulvar")
        if result["eksik"]:
            notes.append(f"eksik: {', '.join(result['eksik'])}")
        if result["yasakli"]:
            notes.append(f"yasakli: {', '.join(result['yasakli'])}")
        if result["hata"]:
            notes.append(f"hata: {result['hata'][:40]}")

        print(
            f"{result['id']:<28}{status:<8}{str(result['kulvar']):<12}"
            f"{result['sure']:>5.1f}s  {'; '.join(notes)}"
        )

    passed = sum(1 for result in results if result["gecti"])
    total = len(results)

    print("=" * 78)
    print(f"Sonuc: {passed}/{total} senaryo gecti")
    print("=" * 78 + "\n")

    report = Path(__file__).resolve().parent / "last_run.json"
    report.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Ayrintili cikti: {report}\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
