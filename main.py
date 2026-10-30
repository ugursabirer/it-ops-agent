"""Kurumsal BT Destek ve Operasyon Asistani - komut satiri uygulamasi.

Kullanim:

    python main.py                # etkilesimli sohbet
    python main.py --demo         # senaryolari sirayla calistirir (gosterim icin)

Cikmak icin 'cikis', konusmayi sifirlamak icin 'yeni' yazin.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from langgraph.types import Command

from config import settings
from graph import build_graph
from tools import OpsToolset, ops_toolset

THREAD_PREFIX = "bt-destek"

# Scripted walkthrough used for the README and the screenshots. Each entry is a
# user message plus the answer to give at the approval gate, if one appears.
DEMO_SCENARIOS: list[tuple[str, str | None]] = [
    ("Merhaba, sen neler yapabiliyorsun?", None),
    ("Dizustu bilgisayar kac yilda bir yenilenir?", None),
    ("E-1003'un zimmetindeki cihazlar neler?", None),
    (
        "Mert Kaya'nin laptopu surekli kapaniyor. Cihazini yenileme hakki var "
        "mi? Varsa gerekli BT talebini ac.",
        "onay",
    ),
    (
        "Elif Demir de yeni bir dizustu istiyor, onun hakki var mi?",
        "iptal",
    ),
    ("Yeni bilgisayar istiyorum.", None),
]


def banner(toolset: OpsToolset) -> None:
    print("=" * 72)
    print(" Kurumsal BT Destek ve Operasyon Asistani")
    print("=" * 72)
    print(f" LLM              : {settings.llm_model} (Ollama, lokal)")
    print(f" Embedding        : {settings.embedding_model}")
    print(f" Operasyon tool'u : {toolset.transport}")
    print(f"                    {toolset.note}")
    print(" Politika tool'u  : FAISS RAG (2 vector store)")
    print(
        f" Yazma onayi      : "
        f"{'kullanici onayi gerekli' if settings.require_approval else 'KAPALI'}"
    )
    print("-" * 72)
    print(" 'cikis' = cik   |   'yeni' = konusmayi sifirla")
    print("=" * 72)


def _interrupts(result: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(result.get("__interrupt__") or ())


def _print_approval_request(payload: dict[str, Any]) -> None:
    print("\n" + "!" * 72)
    print(" ONAY GEREKIYOR - asagidaki yazma islemi kullanici onayi bekliyor")
    print("!" * 72)
    print(f" Tool: {payload.get('tool')}")

    for key, value in (payload.get("parametreler") or {}).items():
        print(f"   {key:<16}: {value}")

    print("!" * 72)


def _print_answer(result: dict[str, Any]) -> None:
    messages = result.get("messages") or []

    if not messages:
        print("\nAsistan: (cevap uretilemedi)")
        return

    content = getattr(messages[-1], "content", "")

    print(f"\nAsistan: {content}\n")


async def run_turn(
    graph: Any,
    config: dict[str, Any],
    user_input: str,
    approval: str | None,
    interactive_approval: bool = True,
) -> None:
    """Run one user turn, pausing at the approval gate when one is reached.

    ``interactive_approval`` is False in demo mode, where there is no one at the
    keyboard: an unscripted gate declines instead of blocking on stdin.
    """

    try:
        result = await graph.ainvoke(
            {"messages": [("user", user_input)]}, config=config
        )

        while pending := _interrupts(result):
            _print_approval_request(pending[0].value)

            if approval is not None:
                answer = approval
                print(f" Onayliyor musunuz? (onay / iptal): {answer}")
            elif interactive_approval:
                answer = input(" Onayliyor musunuz? (onay / iptal): ")
            else:
                answer = "iptal"
                print(
                    " Onayliyor musunuz? (onay / iptal): iptal"
                    "   [demo modu: onay verilmedi]"
                )

            result = await graph.ainvoke(Command(resume=answer), config=config)

        _print_answer(result)

    except Exception as error:  # noqa: BLE001 - the CLI must survive a failure
        print(f"\n[HATA] Istek tamamlanamadi: {error}")
        print(
            "       Ollama'nin calistigini "
            "('ollama list') ve vector store'larin olusturuldugunu kontrol edin.\n"
        )


async def interactive(graph: Any) -> None:
    thread = 0
    config = {"configurable": {"thread_id": f"{THREAD_PREFIX}-{thread}"}}

    while True:
        try:
            user_input = input("Kullanici: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGorusuruz.")
            return

        if not user_input:
            continue

        if user_input.lower() in {"cikis", "exit", "quit"}:
            print("Gorusuruz.")
            return

        if user_input.lower() in {"yeni", "reset"}:
            thread += 1
            config = {"configurable": {"thread_id": f"{THREAD_PREFIX}-{thread}"}}
            print("[BILGI] Yeni konusma baslatildi, gecmis temizlendi.\n")
            continue

        await run_turn(graph, config, user_input, approval=None)


async def demo(graph: Any) -> None:
    for index, (user_input, approval) in enumerate(DEMO_SCENARIOS, start=1):
        print("\n" + "=" * 72)
        print(f" SENARYO {index}/{len(DEMO_SCENARIOS)}")
        print("=" * 72)
        print(f"Kullanici: {user_input}")

        config = {"configurable": {"thread_id": f"{THREAD_PREFIX}-demo-{index}"}}

        await run_turn(
            graph,
            config,
            user_input,
            approval=approval,
            interactive_approval=False,
        )


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kurumsal BT Destek ve Operasyon Asistani"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Hazir senaryolari sirayla calistirir.",
    )
    args = parser.parse_args()

    async with ops_toolset() as toolset:
        graph = build_graph(toolset)
        banner(toolset)

        if args.demo:
            await demo(graph)
        else:
            await interactive(graph)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
