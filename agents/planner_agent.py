"""Planner agent: turns a multi-step request into an explicit plan.

Only the ``talep`` lane runs the planner. Writing the plan down before any tool
is touched does two things: it tells the gathering agents what to look for, and
it makes the agent's reasoning visible in the terminal and in the final answer.
"""

from __future__ import annotations

import re

from langchain_core.messages import SystemMessage

from config import llm, log
from state import AssistantState

MAX_STEPS = 5

PLANNER_PROMPT = """
Sen kurumsal BT destek sisteminin Planner Agent'isin.
Kullanicinin cok adimli istegini, en fazla 5 adimlik bir plana cevirirsin.

Sistemde su bilgi kaynaklari var:
- POLITIKA (dokuman aramasi): donanim yenileme suresi ve istisnalari, yedek
  cihaz, talep kategorileri, oncelik/SLA, atama gruplari, onay hiyerarsisi,
  lisans kurallari, bilgi guvenligi, VPN ve erisim seviyeleri.
- OPERASYONEL VERI (kurumsal sistem): calisan kaydi, zimmetli cihazlar ve
  cihaz yasi, cihazin ariza gecmisi, acik talepler, erisim yetkileri,
  yedek cihaz stogu.
- AKSIYON: yeni BT talebi acma (kullanici onayi gerekir).

Plan yazma kurallari:
- Her adim tek satir, fiille baslayan kisa bir ifade olsun.
- Once hangi politika kuralinin gerektigini, sonra hangi operasyonel verinin
  kontrol edilecegini yaz.
- Istek bir talep acilmasini gerektiriyorsa son adim talebin acilmasi olsun.
- Numarali liste kullan. Aciklama, baslik veya yorum ekleme.

Ornek:
1. Donanim yenileme suresini ve istisna kurallarini politikadan oku
2. Calisanin kimligini ve zimmetli cihazini bul
3. Cihazin yasini ve son 6 aydaki ariza sayisini kontrol et
4. Politika kuralini cihazin durumuyla karsilastirip uygunluk karari ver
5. Uygunsa yedek cihaz stogunu kontrol edip BT talebi ac
"""

_STEP_PATTERN = re.compile(r"^\s*(?:\d+[\.\)]|[-*])\s*(.+)$")


def _parse_plan(text: str) -> list[str]:
    """Pull the step lines out of the model's answer, defensively."""

    steps: list[str] = []

    for line in (text or "").splitlines():
        match = _STEP_PATTERN.match(line)

        if match:
            step = match.group(1).strip()

            if step:
                steps.append(step)

    if not steps:
        # The model answered in prose. Keep the non-empty lines as the plan
        # rather than dropping the planning step entirely.
        steps = [line.strip() for line in (text or "").splitlines() if line.strip()]

    return steps[:MAX_STEPS]


def planner_node(state: AssistantState) -> dict:
    log("[AGENT] Planner Agent")

    response = llm.invoke([SystemMessage(content=PLANNER_PROMPT), *state["messages"]])

    plan = _parse_plan(response.content or "")

    if plan:
        log("[PLAN]")
        for index, step in enumerate(plan, start=1):
            log(f"  {index}. {step}")
    else:
        log("[PLAN] Plan uretilemedi, adimlar olmadan devam ediliyor.")

    # The plan is state, not conversation: it is injected into the specialists'
    # prompts instead of being appended to the message history, so it cannot be
    # mistaken for an answer to the user.
    return {"plan": plan, "last_agent": "talep"}
