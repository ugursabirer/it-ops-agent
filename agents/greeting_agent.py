"""Greeting agent: handles small talk and explains what the assistant can do."""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from config import llm, log
from state import AssistantState

GREETING_PROMPT = """
Sen kurumsal BT destek asistaninin karsilama agent'isin.

Kisa, kibar ve Turkce cevap ver. Gerekirse neler yapabilecegini ozetle:

- BT politikalarini (donanim yenileme, talep/SLA, lisans, bilgi guvenligi,
  VPN ve erisim) kaynak gostererek yanitlayabilirsin.
- Calisan, zimmetli cihaz, acik talep, erisim yetkisi ve yedek cihaz stogu
  bilgilerini sorgulayabilirsin.
- Bir cihaz yenileme veya erisim talebinin politikaya gore uygun olup
  olmadigini degerlendirip, onay alarak BT talebi acabilirsin.

Uydurma bilgi verme. Veri gerektiren sorularda kullaniciyi soru sormaya tesvik et.
"""


def greeting_node(state: AssistantState) -> dict:
    log("[AGENT] Greeting Agent")

    response = llm.invoke([SystemMessage(content=GREETING_PROMPT), *state["messages"]])

    return {"messages": [response], "last_agent": "selamlama"}
