"""Supervisor agent: routes a user message to the right lane.

The supervisor does not answer and does not call tools. It only decides which
of the four lanes handles this turn, which keeps the specialists' prompts short
and their tool choices small.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from config import llm, log
from state import AssistantState

ROUTES = {"selamlama", "politika", "operasyon", "talep"}

SUPERVISOR_PROMPT = """
Sen kurumsal BT destek sisteminin Supervisor Agent'isin.
Kullanici mesajini dort kulvardan birine yonlendirirsin. Cevap uretmezsin.

selamlama
  Selamlama, tesekkur, vedalasma veya "neler yapabilirsin" sorulari.

politika
  Yalnizca kural/politika bilgisi isteyen sorular. Belirli bir calisanin
  verisine ihtiyac duymaz.
  Ornek: "Parola kac karakter olmali?", "Dizustu kac yilda yenilenir?",
  "USB kullanimi serbest mi?", "VPN talebi hangi gruba acilir?"

operasyon
  Sistemdeki mevcut veriyi okuyan, tek adimlik sorgular. "Var mi / nedir /
  listele" diye sorulan ve cevabi tek bir kayitta bulunan her sey buraya girer.
  Ornek: "E-1003'un cihazlari neler?", "Benim acik talebim var mi?",
  "Zeynep Arslan'in VPN yetkisi var mi?", "Yedek dizustu stogu kac?"

  Mevcut bir yetkinin/cihazin/talebin DURUMUNU sormak operasyondur. Yeni bir
  yetki veya cihaz TALEP etmek, ya da buna hak kazanilip kazanilmadigini
  degerlendirmek talep kulvarina girer.

talep
  Politika kurali ile calisanin gercek verisinin birlikte degerlendirilmesini
  gerektiren, cok adimli istekler; ozellikle bir hak/uygunluk degerlendirmesi
  veya yeni bir BT talebi acilmasi istenen durumlar.
  Ornek: "Mert Kaya'nin laptopu surekli kapaniyor, yenileyebilir miyiz?",
  "Cihazimi yenileme hakkim var mi, varsa talep ac",
  "Bana VPN erisimi tanimlanabilir mi?"

Karar kurallari:
- Soru hem kurala hem kisiye ait veriye dayaniyorsa: talep
- Kisa bir takip mesaji geldiyse konusma gecmisine ve son calisan agent'a bak.
- Sadece su dort kelimeden birini yaz, baska hicbir sey yazma:
selamlama
politika
operasyon
talep
"""


def supervisor_node(state: AssistantState) -> dict:
    last_agent = state.get("last_agent", "yok")

    prompt = f"{SUPERVISOR_PROMPT}\nSon calisan specialist agent: {last_agent}"

    response = llm.invoke([SystemMessage(content=prompt), *state["messages"]])

    decision = (response.content or "").strip().lower()

    # The model is asked for one word, but a small model sometimes wraps it in a
    # sentence. Matching on containment is more forgiving than equality, and the
    # previous lane is a better fallback than an arbitrary default.
    route = next((candidate for candidate in ROUTES if candidate in decision), None)

    if route is None:
        route = last_agent if last_agent in ROUTES else "politika"
        log(f"[SUPERVISOR] karar okunamadi ({decision!r}) -> {route}")

    log(f"\n[SUPERVISOR] kulvar: {route}")

    return {"route": route, "policy_loops": 0, "ops_loops": 0}
