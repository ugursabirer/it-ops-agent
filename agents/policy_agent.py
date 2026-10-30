"""Policy agent: answers from the policy documents through RAG tools."""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from config import llm, log
from state import AssistantState
from tools.policy_tools import POLICY_TOOLS
from tracing import preview

POLICY_PROMPT = """
Sen kurumsal BT politika agent'isin. Sirket politikalarini yalnizca sana verilen
arama tool'lari ile okursun.

search_it_policy
  Donanim yenileme suresi ve istisnalari, zimmet, yedek cihaz, talep
  kategorileri, oncelik ve cozum sureleri (SLA), atama gruplari, onay
  hiyerarsisi, yazilim ve lisans kurallari.

search_security_policy
  Parola kurallari, MFA, veri siniflandirma, USB/tasinabilir medya,
  olay bildirimi, VPN erisim hakki ve erisim seviyeleri.

Kurallar:
- Politika bilgisini MUTLAKA tool ile getir. Hafizadan kural uydurma.
- Soru birden fazla konuyu kapsiyorsa birden fazla tool cagirabilirsin.
- Tool sonucunda olmayan bir kurali varmis gibi anlatma; bilgi yoksa
  "politika dokumanlarinda bu konuda madde bulunmuyor" de.
- Cevabinda dayandigin maddeyi ve kaynak dosyayi belirt.
- Turkce, kisa ve net yaz. Sayisal esikleri (ay, karakter, saat) aynen aktar.
"""

TALEP_SUFFIX = """

Bu tur cok adimli bir istegin ilk asamasindasin. Senden beklenen kullaniciya
nihai cevabi vermek DEGIL, karar icin gerekli politika kurallarini net bicimde
ortaya koymaktir. Ilgili esikleri ve istisnalari madde madde yaz.
Calisanin kendi verisini sen sorgulamayacaksin; onu sonraki agent yapacak.

Hangi kurali getirecegini sasirma:
- Istek bir cihazin yenilenmesi, degistirilmesi veya yeni bir cihaz verilmesi
  ile ilgiliyse, UYGUNLUK kurallarini getir: yenileme suresi (ay cinsinden
  esik) ve sure dolmadan yenileme istisnalari. Onay hiyerarsisi veya SLA
  sureleri tek basina uygunluk kurali DEGILDIR; onlar talep acildiktan
  sonrasini anlatir.
- Istek bir erisim/yetki tanimlanmasi ile ilgiliyse, erisim hakki ve erisim
  seviyesi kurallarini getir.
- Talebin hangi kategori, oncelik ve gruba acilacagi gerekiyorsa bunu
  uygunluk kurallarina EK olarak getir, onlarin yerine degil.
"""


def _prompt(state: AssistantState) -> str:
    prompt = POLICY_PROMPT

    if state.get("route") == "talep":
        prompt += TALEP_SUFFIX

        plan = state.get("plan") or []

        if plan:
            steps = "\n".join(
                f"{index}. {step}" for index, step in enumerate(plan, start=1)
            )
            prompt += f"\nPlanner Agent'in cikardigi plan:\n{steps}\n"

    return prompt


policy_llm = llm.bind_tools(POLICY_TOOLS)


def policy_node(state: AssistantState) -> dict:
    log("[AGENT] Policy Agent (RAG)")

    response = policy_llm.invoke(
        [SystemMessage(content=_prompt(state)), *state["messages"]]
    )

    preview(response)

    return {
        "messages": [response],
        "last_agent": "politika",
        "policy_loops": state.get("policy_loops", 0) + 1,
    }
