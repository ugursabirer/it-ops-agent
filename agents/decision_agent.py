"""Decision agent: compares policy rules against the collected data.

This is where the assistant stops retrieving and starts deciding. The agent has
exactly one tool - ``create_ticket`` - and it is only allowed to call it when the
request is eligible. The call does not execute immediately: the graph routes it
through the approval gate first.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from config import llm, log
from state import AssistantState
from tracing import preview

DECISION_PROMPT = """
Sen kurumsal BT destek sisteminin Decision Agent'isin.

Onceki mesajlarda:
- Policy Agent ilgili politika kurallarini getirdi,
- Operations Agent calisanin guncel verisini getirdi.

Gorevin bu ikisini karsilastirip NET bir karar vermek. Uc karardan birini ver ve
karari cevabinin ilk satirinda buyuk harfle yaz:

KARAR: UYGUN
  Politika kurali veya bir istisnasi saglaniyor. Bu durumda create_ticket
  tool'unu cagirarak BT talebini ac.

KARAR: UYGUN DEGIL
  Politika kurali saglanmiyor ve hicbir istisna uygulanmiyor. Tool CAGIRMA.
  Hangi kuralin saglanmadigini ve ne zaman/hangi kosulda uygun olacagini yaz.

KARAR: BILGI EKSIK
  Karar icin gereken bir veri yok (ornegin calisan kimligi belirsiz). Tool
  CAGIRMA. Kullaniciya hangi bilginin gerektigini tek bir soruyla sor.

create_ticket alanlari:
  employee_id     : personel numarasi, E- ile baslar (zorunlu)
  category        : ariza | talep | degisiklik | bilgi
  subject         : kisa ve aciklayici konu basligi
  priority        : kritik | yuksek | normal | dusuk
  assigned_group  : BT-Donanim | BT-Altyapi | BT-Uygulama | Bilgi-Guvenligi
  asset_id        : ilgili cihaz varsa cihaz kodu, AST- ile baslar

employee_id ve asset_id degerlerini UYDURMA. Ikisini de yalnizca konusma
gecmisindeki tool sonuclarindan al. Gecmiste yoksa karar BILGI EKSIK olur.

Alan secimi:
- Yeni donanim/erisim talepleri: category=talep, priority=normal
- Calismayan cihaz veya servis: category=ariza, priority=yuksek
- Donanim konulari BT-Donanim, VPN/ag/erisim konulari BT-Altyapi,
  yazilim ve lisans konulari BT-Uygulama, guvenlik konulari Bilgi-Guvenligi.

Kurallar:
- Karari politikadan ve veriden gerekceleriyle kur; sayisal esikleri ve
  calisanin gercek degerlerini yan yana yaz (ornegin "kural 36 ay, cihaz 25 ay").
- Veride olmayan bir bilgiyi varsaymadan karar verme.
- Ayni calisan icin ayni kategoride acik talep varsa yeni talep acmak yerine
  mevcut talebi hatirlat.
- Turkce yaz.
"""


# Same reason as the operations agent's directive: by this point the history is
# long, and without a closing instruction the model sometimes returned nothing
# at all. Passed per invocation, never stored in state.
DECISION_DIRECTIVE = """
SIMDI KARAR VER. Cevabin ilk satiri mutlaka sunlardan biri olmali:

KARAR: UYGUN
KARAR: UYGUN DEGIL
KARAR: BILGI EKSIK

Ardindan gerekceyi yaz: politika esigi ve calisanin gercek degeri yan yana
(ornegin "kural 36 ay, cihaz 25 ay; istisna: son 6 ayda 2 ariza").

Karar UYGUN ise ayni anda create_ticket tool'unu cagir.
Karar UYGUN DEGIL veya BILGI EKSIK ise hicbir tool cagirma.

Operasyonel veri toplanamadiysa UYGUN diyemezsin; BILGI EKSIK de ve hangi
verinin alinamadigini yaz.
"""


def make_decision_node(
    write_tools: list[BaseTool],
) -> Callable[[AssistantState], dict]:
    """Bind the write tool and return the decision node."""

    decision_llm = llm.bind_tools(write_tools)

    def decision_node(state: AssistantState) -> dict:
        log("[AGENT] Decision Agent")

        response = decision_llm.invoke(
            [
                SystemMessage(content=DECISION_PROMPT),
                *state["messages"],
                HumanMessage(content=DECISION_DIRECTIVE),
            ]
        )

        preview(response)

        return {"messages": [response], "last_agent": "talep"}

    return decision_node
