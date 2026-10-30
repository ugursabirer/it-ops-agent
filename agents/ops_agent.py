"""Operations agent: reads corporate operational data through the MCP tools.

The tool list is injected at build time because it depends on which transport
is active (MCP or the local fallback), so this module exposes a factory rather
than a ready node.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from config import llm, log
from state import AssistantState
from tracing import preview

OPS_PROMPT = """
Sen kurumsal BT operasyon agent'isin. Sirket sistemindeki guncel veriyi
yalnizca sana verilen tool'lar ile okursun.

Elindeki tool'lar:
- find_employee: calisani personel numarasi (E-1003) veya adiyla bulur.
- get_employee_assets: calisanin zimmetli cihazlarini, cihaz yasini (yas_ay)
  ve garanti durumunu getirir.
- get_asset_failure_history: bir cihazin son N aydaki ariza sayisini getirir.
- get_employee_tickets: calisanin BT taleplerini getirir.
- get_employee_entitlements: calisanin erisim yetkilerini (VPN, ERP, Jira) getirir.
- get_spare_stock: yedek cihaz stogunu getirir.

Kurallar:
- Veriyi MUTLAKA tool ile getir. Cihaz, talep, yetki veya stok bilgisi uydurma.
- Calisan kimligi belli degilse once find_employee ile bul; personel numarasini
  sonraki cagrilarda kullan.
- Cihaz yasini kendin hesaplamaya calisma; yas_ay alani zaten hesaplanmis geliyor.
  Bu alan AY cinsindendir: "25 ay" diye yaz, "25 yasinda" deme.
- Tool "bulunamadi" veya "hata" donduyse bunu oldugu gibi aktar, tahmin yurutme.
- Kullanici kim oldugunu soylemediyse ve veri kisiye bagliysa tool cagirmak
  yerine "hangi calisan icin sorguluyorsunuz?" diye sor.
- Turkce, kisa ve veriye dayali yaz.
"""

TALEP_SUFFIX = """

Bu cok adimli bir istegin veri toplama asamasidir. Onceki mesajlarda Policy
Agent gerekli politika kurallarini ortaya koydu. Senden beklenen, o kurallarin
degerlendirilebilmesi icin gereken TUM operasyonel veriyi toplamaktir:
calisan kaydi, ilgili cihaz ve yasi, cihazin ariza gecmisi, varsa acik talepler
ve gerekiyorsa yedek cihaz stogu. Nihai karari sen verme, sadece veriyi topla
ve ozetle.
"""

# A long message history pushes the actual job out of a small model's focus: in
# testing it would summarise the policy agent's answer instead of fetching any
# data. A directive appended as the last message fixes that. It is passed only
# to this invocation and never stored in state, so it cannot accumulate.
TALEP_DIRECTIVE = """
SIRADAKI ADIM - bunu simdi yap:

Yukaridaki politika kurallarini degerlendirmek icin gereken operasyonel veriyi
TOOL CAGIRARAK topla. Henuz toplanmamis veri varsa tool cagir; aciklama yazma.

Tipik sira:
1. find_employee ile calisani bul ve employee_id'yi ogren
2. get_employee_assets ile ilgili cihazi ve yas_ay degerini al
3. get_asset_failure_history ile cihazin son 6 aydaki ariza sayisini al
4. gerekiyorsa get_employee_tickets ve get_spare_stock

ONEMLI - kimlik tahmin etme:
Bir adimin parametresi onceki adimin sonucundan geliyorsa o adimi AYNI ANDA
cagirma; once o sonucu bekle. Hicbir kimlik degerini kendin uretme veya
ornekten kopyalama. get_asset_failure_history'yi yalnizca
get_employee_assets sonucunda GERCEKTEN donen asset_id ile cagir. Ayni kural
employee_id icin de gecerli: onu find_employee sonucundan al.

Gerekli tum veri zaten onceki tool sonuclarinda varsa yeni tool cagirma;
bunun yerine topladigin veriyi kisa maddeler halinde ozetle.
"""


def _prompt(state: AssistantState) -> str:
    prompt = f"{OPS_PROMPT}\nBugunun tarihi: {date.today().isoformat()}\n"

    if state.get("route") == "talep":
        prompt += TALEP_SUFFIX

        plan = state.get("plan") or []

        if plan:
            steps = "\n".join(
                f"{index}. {step}" for index, step in enumerate(plan, start=1)
            )
            prompt += f"\nPlanner Agent'in cikardigi plan:\n{steps}\n"

    return prompt


def make_ops_node(read_tools: list[BaseTool]) -> Callable[[AssistantState], dict]:
    """Bind the read-only operational tools and return the agent node."""

    ops_llm = llm.bind_tools(read_tools)

    def ops_node(state: AssistantState) -> dict:
        log("[AGENT] Operations Agent (MCP)")

        conversation = [
            SystemMessage(content=_prompt(state)),
            *state["messages"],
        ]

        if state.get("route") == "talep":
            conversation.append(HumanMessage(content=TALEP_DIRECTIVE))

        response = ops_llm.invoke(conversation)

        preview(response)

        return {
            "messages": [response],
            "last_agent": "operasyon",
            "ops_loops": state.get("ops_loops", 0) + 1,
        }

    return ops_node
