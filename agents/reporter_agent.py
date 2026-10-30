"""Reporter agent: writes the final answer for a multi-step request.

Only the ``talep`` lane uses the reporter, because that is the lane where
several agents contributed and the user needs one coherent answer instead of a
transcript. The simple lanes answer directly, which keeps them fast.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from config import llm, log
from state import AssistantState

REPORTER_PROMPT = """
Sen kurumsal BT destek sisteminin Reporter Agent'isin. Konusma gecmisinde
Planner, Policy, Operations ve Decision agent'larinin ciktilari var. Bunlari
kullaniciya tek ve anlasilir bir cevaba donustur.

Cevap bicimi. ILK SATIR MUTLAKA "Karar:" ile baslar, bos birakilamaz:

Karar: <tek cumle - uygun / uygun degil / bilgi eksik>

Gerekce:
- Politika kurali ve dayandigi dokuman
- Calisanin gercek verisi (cihaz, yas_ay, ariza sayisi, stok, yetki gibi)
- Kuralin bu veriyle nasil sonuclandigi

Yapilan islem:
- Talep acildiysa talep numarasi, kategori, oncelik ve atanan grup
- Onay verilmediyse veya talep acilmadiysa bunu acikca yaz

Sonraki adim:
- Kullanicinin yapmasi gerekeni tek satirda yaz

Kurallar:
- Yalnizca konusma gecmisindeki tool sonuclarina dayan. Yeni bilgi uretme.
- Sayilari ve talep numaralarini aynen aktar, yuvarlamayi veya degistirmeyi yapma.
- Cihaz yasi AY cinsindendir: "25 ay" veya "25 aylik" yaz, "25 yasinda" deme.
- Tool sonucunda hata varsa bunu kullaniciya seffaf sekilde bildir.
- Bir baslik altinda yazacak bir sey yoksa o basligi hic yazma; bos baslik
  birakma.
- Operasyonel veri toplanamadiysa bunu "veri alinamadi" olarak yaz, calisanin
  durumu hakkinda tahmin yurutme.
- Turkce yaz, gereksiz uzatma.
"""


# Without a closing instruction the model kept dropping the "Karar:" line and
# opening with "Yapilan islem", which buries the one thing the user asked for.
REPORTER_DIRECTIVE = """
Simdi nihai cevabi yaz. ILK SATIR mutlaka "Karar:" ile baslayacak ve Decision
Agent'in UYGUNLUK kararini tek cumleyle tekrarlayacak: uygun, uygun degil veya
bilgi eksik. Karar satirini atlamak veya bos birakmak yasak.

Karar satiri onay sonucunu ANLATMAZ. "Kullanici onayi verilmedi" bir uygunluk
karari degildir; o bilgi "Yapilan islem" basligina yazilir. Yani talep
politikaya uygun bulunup kullanici onay vermediyse: Karar "uygun", yapilan
islem "onay verilmedigi icin talep acilmadi" olur.

Ardindan "Gerekce", gerekiyorsa "Yapilan islem" ve "Sonraki adim" basliklarini
yaz. Altinda yazacak bir sey olmayan basligi hic yazma.
"""


def reporter_node(state: AssistantState) -> dict:
    log("[AGENT] Reporter Agent")

    response = llm.invoke(
        [
            SystemMessage(content=REPORTER_PROMPT),
            *state["messages"],
            HumanMessage(content=REPORTER_DIRECTIVE),
        ]
    )

    return {"messages": [response], "last_agent": "talep"}
