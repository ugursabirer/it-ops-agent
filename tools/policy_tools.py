"""RAG tools over the corporate policy documents.

Two FAISS stores are exposed as two separate tools so the agent has to decide
which body of rules a question belongs to. Each result carries the source file
and section, which the reporter agent uses to cite its answer.
"""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

from config import embeddings, log, settings

_TOP_K = 3


def _load(store_name: str) -> FAISS:
    path = settings.vector_path / store_name

    if not path.exists():
        raise FileNotFoundError(
            f"Vector store bulunamadi: {path}. "
            "Once 'python scripts/01_create_vector_db.py' komutunu calistirin."
        )

    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)


_stores: dict[str, FAISS] = {}


def _store(store_name: str) -> FAISS:
    """Load a store once and keep it in memory for the rest of the session."""

    if store_name not in _stores:
        _stores[store_name] = _load(store_name)

    return _stores[store_name]


def _search(store_name: str, query: str) -> str:
    try:
        results = _store(store_name).similarity_search(query, k=_TOP_K)
    except FileNotFoundError as error:
        return f"POLITIKA KAYNAGI OKUNAMADI: {error}"

    if not results:
        return (
            "Politika dokumanlarinda bu konuyla ilgili bir madde bulunamadi. "
            "Cevap uretmek yerine bilginin bulunmadigini belirt."
        )

    return "\n\n---\n\n".join(
        f"Kaynak: {document.metadata.get('kaynak', 'bilinmiyor')}\n"
        f"Bolum: {document.metadata.get('bolum', 'bilinmiyor')}\n\n"
        f"{document.page_content}"
        for document in results
    )


@tool
def search_it_policy(query: str) -> str:
    """Donanim yenileme, zimmet, talep/SLA ve yazilim lisans politikalarinda arar.

    Cihaz yenileme suresi, istisnalar, yedek cihaz, talep kategorileri,
    oncelik ve cozum sureleri, atama gruplari, onay hiyerarsisi ve lisans
    kurallari bu tool ile sorgulanir.
    """

    log(f"  [TOOL] search_it_policy({query!r})")

    return _search("it_policy", query)


@tool
def search_security_policy(query: str) -> str:
    """Bilgi guvenligi, VPN ve uzaktan erisim politikalarinda arar.

    Parola kurallari, MFA, veri siniflandirma, USB kullanimi, olay bildirimi,
    VPN erisim hakki ve erisim seviyeleri bu tool ile sorgulanir.
    """

    log(f"  [TOOL] search_security_policy({query!r})")

    return _search("security_policy", query)


POLICY_TOOLS = [search_it_policy, search_security_policy]
