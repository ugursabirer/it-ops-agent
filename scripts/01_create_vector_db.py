"""Build the FAISS vector stores used by the policy RAG tools.

Run once after cloning:

    python scripts/01_create_vector_db.py

Two separate stores are created so that the policy agent has a real tool choice
to make instead of searching one undifferentiated index:

* ``it_policy``       - hardware renewal, request/SLA and software licensing
* ``security_policy`` - information security, VPN and remote access

Chunking is heading based rather than fixed-size: policy documents are written
as short, self-contained clauses under '##' headings, and keeping a clause whole
retrieves far better than cutting it at an arbitrary character count. Every
chunk is prefixed with its document title and heading so the retrieved text
still carries its context when it reaches the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from config import embeddings, settings  # noqa: E402

MAX_CHUNK_CHARS = 900

STORES: dict[str, list[str]] = {
    "it_policy": [
        "donanim_yenileme_politikasi.md",
        "talep_ve_sla_politikasi.md",
        "yazilim_lisans_politikasi.md",
    ],
    "security_policy": [
        "bilgi_guvenligi_politikasi.md",
        "vpn_ve_erisim_politikasi.md",
    ],
}


def split_by_heading(markdown: str) -> list[tuple[str, str]]:
    """Split a markdown policy into (heading, body) pairs on '##' boundaries."""

    title = "Politika"
    heading = "Genel"
    buffer: list[str] = []
    sections: list[tuple[str, str]] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((f"{title} / {heading}", body))

    for line in markdown.splitlines():
        if line.startswith("# "):
            flush()
            buffer = []
            title = line[2:].strip()
            heading = "Genel"
        elif line.startswith("## "):
            flush()
            buffer = []
            heading = line[3:].strip()
        else:
            buffer.append(line)

    flush()

    return sections


def build_documents(file_names: list[str]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS, chunk_overlap=120
    )

    documents: list[Document] = []

    for file_name in file_names:
        path = settings.policy_path / file_name
        markdown = path.read_text(encoding="utf-8")

        for context, body in split_by_heading(markdown):
            # Oversized sections (long tables) are cut further, but each piece
            # keeps the heading prefix so it stays interpretable on its own.
            pieces = (
                splitter.split_text(body) if len(body) > MAX_CHUNK_CHARS else [body]
            )

            for piece in pieces:
                documents.append(
                    Document(
                        page_content=f"[{context}]\n{piece}",
                        metadata={"kaynak": file_name, "bolum": context},
                    )
                )

    return documents


def main() -> None:
    settings.vector_path.mkdir(parents=True, exist_ok=True)

    for store_name, file_names in STORES.items():
        documents = build_documents(file_names)

        print(f"\n{store_name}: {len(documents)} chunk")
        for document in documents:
            print(f"  - {document.metadata['bolum']}")

        vector_db = FAISS.from_documents(documents, embeddings)

        output_path = settings.vector_path / store_name
        output_path.mkdir(parents=True, exist_ok=True)
        vector_db.save_local(str(output_path))

        print(f"[OK] {store_name} olusturuldu -> {output_path}")


if __name__ == "__main__":
    main()
