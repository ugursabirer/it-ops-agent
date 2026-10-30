"""Custom MCP server exposing the IT operations database as MCP tools.

The server speaks MCP over stdio, which means the agent starts it as a child
process and no second terminal is needed. Run it directly only when you want to
inspect it by hand:

    python ops_mcp/server.py

Tools are deliberately read-only except for ``create_ticket``; that one is the
single write action in the system and the agent is only allowed to reach it
after the user has approved the step.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep stderr for tool traces only; FastMCP's startup logging is not useful
# to the agent and would be interleaved with the trace output.
os.environ.setdefault("FASTMCP_LOG_LEVEL", "ERROR")

from fastmcp import FastMCP  # noqa: E402

from ops_mcp import database  # noqa: E402

mcp = FastMCP("IT Operations MCP Server")


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _trace(message: str) -> None:
    """Log to stderr: stdout is reserved for the MCP protocol itself."""

    print(f"[MCP TOOL] {message}", file=sys.stderr, flush=True)


@mcp.tool
def find_employee(query: str) -> str:
    """Calisani personel numarasi (orn. E-1003) veya adiyla bulur.

    Donen kayit calisanin departmanini, unvanini ve yoneticisini icerir.
    Birden fazla kisi eslesirse adaylari listeler.
    """

    _trace(f"find_employee({query})")

    employee = database.find_employee(query)

    if employee is None:
        return _dump({"sonuc": "Calisan bulunamadi.", "sorgu": query})

    return _dump(employee)


@mcp.tool
def get_employee_assets(employee_id: str) -> str:
    """Calisana zimmetli cihazlari listeler.

    Her cihaz icin yas_ay (satin almadan bugune kadar gecen ay sayisi) ve
    garanti_durumu alanlari hesaplanmis olarak doner.
    """

    _trace(f"get_employee_assets({employee_id})")

    assets = database.list_employee_assets(employee_id)

    if not assets:
        return _dump(
            {
                "employee_id": employee_id,
                "sonuc": "Bu calisana zimmetli cihaz bulunamadi.",
            }
        )

    return _dump({"employee_id": employee_id, "cihazlar": assets})


@mcp.tool
def get_asset_failure_history(asset_id: str, months: int = 6) -> str:
    """Bir cihaz icin verilen ay penceresindeki ariza kayitlarini sayar.

    Donanim yenileme politikasindaki tekrarlayan ariza istisnasini
    degerlendirmek icin kullanilir.
    """

    _trace(f"get_asset_failure_history({asset_id}, months={months})")

    return _dump(database.count_recent_failures(asset_id, months=months))


@mcp.tool
def get_employee_tickets(employee_id: str, only_open: bool = False) -> str:
    """Calisanin BT taleplerini listeler. only_open=True ise sadece acik olanlar."""

    _trace(f"get_employee_tickets({employee_id}, only_open={only_open})")

    tickets = database.list_tickets(employee_id, only_open=only_open)

    return _dump(
        {
            "employee_id": employee_id,
            "sadece_acik": only_open,
            "talepler": tickets,
        }
    )


@mcp.tool
def get_employee_entitlements(employee_id: str) -> str:
    """Calisanin sistem erisim yetkilerini (VPN, ERP, Jira vb.) listeler."""

    _trace(f"get_employee_entitlements({employee_id})")

    entitlements = database.list_entitlements(employee_id)

    if not entitlements:
        return _dump(
            {
                "employee_id": employee_id,
                "yetkiler": [],
                "sonuc": "Bu calisan icin tanimli erisim yetkisi yok.",
            }
        )

    return _dump({"employee_id": employee_id, "yetkiler": entitlements})


@mcp.tool
def get_spare_stock(asset_type: str = "") -> str:
    """Yedek cihaz stogunu getirir. asset_type bos ise tum stok doner.

    Gecerli asset_type degerleri: dizustu, cep_telefonu, monitor.
    """

    _trace(f"get_spare_stock({asset_type!r})")

    stock = database.list_spare_stock(asset_type or None)

    return _dump({"asset_type": asset_type or "tumu", "stok": stock})


@mcp.tool
def create_ticket(
    employee_id: str,
    category: str,
    subject: str,
    priority: str,
    assigned_group: str,
    asset_id: str = "",
) -> str:
    """Yeni bir BT talebi olusturur. Bu tek yazma islemidir.

    category: ariza | talep | degisiklik | bilgi
    priority: kritik | yuksek | normal | dusuk
    assigned_group: BT-Donanim | BT-Altyapi | BT-Uygulama | Bilgi-Guvenligi
    """

    _trace(f"create_ticket({employee_id}, {category}, {subject!r})")

    employee = database.find_employee(employee_id)

    if employee is None or employee.get("belirsiz"):
        return _dump(
            {
                "hata": "Gecerli bir calisan kimligi olmadan talep acilamaz.",
                "employee_id": employee_id,
            }
        )

    valid_categories = {"ariza", "talep", "degisiklik", "bilgi"}
    valid_priorities = {"kritik", "yuksek", "normal", "dusuk"}

    if category not in valid_categories:
        return _dump(
            {
                "hata": f"Gecersiz kategori: {category}",
                "gecerli_degerler": sorted(valid_categories),
            }
        )

    if priority not in valid_priorities:
        return _dump(
            {
                "hata": f"Gecersiz oncelik: {priority}",
                "gecerli_degerler": sorted(valid_priorities),
            }
        )

    ticket = database.insert_ticket(
        employee_id=employee_id,
        category=category,
        subject=subject,
        priority=priority,
        assigned_group=assigned_group,
        asset_id=asset_id or None,
    )

    return _dump({"sonuc": "Talep olusturuldu.", "talep": ticket})


if __name__ == "__main__":
    # show_banner=False keeps stderr clean; the agent uses it for tool traces.
    mcp.run(show_banner=False)
