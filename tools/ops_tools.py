"""Local fallback for the operational tools served by the MCP server.

The primary path is the custom MCP server in ``ops_mcp/server.py``. When MCP is
disabled in configuration, or the server cannot be started, the agent falls back
to these functions. They carry the same names, signatures and docstrings as the
MCP tools, so neither the graph nor the agent prompts have to change: only the
transport differs.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from config import log
from ops_mcp import database


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _failure(error: Exception, **context: Any) -> str:
    """Turn an exception into a result the model can reason about."""

    return _dump(
        {
            "hata": "Operasyonel veri kaynagina erisilemedi.",
            "ayrinti": str(error),
            **context,
        }
    )


@tool
def find_employee(query: str) -> str:
    """Calisani personel numarasi (orn. E-1003) veya adiyla bulur.

    Donen kayit calisanin departmanini, unvanini ve yoneticisini icerir.
    Birden fazla kisi eslesirse adaylari listeler.
    """

    log(f"  [TOOL] find_employee({query!r})")

    try:
        employee = database.find_employee(query)
    except Exception as error:  # noqa: BLE001 - surfaced to the model
        return _failure(error, sorgu=query)

    if employee is None:
        return _dump({"sonuc": "Calisan bulunamadi.", "sorgu": query})

    return _dump(employee)


@tool
def get_employee_assets(employee_id: str) -> str:
    """Calisana zimmetli cihazlari listeler.

    Her cihaz icin yas_ay (satin almadan bugune kadar gecen ay sayisi) ve
    garanti_durumu alanlari hesaplanmis olarak doner.
    """

    log(f"  [TOOL] get_employee_assets({employee_id!r})")

    try:
        assets = database.list_employee_assets(employee_id)
    except Exception as error:  # noqa: BLE001
        return _failure(error, employee_id=employee_id)

    if not assets:
        return _dump(
            {
                "employee_id": employee_id,
                "sonuc": "Bu calisana zimmetli cihaz bulunamadi.",
            }
        )

    return _dump({"employee_id": employee_id, "cihazlar": assets})


@tool
def get_asset_failure_history(asset_id: str, months: int = 6) -> str:
    """Bir cihaz icin verilen ay penceresindeki ariza kayitlarini sayar.

    Donanim yenileme politikasindaki tekrarlayan ariza istisnasini
    degerlendirmek icin kullanilir.
    """

    log(f"  [TOOL] get_asset_failure_history({asset_id!r}, months={months})")

    try:
        return _dump(database.count_recent_failures(asset_id, months=months))
    except Exception as error:  # noqa: BLE001
        return _failure(error, asset_id=asset_id)


@tool
def get_employee_tickets(employee_id: str, only_open: bool = False) -> str:
    """Calisanin BT taleplerini listeler. only_open=True ise sadece acik olanlar."""

    log(f"  [TOOL] get_employee_tickets({employee_id!r}, only_open={only_open})")

    try:
        tickets = database.list_tickets(employee_id, only_open=only_open)
    except Exception as error:  # noqa: BLE001
        return _failure(error, employee_id=employee_id)

    return _dump(
        {
            "employee_id": employee_id,
            "sadece_acik": only_open,
            "talepler": tickets,
        }
    )


@tool
def get_employee_entitlements(employee_id: str) -> str:
    """Calisanin sistem erisim yetkilerini (VPN, ERP, Jira vb.) listeler."""

    log(f"  [TOOL] get_employee_entitlements({employee_id!r})")

    try:
        entitlements = database.list_entitlements(employee_id)
    except Exception as error:  # noqa: BLE001
        return _failure(error, employee_id=employee_id)

    if not entitlements:
        return _dump(
            {
                "employee_id": employee_id,
                "yetkiler": [],
                "sonuc": "Bu calisan icin tanimli erisim yetkisi yok.",
            }
        )

    return _dump({"employee_id": employee_id, "yetkiler": entitlements})


@tool
def get_spare_stock(asset_type: str = "") -> str:
    """Yedek cihaz stogunu getirir. asset_type bos ise tum stok doner.

    Gecerli asset_type degerleri: dizustu, cep_telefonu, monitor.
    """

    log(f"  [TOOL] get_spare_stock({asset_type!r})")

    try:
        stock = database.list_spare_stock(asset_type or None)
    except Exception as error:  # noqa: BLE001
        return _failure(error, asset_type=asset_type)

    return _dump({"asset_type": asset_type or "tumu", "stok": stock})


@tool
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

    log(f"  [TOOL] create_ticket({employee_id!r}, {category!r}, {subject!r})")

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

    try:
        employee = database.find_employee(employee_id)

        if employee is None or employee.get("belirsiz"):
            return _dump(
                {
                    "hata": "Gecerli bir calisan kimligi olmadan talep acilamaz.",
                    "employee_id": employee_id,
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
    except Exception as error:  # noqa: BLE001
        return _failure(error, employee_id=employee_id)

    return _dump({"sonuc": "Talep olusturuldu.", "talep": ticket})


READ_TOOLS = [
    find_employee,
    get_employee_assets,
    get_asset_failure_history,
    get_employee_tickets,
    get_employee_entitlements,
    get_spare_stock,
]

WRITE_TOOLS = [create_ticket]

OPS_TOOLS = READ_TOOLS + WRITE_TOOLS
