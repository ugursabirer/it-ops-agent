"""SQLite access layer for the IT operations data.

This module is the single place that talks to the database. It is imported by
two different consumers:

* ``ops_mcp/server.py`` exposes these functions as MCP tools.
* ``tools/ops_tools.py`` wraps them as plain LangChain tools, which is the
  fallback path used when MCP is disabled or unreachable.

Derived values that a small language model should never have to compute itself
(asset age in months, warranty state, failure counts inside a time window) are
calculated here in Python so the agent only has to reason about the result.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from typing import Any

_DEFAULT_DB = Path(__file__).resolve().parent / "it_ops.db"


def _db_path() -> Path:
    """Resolve the database path, honouring IT_OPS_DB_PATH when set."""

    raw = os.getenv("IT_OPS_DB_PATH")

    if not raw:
        return _DEFAULT_DB

    path = Path(raw)

    if path.is_absolute():
        return path

    return Path(__file__).resolve().parent.parent / path


def connect() -> sqlite3.Connection:
    path = _db_path()

    if not path.exists():
        raise FileNotFoundError(
            f"Veritabani bulunamadi: {path}. "
            "Once 'python ops_mcp/create_database.py' komutunu calistirin."
        )

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row

    return connection


def _rows(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with closing(connect()) as connection:
        cursor = connection.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def _months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def _parse(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


# ---------------------------------------------------------------- employees


def find_employee(query: str) -> dict[str, Any] | None:
    """Find one employee by id (E-1001) or by a part of their full name."""

    needle = query.strip()

    exact = _rows(
        "SELECT * FROM employees WHERE UPPER(employee_id) = UPPER(?)",
        (needle,),
    )

    if exact:
        return _with_manager(exact[0])

    fuzzy = _rows(
        "SELECT * FROM employees WHERE full_name LIKE ? COLLATE NOCASE",
        (f"%{needle}%",),
    )

    if len(fuzzy) == 1:
        return _with_manager(fuzzy[0])

    if len(fuzzy) > 1:
        return {
            "belirsiz": True,
            "mesaj": "Birden fazla calisan eslesti, kimligi netlestirin.",
            "adaylar": [
                {"employee_id": row["employee_id"], "full_name": row["full_name"]}
                for row in fuzzy
            ],
        }

    return None


def _with_manager(employee: dict[str, Any]) -> dict[str, Any]:
    manager_id = employee.get("manager_id")

    if manager_id:
        managers = _rows(
            "SELECT full_name FROM employees WHERE employee_id = ?",
            (manager_id,),
        )
        employee["manager_name"] = managers[0]["full_name"] if managers else None

    return employee


# ------------------------------------------------------------------- assets


def list_employee_assets(employee_id: str) -> list[dict[str, Any]]:
    """Assets currently assigned to an employee, enriched with age/warranty."""

    assets = _rows(
        "SELECT * FROM assets WHERE UPPER(employee_id) = UPPER(?) "
        "ORDER BY asset_type, purchase_date",
        (employee_id,),
    )

    return [_enrich_asset(asset) for asset in assets]


def get_asset(asset_id: str) -> dict[str, Any] | None:
    assets = _rows("SELECT * FROM assets WHERE UPPER(asset_id) = UPPER(?)", (asset_id,))

    if not assets:
        return None

    return _enrich_asset(assets[0])


def _enrich_asset(asset: dict[str, Any]) -> dict[str, Any]:
    today = date.today()

    purchase = _parse(asset.get("purchase_date"))
    warranty_end = _parse(asset.get("warranty_end"))

    asset["yas_ay"] = _months_between(purchase, today) if purchase else None
    asset["garanti_durumu"] = (
        "yok"
        if warranty_end is None
        else ("devam_ediyor" if warranty_end >= today else "bitti")
    )

    return asset


def count_recent_failures(asset_id: str, months: int = 6) -> dict[str, Any]:
    """Count hardware-failure tickets raised for an asset in a time window.

    The asset is verified first. Without that check an unknown asset id would
    return zero failures, which reads exactly like a healthy device - and a
    model that guesses an id instead of reading it from ``get_employee_assets``
    would be told its guess was fine. An unknown id has to be an error.
    """

    if get_asset(asset_id) is None:
        return {
            "asset_id": asset_id,
            "hata": "Boyle bir cihaz kaydi yok, ariza gecmisi sorgulanamadi.",
            "ipucu": (
                "Cihaz kimligini tahmin etmeyin. Once get_employee_assets "
                "cagirip donen asset_id degerini kullanin."
            ),
        }

    today = date.today()
    tickets = _rows(
        "SELECT ticket_id, subject, status, created_at FROM tickets "
        "WHERE UPPER(asset_id) = UPPER(?) AND category = 'ariza' "
        "ORDER BY created_at DESC",
        (asset_id,),
    )

    recent = [
        ticket
        for ticket in tickets
        if (created := _parse(ticket["created_at"]))
        and _months_between(created, today) <= months
    ]

    return {
        "asset_id": asset_id,
        "pencere_ay": months,
        "ariza_sayisi": len(recent),
        "arizalar": recent,
    }


# ------------------------------------------------------------------ tickets


def list_tickets(employee_id: str, only_open: bool = False) -> list[dict[str, Any]]:
    query = "SELECT * FROM tickets WHERE UPPER(employee_id) = UPPER(?)"
    params: tuple[Any, ...] = (employee_id,)

    if only_open:
        query += " AND status = 'acik'"

    query += " ORDER BY created_at DESC"

    return _rows(query, params)


def insert_ticket(
    employee_id: str,
    category: str,
    subject: str,
    priority: str,
    assigned_group: str,
    asset_id: str | None = None,
) -> dict[str, Any]:
    """Create a new ticket. This is the only write operation in the system."""

    with closing(connect()) as connection:
        last = connection.execute(
            "SELECT ticket_id FROM tickets ORDER BY ticket_id DESC LIMIT 1"
        ).fetchone()

        next_number = int(last["ticket_id"].split("-")[1]) + 1 if last else 9001
        ticket_id = f"TKT-{next_number}"

        connection.execute(
            "INSERT INTO tickets (ticket_id, employee_id, asset_id, category, "
            "subject, status, priority, assigned_group, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'acik', ?, ?, ?)",
            (
                ticket_id,
                employee_id.upper(),
                asset_id.upper() if asset_id else None,
                category,
                subject,
                priority,
                assigned_group,
                date.today().isoformat(),
            ),
        )
        connection.commit()

    return {
        "ticket_id": ticket_id,
        "employee_id": employee_id.upper(),
        "asset_id": asset_id,
        "category": category,
        "subject": subject,
        "status": "acik",
        "priority": priority,
        "assigned_group": assigned_group,
        "created_at": date.today().isoformat(),
    }


# ------------------------------------------------------------- entitlements


def list_entitlements(employee_id: str) -> list[dict[str, Any]]:
    return _rows(
        "SELECT system_name, access_level, granted_at, last_used_at "
        "FROM entitlements WHERE UPPER(employee_id) = UPPER(?) "
        "ORDER BY system_name",
        (employee_id,),
    )


# --------------------------------------------------------------------- stock


def list_spare_stock(asset_type: str | None = None) -> list[dict[str, Any]]:
    if asset_type:
        return _rows(
            "SELECT * FROM spare_stock WHERE asset_type = ? COLLATE NOCASE "
            "ORDER BY available_count DESC",
            (asset_type,),
        )

    return _rows("SELECT * FROM spare_stock ORDER BY asset_type, model")
