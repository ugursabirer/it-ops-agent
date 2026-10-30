"""Create and seed the IT operations SQLite database.

Run once before starting the assistant:

    python ops_mcp/create_database.py

The script is idempotent: it drops and recreates the tables, so it can be used
to reset the demo data to a known state (useful after the agent has created
tickets during a demo).

The target path comes from ``database._db_path()``, the same resolver the
application and the MCP server use, so ``IT_OPS_DB_PATH`` points all of them at
the same file. Resolving it separately here meant a configured path was seeded
in one place and read from another.
"""

from __future__ import annotations

import sqlite3
import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ops_mcp.database import _db_path  # noqa: E402

SCHEMA = """
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS assets;
DROP TABLE IF EXISTS tickets;
DROP TABLE IF EXISTS entitlements;
DROP TABLE IF EXISTS spare_stock;

CREATE TABLE employees (
    employee_id TEXT PRIMARY KEY,
    full_name   TEXT NOT NULL,
    department  TEXT NOT NULL,
    title       TEXT NOT NULL,
    email       TEXT NOT NULL,
    manager_id  TEXT
);

CREATE TABLE assets (
    asset_id      TEXT PRIMARY KEY,
    employee_id   TEXT,
    asset_type    TEXT NOT NULL,
    model         TEXT NOT NULL,
    purchase_date TEXT NOT NULL,
    warranty_end  TEXT,
    status        TEXT NOT NULL
);

CREATE TABLE tickets (
    ticket_id      TEXT PRIMARY KEY,
    employee_id    TEXT NOT NULL,
    asset_id       TEXT,
    category       TEXT NOT NULL,
    subject        TEXT NOT NULL,
    status         TEXT NOT NULL,
    priority       TEXT NOT NULL,
    assigned_group TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE entitlements (
    employee_id  TEXT NOT NULL,
    system_name  TEXT NOT NULL,
    access_level TEXT NOT NULL,
    granted_at   TEXT NOT NULL,
    last_used_at TEXT,
    PRIMARY KEY (employee_id, system_name)
);

CREATE TABLE spare_stock (
    asset_type      TEXT NOT NULL,
    model           TEXT NOT NULL,
    available_count INTEGER NOT NULL,
    PRIMARY KEY (asset_type, model)
);
"""

EMPLOYEES = [
    (
        "E-1001",
        "Ahmet Yildiz",
        "Yazilim Gelistirme",
        "Senior Developer",
        "ahmet.yildiz@ornekholding.com",
        "E-2001",
    ),
    (
        "E-1002",
        "Elif Demir",
        "Finans",
        "Finans Uzmani",
        "elif.demir@ornekholding.com",
        "E-2002",
    ),
    (
        "E-1003",
        "Mert Kaya",
        "Satis",
        "Bolge Satis Yoneticisi",
        "mert.kaya@ornekholding.com",
        "E-2002",
    ),
    (
        "E-1004",
        "Zeynep Arslan",
        "Veri Bilimi",
        "Data Scientist",
        "zeynep.arslan@ornekholding.com",
        "E-2001",
    ),
    (
        "E-1005",
        "Can Ozturk",
        "Insan Kaynaklari",
        "IK Uzmani",
        "can.ozturk@ornekholding.com",
        "E-2003",
    ),
    (
        "E-2001",
        "Burak Sen",
        "Yazilim Gelistirme",
        "Direktor",
        "burak.sen@ornekholding.com",
        None,
    ),
    (
        "E-2002",
        "Deniz Aydin",
        "Ticari",
        "Direktor",
        "deniz.aydin@ornekholding.com",
        None,
    ),
    (
        "E-2003",
        "Seda Koc",
        "Insan Kaynaklari",
        "Direktor",
        "seda.koc@ornekholding.com",
        None,
    ),
]

# purchase_date values are chosen so that the policy rules in
# data/policies/donanim_yenileme_politikasi.md produce interesting cases:
# AST-5001 and AST-5005 are past the 36-month cycle, AST-5004 is not but has a
# repeated-failure history, and AST-5003 is neither.
ASSETS = [
    (
        "AST-5001",
        "E-1001",
        "dizustu",
        "Dell Latitude 5420",
        "2023-02-15",
        "2026-02-15",
        "aktif",
    ),
    (
        "AST-5002",
        "E-1001",
        "cep_telefonu",
        "iPhone 13",
        "2025-06-01",
        "2027-06-01",
        "aktif",
    ),
    (
        "AST-5003",
        "E-1002",
        "dizustu",
        "MacBook Air M2",
        "2025-03-10",
        "2028-03-10",
        "aktif",
    ),
    (
        "AST-5004",
        "E-1003",
        "dizustu",
        "Lenovo ThinkPad T14",
        "2024-09-20",
        "2027-09-20",
        "aktif",
    ),
    (
        "AST-5005",
        "E-1004",
        "dizustu",
        "MacBook Pro 14 M1",
        "2022-11-05",
        "2025-11-05",
        "aktif",
    ),
    (
        "AST-5006",
        "E-1005",
        "dizustu",
        "HP EliteBook 840",
        "2024-01-15",
        "2027-01-15",
        "aktif",
    ),
    (
        "AST-5007",
        None,
        "dizustu",
        "Dell Latitude 5440",
        "2026-03-01",
        "2029-03-01",
        "stokta",
    ),
]

TICKETS = [
    (
        "TKT-9001",
        "E-1003",
        "AST-5004",
        "ariza",
        "Dizustu bilgisayar aniden kapaniyor",
        "kapandi",
        "yuksek",
        "BT-Donanim",
        "2026-06-18",
    ),
    (
        "TKT-9002",
        "E-1003",
        "AST-5004",
        "ariza",
        "Cihaz asiri isiniyor ve kapaniyor",
        "kapandi",
        "yuksek",
        "BT-Donanim",
        "2026-09-02",
    ),
    (
        "TKT-9003",
        "E-1001",
        None,
        "talep",
        "Ek monitor talebi",
        "acik",
        "normal",
        "BT-Donanim",
        "2026-10-20",
    ),
    (
        "TKT-9004",
        "E-1002",
        None,
        "bilgi",
        "Parola degistirme adimlari",
        "kapandi",
        "dusuk",
        "BT-Altyapi",
        "2026-08-11",
    ),
    (
        "TKT-9005",
        "E-1004",
        "AST-5005",
        "ariza",
        "Klavye tuslari calismiyor",
        "acik",
        "normal",
        "BT-Donanim",
        "2026-10-25",
    ),
]

ENTITLEMENTS = [
    ("E-1001", "vpn", "gelistirici", "2024-01-10", "2026-10-29"),
    ("E-1001", "jira", "standart", "2024-01-10", "2026-10-30"),
    ("E-1002", "vpn", "standart", "2025-03-12", "2026-10-28"),
    ("E-1002", "erp", "standart", "2025-03-12", "2026-10-30"),
    ("E-1004", "vpn", "gelistirici", "2023-02-01", "2026-10-27"),
    ("E-1004", "veri_ambari", "standart", "2023-02-01", "2026-10-29"),
    ("E-1005", "vpn", "standart", "2024-02-01", "2026-05-02"),
]

SPARE_STOCK = [
    ("dizustu", "Dell Latitude 5440", 4),
    ("dizustu", "MacBook Air M3", 2),
    ("cep_telefonu", "iPhone 15", 6),
    ("monitor", "Dell U2723QE", 0),
]


def create(db_path: Path | None = None) -> Path:
    """Create and seed the database, returning the path that was written."""

    target = db_path or _db_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)

        connection.executemany(
            "INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?)", EMPLOYEES
        )
        connection.executemany(
            "INSERT INTO assets VALUES (?, ?, ?, ?, ?, ?, ?)", ASSETS
        )
        connection.executemany(
            "INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", TICKETS
        )
        connection.executemany(
            "INSERT INTO entitlements VALUES (?, ?, ?, ?, ?)", ENTITLEMENTS
        )
        connection.executemany("INSERT INTO spare_stock VALUES (?, ?, ?)", SPARE_STOCK)

        connection.commit()

    return target


def main() -> None:
    target = create()

    print(f"[OK] SQLite veritabani olusturuldu: {target}")
    print(
        f"     {len(EMPLOYEES)} calisan, {len(ASSETS)} varlik, "
        f"{len(TICKETS)} talep, {len(ENTITLEMENTS)} yetki, "
        f"{len(SPARE_STOCK)} stok kaydi."
    )


if __name__ == "__main__":
    main()
