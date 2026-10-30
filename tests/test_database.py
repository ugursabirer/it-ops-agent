"""The operational data layer, including the values it derives for the agent."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _db(ops_database):
    """Point every test in this module at the throwaway database."""

    return ops_database


def test_find_employee_by_id_is_case_insensitive():
    from ops_mcp import database

    assert database.find_employee("e-1003")["full_name"] == "Mert Kaya"


def test_find_employee_by_partial_name_resolves_manager():
    from ops_mcp import database

    employee = database.find_employee("Mert")

    assert employee["employee_id"] == "E-1003"
    assert employee["manager_name"] == "Deniz Aydin"


def test_find_employee_reports_ambiguity_instead_of_guessing():
    from ops_mcp import database

    # Several seeded employees share this substring in their name.
    result = database.find_employee("a")

    assert result["belirsiz"] is True
    assert len(result["adaylar"]) > 1


def test_find_employee_returns_none_when_missing():
    from ops_mcp import database

    assert database.find_employee("E-9999") is None


def test_assets_carry_derived_age_and_warranty():
    from ops_mcp import database

    assets = database.list_employee_assets("E-1003")

    assert len(assets) == 1

    asset = assets[0]

    assert asset["asset_id"] == "AST-5004"
    # The model must never have to compute these itself.
    assert isinstance(asset["yas_ay"], int)
    assert asset["yas_ay"] > 0
    assert asset["garanti_durumu"] in {"devam_ediyor", "bitti", "yok"}


def test_expired_warranty_is_reported_as_finished():
    from ops_mcp import database

    asset = database.get_asset("AST-5001")

    assert asset["garanti_durumu"] == "bitti"


def test_failure_history_counts_only_the_time_window():
    from ops_mcp import database

    result = database.count_recent_failures("AST-5004", months=6)

    assert result["ariza_sayisi"] == 2
    assert {ticket["ticket_id"] for ticket in result["arizalar"]} == {
        "TKT-9001",
        "TKT-9002",
    }


def test_unknown_asset_is_an_error_not_zero_failures():
    """The regression this guards against changed a decision's outcome.

    An unverified id returned ``ariza_sayisi: 0``, which is indistinguishable
    from a healthy device, so a guessed id silently flipped the answer.
    """

    from ops_mcp import database

    result = database.count_recent_failures("LAPTOP-E-1003-001")

    assert "hata" in result
    assert "ariza_sayisi" not in result


def test_employee_without_entitlements_returns_empty_list():
    from ops_mcp import database

    assert database.list_entitlements("E-1003") == []


def test_spare_stock_can_be_filtered_by_type():
    from ops_mcp import database

    laptops = database.list_spare_stock("dizustu")

    assert laptops
    assert all(item["asset_type"] == "dizustu" for item in laptops)


def test_insert_ticket_allocates_the_next_id():
    from ops_mcp import database

    ticket = database.insert_ticket(
        employee_id="E-1001",
        category="talep",
        subject="Test talebi",
        priority="normal",
        assigned_group="BT-Donanim",
    )

    assert ticket["ticket_id"].startswith("TKT-")
    assert ticket["status"] == "acik"

    tickets = database.list_tickets("E-1001", only_open=True)

    assert ticket["ticket_id"] in {item["ticket_id"] for item in tickets}
