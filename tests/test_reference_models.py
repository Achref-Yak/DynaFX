"""Tests for reference .sysd models (F1-F3)."""
import os
import pytest

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")


@pytest.fixture
def model_path():
    return MODELS_DIR


def test_vmi_parses_and_simulates(model_path):
    from dynafx.dynamics.dsl import parse_sysd_file

    path = os.path.join(model_path, "vmi.sysd")
    assert os.path.exists(path)
    m = parse_sysd_file(path)
    r = m.simulate(method="euler")
    assert "Supplier_Inventory" in r["values"]
    assert "Retailer_Inventory" in r["values"]
    assert all(v >= 0 for vals in r["values"].values() for v in vals)


def test_vmi_retailer_does_not_deplete(model_path):
    from dynafx.dynamics.dsl import parse_sysd_file

    m = parse_sysd_file(os.path.join(model_path, "vmi.sysd"))
    r = m.simulate(method="euler", params={"base_demand": 500})
    min_retail = min(r["values"]["Retailer_Inventory"])
    assert min_retail >= 0, f"Retailer depleted to {min_retail}"


def test_reverse_logistics_parses_and_simulates(model_path):
    from dynafx.dynamics.dsl import parse_sysd_file

    path = os.path.join(model_path, "reverse_logistics.sysd")
    assert os.path.exists(path)
    m = parse_sysd_file(path)
    r = m.simulate(method="euler")
    assert "Returned_Products" in r["values"]
    assert "Refurbished_Goods" in r["values"]
    assert "Scrap" in r["values"]
    assert all(v >= 0 for vals in r["values"].values() for v in vals)


def test_reverse_logistics_mass_balance(model_path):
    from dynafx.dynamics.dsl import parse_sysd_file

    m = parse_sysd_file(os.path.join(model_path, "reverse_logistics.sysd"))
    r = m.simulate(method="euler")
    refurb = r["values"]["Refurbished_Goods"][-1]
    scrap = r["values"]["Scrap"][-1]
    # Total items that entered the system = return_rate * t_end
    total_input = 100.0 * 200.0
    total_in_stocks = r["values"]["Returned_Products"][-1] + refurb + scrap
    mass_error = abs(total_input - total_in_stocks)
    # CONVEY pipeline holds at most return_rate * inspection_delay items at end
    assert mass_error < 100.0 * 3.0 + 50, f"Mass error: {mass_error}"


def test_cold_chain_parses_and_simulates(model_path):
    from dynafx.dynamics.dsl import parse_sysd_file

    path = os.path.join(model_path, "cold_chain.sysd")
    assert os.path.exists(path)
    m = parse_sysd_file(path)
    r = m.simulate(method="euler")
    assert "Cold_Inventory" in r["values"]
    assert "Spoilage" in r["values"]
    assert all(v >= 0 for vals in r["values"].values() for v in vals)


def test_cold_chain_spoilage_accumulates(model_path):
    from dynafx.dynamics.dsl import parse_sysd_file

    m = parse_sysd_file(os.path.join(model_path, "cold_chain.sysd"))
    r = m.simulate(method="euler")
    assert r["values"]["Spoilage"][-1] > 0


def test_cold_chain_inventory_declines_with_spoilage(model_path):
    from dynafx.dynamics.dsl import parse_sysd_file

    m = parse_sysd_file(os.path.join(model_path, "cold_chain.sysd"))
    r = m.simulate(method="euler", params={"temp_deviation": 2})
    inv = r["values"]["Cold_Inventory"]
    assert inv[-1] < inv[0], "Inventory should decline with high temp deviation"


def test_all_models_parse_and_simulate(model_path):
    from dynafx.dynamics.dsl import parse_sysd_file

    models = ["vmi.sysd", "reverse_logistics.sysd", "cold_chain.sysd"]
    for fn in models:
        path = os.path.join(model_path, fn)
        assert os.path.exists(path), f"Missing model: {path}"
        m = parse_sysd_file(path)
        r = m.simulate(method="euler")
        assert len(r["values"]) > 0, f"No values for {fn}"
