"""Tests for CSV import/export, interpolation, merge, and fill strategies."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TMPDIR", "/tmp")

import pytest
from dynafx.dynamics.dsl import parse_sysd


SIMPLE_MODEL_SRC = """
model 'CSVTest'
  dt 1
  from 0 to 10
  stock 'X': 0
    + 'in': external_data
"""


@pytest.fixture
def model():
    return parse_sysd(SIMPLE_MODEL_SRC)


# ═══════════════════════════════════════════════════════════════
# CSV Import
# ═══════════════════════════════════════════════════════════════

class TestCSVImportBasic:
    def test_import_basic(self):
        src = '''
model 'Test'
  stock X: 100
  dt 1
  from 0 to 10
'''
        m = parse_sysd(src)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,demand,price\n")
            f.write("0,100,10\n")
            f.write("1,110,11\n")
            f.write("2,120,12\n")
            f.write("3,130,13\n")
            path = f.name
        try:
            data = m.import_data(path)
            assert "demand" in data
            assert "price" in data
            assert len(data["demand"]) == 4
            assert data["demand"][0] == (0, 100)
            assert data["demand"][3] == (3, 130)
        finally:
            os.unlink(path)

    def test_import_with_spaces_in_names(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,Raw Demand,Unit Cost\n")
            f.write("0,100,5.5\n")
            f.write("1,110,5.6\n")
            path = f.name
        try:
            m = parse_sysd("model 'T'\n  stock X: 0\n  dt 1")
            data = m.import_data(path)
            assert "Raw Demand" in data
            assert "Unit Cost" in data
        finally:
            os.unlink(path)

    def test_import_empty_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,value\n")
            path = f.name
        try:
            m = parse_sysd("model 'T'\n  stock X: 0\n  dt 1")
            data = m.import_data(path)
            assert "value" in data
            assert len(data["value"]) == 0
        finally:
            os.unlink(path)

    def test_import_non_numeric_skipped(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,value\n")
            f.write("0,100\n")
            f.write("1,abc\n")
            f.write("2,200\n")
            path = f.name
        try:
            m = parse_sysd("model 'T'\n  stock X: 0\n  dt 1")
            data = m.import_data(path, fill="forward")
            assert len(data["value"]) == 3
            assert data["value"] == [(0.0, 100.0), (1.0, 100.0), (2.0, 200.0)]
        finally:
            os.unlink(path)

    def test_float_times(self, model):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("t,external_data\n")
            f.write("0,100\n")
            f.write("1,200\n")
            f.write("2,300\n")
            path = f.name
        try:
            data = model.import_data(path)
            assert "external_data" in data
            assert data["external_data"] == [(0.0, 100.0), (1.0, 200.0), (2.0, 300.0)]
            assert model._imported_data is data
        finally:
            os.unlink(path)

    def test_datetime_auto_detect(self, model):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,external_data\n")
            f.write("2025-01-01T00:00:00,100\n")
            f.write("2025-01-01T01:00:00,200\n")
            f.write("2025-01-01T02:00:00,300\n")
            path = f.name
        try:
            data = model.import_data(path)
            assert "external_data" in data
            assert len(data["external_data"]) == 3
            times = [p[0] for p in data["external_data"]]
            assert times[0] == 0.0
            assert times[1] == 1.0
            assert times[2] == 2.0
            vals = [p[1] for p in data["external_data"]]
            assert vals == [100.0, 200.0, 300.0]
        finally:
            os.unlink(path)

    def test_datetime_days_unit(self, model):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ts,val\n")
            f.write("2025-01-01T00:00:00,100\n")
            f.write("2025-01-02T00:00:00,200\n")
            path = f.name
        try:
            data = model.import_data(path, time_unit="days")
            times = [p[0] for p in data["val"]]
            assert times[0] == 0.0
            assert times[1] == 1.0
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════
# Fill strategies
# ═══════════════════════════════════════════════════════════════

class TestFillStrategies:
    def test_forward_fill(self, model):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("t,val\n")
            f.write("0,100\n")
            f.write("1,\n")
            f.write("2,300\n")
            path = f.name
        try:
            data = model.import_data(path, fill="forward")
            assert len(data["val"]) == 3
            assert data["val"][0] == (0.0, 100.0)
            assert data["val"][1] == (1.0, 100.0)
            assert data["val"][2] == (2.0, 300.0)
        finally:
            os.unlink(path)

    def test_interpolate_fill(self, model):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("t,val\n")
            f.write("0,100\n")
            f.write("2,\n")
            f.write("4,300\n")
            path = f.name
        try:
            data = model.import_data(path, fill="interpolate")
            assert len(data["val"]) == 3
            assert data["val"][0] == (0.0, 100.0)
            assert abs(data["val"][1][0] - 2.0) < 0.01
            assert abs(data["val"][1][1] - 200.0) < 0.01
            assert data["val"][2] == (4.0, 300.0)
        finally:
            os.unlink(path)

    def test_zero_fill(self, model):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("t,val\n")
            f.write("0,100\n")
            f.write("1,\n")
            path = f.name
        try:
            data = model.import_data(path, fill="zero")
            assert len(data["val"]) == 2
            assert data["val"][0] == (0.0, 100.0)
            assert data["val"][1] == (1.0, 0.0)
        finally:
            os.unlink(path)

    def test_merge_data(self, model):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f1:
            f1.write("t,a\n0,10\n1,20\n")
            p1 = f1.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f2:
            f2.write("t,b\n0,100\n1,200\n")
            p2 = f2.name
        try:
            merged = model.merge_data([p1, p2])
            assert "a" in merged
            assert "b" in merged
            assert merged["a"] == [(0.0, 10.0), (1.0, 20.0)]
            assert merged["b"] == [(0.0, 100.0), (1.0, 200.0)]
        finally:
            os.unlink(p1)
            os.unlink(p2)

    def test_merge_data_duplicate_raises(self, model):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f1:
            f1.write("t,a\n0,1\n")
            p1 = f1.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f2:
            f2.write("t,a\n0,2\n")
            p2 = f2.name
        try:
            with pytest.raises(ValueError, match="Duplicated variable"):
                model.merge_data([p1, p2])
        finally:
            os.unlink(p1)
            os.unlink(p2)


# ═══════════════════════════════════════════════════════════════
# Interpolation
# ═══════════════════════════════════════════════════════════════

class TestInterpolation:
    def test_interpolate_exact(self):
        m = parse_sysd("model 'T'\n  stock X: 0\n  dt 1")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,value\n")
            f.write("0,100\n")
            f.write("1,200\n")
            f.write("2,300\n")
            path = f.name
        try:
            m.import_data(path)
            interp = m.get_imported_interpolator("value")
            assert interp(0) == 100
            assert interp(1) == 200
            assert interp(2) == 300
        finally:
            os.unlink(path)

    def test_interpolate_between(self):
        m = parse_sysd("model 'T'\n  stock X: 0\n  dt 1")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,value\n")
            f.write("0,100\n")
            f.write("2,300\n")
            path = f.name
        try:
            m.import_data(path)
            interp = m.get_imported_interpolator("value")
            assert interp(1) == 200
        finally:
            os.unlink(path)

    def test_interpolate_extrapolate_before(self):
        m = parse_sysd("model 'T'\n  stock X: 0\n  dt 1")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,value\n")
            f.write("5,100\n")
            f.write("10,200\n")
            path = f.name
        try:
            m.import_data(path)
            interp = m.get_imported_interpolator("value")
            assert interp(0) == 100
        finally:
            os.unlink(path)

    def test_interpolate_extrapolate_after(self):
        m = parse_sysd("model 'T'\n  stock X: 0\n  dt 1")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,value\n")
            f.write("0,100\n")
            f.write("5,200\n")
            path = f.name
        try:
            m.import_data(path)
            interp = m.get_imported_interpolator("value")
            assert interp(10) == 200
        finally:
            os.unlink(path)

    def test_interpolate_missing_variable(self):
        m = parse_sysd("model 'T'\n  stock X: 0\n  dt 1")
        interp = m.get_imported_interpolator("nonexistent")
        assert interp(5) == 0.0

    def test_interpolator_returns_interpolated_values(self, model):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("t,val\n")
            f.write("0,0\n")
            f.write("2,100\n")
            f.write("4,200\n")
            path = f.name
        try:
            model.import_data(path)
            interp = model.get_imported_interpolator("val")
            assert interp(0.0) == 0.0
            assert interp(1.0) == 50.0
            assert interp(2.0) == 100.0
            assert interp(3.0) == 150.0
        finally:
            os.unlink(path)

    def test_interpolator_clamps_boundaries(self, model):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("t,val\n")
            f.write("5,100\n")
            f.write("10,200\n")
            path = f.name
        try:
            model.import_data(path)
            interp = model.get_imported_interpolator("val")
            assert interp(0.0) == 100.0
            assert interp(12.0) == 200.0
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════
# CSV Export
# ═══════════════════════════════════════════════════════════════

class TestCSVExport:
    def test_export_basic(self):
        src = '''
model 'Test'
  stock X: 100
  stock Y: 50
  aux growth: X * 0.1
  + growth
  dt 1
  from 0 to 5
'''
        m = parse_sysd(src)
        r = m.simulate()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            r.export_results(path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 7
            assert lines[0].strip().startswith("time")
            assert "X" in lines[0]
            assert "Y" in lines[0]
        finally:
            os.unlink(path)

    def test_export_import_roundtrip(self):
        src = '''
model 'Test'
  stock X: 100
  aux growth: X * 0.1
  + growth
  dt 1
  from 0 to 10
'''
        m = parse_sysd(src)
        r = m.simulate()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            export_path = f.name
        try:
            r.export_results(export_path)
            m2 = parse_sysd("model 'T'\n  stock Y: 0\n  dt 1")
            data = m2.import_data(export_path)
            assert "X" in data
            assert len(data["X"]) == 11
            assert data["X"][0][1] == 100
        finally:
            os.unlink(export_path)


# ═══════════════════════════════════════════════════════════════
# Integration with simulation
# ═══════════════════════════════════════════════════════════════

class TestImportIntegration:
    def test_imported_data_available(self):
        m = parse_sysd("model 'T'\n  stock X: 0\n  dt 1")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,forcing\n")
            f.write("0,10\n")
            f.write("5,20\n")
            path = f.name
        try:
            data = m.import_data(path)
            interp = m.get_imported_interpolator("forcing")
            assert interp(0) == 10
            assert interp(5) == 20
            assert interp(2.5) == 15
        finally:
            os.unlink(path)
