"""Tests for CSV import/export — Phase 7."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TMPDIR", "/tmp")

from cognitive_engine.system.dsl import parse_sysd


# ═══════════════════════════════════════════════════════════════
# CSV Import
# ═══════════════════════════════════════════════════════════════

class TestCSVImport:
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
            # Non-numeric row becomes None → forward-filled to 100
            assert data["value"] == [(0.0, 100.0), (1.0, 100.0), (2.0, 200.0)]
        finally:
            os.unlink(path)


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
            assert interp(1) == 200  # midpoint
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
            assert interp(0) == 100  # clamps to first value
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
            assert interp(10) == 200  # clamps to last value
        finally:
            os.unlink(path)

    def test_interpolate_missing_variable(self):
        m = parse_sysd("model 'T'\n  stock X: 0\n  dt 1")
        interp = m.get_imported_interpolator("nonexistent")
        assert interp(5) == 0.0  # returns 0 for missing data


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
            assert len(lines) == 7  # header + 6 rows
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
            # Import the exported data
            m2 = parse_sysd("model 'T'\n  stock Y: 0\n  dt 1")
            data = m2.import_data(export_path)
            assert "X" in data
            assert len(data["X"]) == 11  # 0 to 10 inclusive
            # Values should match
            assert data["X"][0][1] == 100  # initial value
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
            assert interp(2.5) == 15  # interpolated
        finally:
            os.unlink(path)
