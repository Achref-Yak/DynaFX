"""Tests for timestamp-aware CSV import and data pipeline."""
import pytest
import os
import tempfile
from cognitive_engine.system.dsl import parse_sysd


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


class TestCSVImport:
    def test_float_times(self, model):
        """Float time column is parsed correctly."""
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
        """Datetime column is auto-detected and converted to float hours."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,external_data\n")
            f.write("2025-01-01T00:00:00,100\n")
            f.write("2025-01-01T01:00:00,200\n")
            f.write("2025-01-01T02:00:00,300\n")
            path = f.name

        try:
            data = model.import_data(path)
            assert "external_data" in data
            # First timestamp is reference (t=0), second is 1 hour later
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
        """time_unit='days' converts timestamps to days."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("ts,val\n")
            f.write("2025-01-01T00:00:00,100\n")
            f.write("2025-01-02T00:00:00,200\n")
            path = f.name

        try:
            data = model.import_data(path, time_unit="days")
            times = [p[0] for p in data["val"]]
            assert times[0] == 0.0
            assert times[1] == 1.0  # 1 day = 1.0
        finally:
            os.unlink(path)

    def test_interpolator_returns_interpolated_values(self, model):
        """get_imported_interpolator returns linear interpolation."""
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
        """Interpolator clamps at boundaries."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("t,val\n")
            f.write("5,100\n")
            f.write("10,200\n")
            path = f.name

        try:
            model.import_data(path)
            interp = model.get_imported_interpolator("val")
            assert interp(0.0) == 100.0  # before first → clamp
            assert interp(12.0) == 200.0  # after last → clamp
        finally:
            os.unlink(path)

    def test_forward_fill(self, model):
        """Forward fill propagates last value."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("t,val\n")
            f.write("0,100\n")
            f.write("1,\n")  # empty → None → forward-filled
            f.write("2,300\n")
            path = f.name

        try:
            data = model.import_data(path, fill="forward")
            assert len(data["val"]) == 3
            assert data["val"][0] == (0.0, 100.0)
            assert data["val"][1] == (1.0, 100.0)  # forward-filled
            assert data["val"][2] == (2.0, 300.0)
        finally:
            os.unlink(path)

    def test_interpolate_fill(self, model):
        """Interpolate fill uses linear interpolation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("t,val\n")
            f.write("0,100\n")
            f.write("2,\n")  # missing → None → interpolated
            f.write("4,300\n")
            path = f.name

        try:
            data = model.import_data(path, fill="interpolate")
            assert len(data["val"]) == 3
            assert data["val"][0] == (0.0, 100.0)
            assert abs(data["val"][1][0] - 2.0) < 0.01
            assert abs(data["val"][1][1] - 200.0) < 0.01  # linear: half way
            assert data["val"][2] == (4.0, 300.0)
        finally:
            os.unlink(path)

    def test_zero_fill(self, model):
        """Zero fill sets missing values to 0.0."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("t,val\n")
            f.write("0,100\n")
            f.write("1,\n")
            path = f.name

        try:
            data = model.import_data(path, fill="zero")
            assert len(data["val"]) == 2
            assert data["val"][0] == (0.0, 100.0)
            assert data["val"][1] == (1.0, 0.0)  # zero-filled
        finally:
            os.unlink(path)

    def test_merge_data(self, model):
        """merge_data combines multiple CSV files."""
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
        """merge_data raises on duplicate variable names."""
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
