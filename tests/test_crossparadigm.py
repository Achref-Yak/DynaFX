"""Cross-paradigm interaction tests: SD + ABM + DES.

Tests that paradigms can coexist in the same simulation and share
the unified state dict.
"""

import pytest
from dynafx.system.dsl import parse_sysd


class TestSDAndABM:
    def test_sd_stocks_and_abm_agents(self):
        m = parse_sysd(
            'Test\ndt 1\nfrom 0 to 5\n'
            'stock "resources" = 100\n'
            '  - "consumption": 10\n'
            'agent "Worker": 2\n'
            '  property "output": 0\n'
            '  rule produce: always\n'
            '    output += 5\n'
        )
        r = m.simulate()
        assert "resources" in r.values
        assert r.abm_engine is not None
        assert len(r.abm_engine.instances) == 2

    def test_abm_sees_sd_state(self):
        m = parse_sysd(
            'Test\ndt 1\nfrom 0 to 3\n'
            'stock "level" = 10\n'
            'agent \"Observer\": 1\n'
            '  property \"seen\": 0\n'
            '  rule observe: always\n'
            '    seen = level\n'
        )
        r = m.simulate()
        inst = r.abm_engine.instances[0]
        assert inst.state["seen"] == 10.0

    def test_sd_runs_normally_with_agents(self):
        m = parse_sysd(
            'Test\ndt 1\nfrom 0 to 3\n'
            'stock "S" = 100\n'
            '  - "decay": S * 0.1\n'
            'agent "Trivial": 1\n'
            '  property "x": 0\n'
        )
        r = m.simulate()
        # SD still simulates correctly
        assert r.values["S"][0] == 100.0
        assert r.values["S"][-1] < 100.0


class TestSDAndDES:
    def test_sd_stocks_and_des_queues(self):
        m = parse_sysd(
            'Test\ndt 1\nfrom 0 to 5\n'
            'stock "inventory" = 50\n'
            '  + "restock": 5\n'
            'queue "Orders": capacity 10\n'
            'resource "Warehouse": capacity 3\n'
        )
        r = m.simulate()
        assert "inventory" in r.values
        assert r.des_engine is not None
        assert "Orders" in r.des_engine.queues
        assert "Warehouse" in r.des_engine.resources

    def test_des_coexists_with_sd_flows(self):
        m = parse_sysd(
            'Test\ndt 1\nfrom 0 to 3\n'
            'stock "balance" = 1000\n'
            '  + "income": 100\n'
            '  - "expense": 50\n'
            'queue "Tickets": capacity 5\n'
        )
        r = m.simulate()
        assert r.values["balance"][0] == 1000.0
        assert r.values["balance"][-1] > 1000.0
        assert r.des_engine is not None


class TestABMAndDES:
    def test_abm_agents_with_des_queues(self):
        m = parse_sysd(
            'Test\ndt 1\nfrom 0 to 3\n'
            'queue \"ServiceQueue\": capacity 5\n'
            'resource \"Server\": capacity 2\n'
            'agent \"Customer\": 3\n'
            '  property \"served\": 0\n'
            '  rule serve: always\n'
            '    served += 1\n'
        )
        r = m.simulate()
        assert r.abm_engine is not None
        assert r.des_engine is not None
        assert len(r.abm_engine.instances) == 3


class TestAllThreeParadigms:
    def test_sd_abm_des_together(self):
        m = parse_sysd(
            'Test\ndt 1\nfrom 0 to 5\n'
            'stock "population" = 1000\n'
            '  + "birth": 10\n'
            'queue "Clinic": capacity 5\n'
            'resource "Doctor": capacity 2\n'
            'agent "Patient": 3\n'
            '  property "healthy": 1\n'
            '  rule check: always\n'
            '    healthy += 1\n'
        )
        r = m.simulate()
        assert "population" in r.values
        assert r.abm_engine is not None
        assert r.des_engine is not None
        assert r.values["population"][-1] > 1000.0

    def test_all_paradigms_independent(self):
        """Each paradigm produces output without interfering."""
        m = parse_sysd(
            'Test\ndt 1\nfrom 0 to 3\n'
            'stock "S1" = 10\n'
            '  + "f1": 1\n'
            'agent "A": 2\n'
            '  property "p": 0\n'
            '  rule r1: always\n'
            '    p += 1\n'
            'queue "Q": capacity 3\n'
            'resource "R": capacity 1\n'
        )
        r = m.simulate()
        # SD stock evolved
        assert r.values["S1"][-1] == 13.0
        # ABM agents evolved
        for inst in r.abm_engine.instances:
            assert inst.state["p"] == 3.0
        # DES queue exists
        assert "Q" in r.des_engine.queues

    def test_no_agents_no_des(self):
        """Pure SD model produces no ABM/DES engines."""
        m = parse_sysd(
            'Test\ndt 1\nfrom 0 to 3\n'
            'stock "X" = 5\n'
            '  + "dx": 1\n'
        )
        r = m.simulate()
        assert r.abm_engine is None
        assert r.des_engine is None


class TestCLIIntegration:
    def test_cli_paradigm_all(self):
        """CLI --paradigm all runs everything."""
        import subprocess, tempfile, os
        model_content = (
            'Test CLI\ndt 1\nfrom 0 to 3\n'
            'stock "S" = 10\n'
            '  + "dx": 1\n'
            'agent "A": 1\n'
            '  property "x": 0\n'
            '  rule r: always\n'
            '    x += 1\n'
            'queue "Q": capacity 5\n'
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sysd", delete=False) as f:
            f.write(model_content)
            f.flush()
            result = subprocess.run(
                ["python", "-m", "dynafx.system", "simulate", f.name,
                 "--paradigm", "all"],
                capture_output=True, text=True, timeout=30
            )
            os.unlink(f.name)
            assert result.returncode == 0
            assert "ABM:" in result.stdout
            assert "DES" in result.stdout

    def test_cli_paradigm_sd_only(self):
        """CLI --paradigm sd skips ABM/DES."""
        import subprocess, tempfile, os
        model_content = (
            'Test SD\ndt 1\nfrom 0 to 3\n'
            'stock "S" = 10\n'
            '  + "dx": 1\n'
            'agent "A": 1\n'
            '  property "x": 0\n'
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sysd", delete=False) as f:
            f.write(model_content)
            f.flush()
            result = subprocess.run(
                ["python", "-m", "dynafx.system", "simulate", f.name,
                 "--paradigm", "sd"],
                capture_output=True, text=True, timeout=30
            )
            os.unlink(f.name)
            assert result.returncode == 0
            assert "ABM" not in result.stdout
            assert "DES" not in result.stdout
