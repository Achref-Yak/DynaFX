"""Tests for ensemble sensitivity simulation."""

import math
from cognitive_engine.system.dsl import parse_sysd


def test_simulate_ensemble_basic():
    m = parse_sysd("""
model 'SM'
  dt 1
  from 0 to 10
  stock 'S': 100
    - 'Out': 0.1 * S
""")
    ens = m.simulate_ensemble(params={"dt": (0.5, 1.5)}, n=5)
    assert "mean" in ens
    assert "std" in ens
    assert "p5" in ens
    assert "p95" in ens
    assert "trajectories" in ens
    assert len(ens["trajectories"]) == 5
    assert ens["stocks"] == ["S"]


def test_ensemble_mean_reasonable():
    m = parse_sysd("""
model 'EM'
  dt 1
  from 0 to 5
  stock 'X': 0
    + 'In': 10
""")
    ens = m.simulate_ensemble(params={}, n=3)
    assert abs(ens["mean"]["X"][-1] - 50.0) < 1e-9


def test_ensemble_with_uncertain_param():
    m = parse_sysd("""
model 'UP'
  dt 1
  from 0 to 10
  stock 'X': 0
    + 'In': rate
  aux 'rate': growth * dt
""")
    ens = m.simulate_ensemble(params={"growth": (0.5, 1.5)}, n=10)
    final_mean = ens["mean"]["X"][-1]
    assert 0 < final_mean < 150


def test_ensemble_with_normal_dist():
    m = parse_sysd("""
model 'ND'
  dt 1
  from 0 to 5
  stock 'S': 100
    - 'Out': S * decay
  aux 'decay': 0.1
""")
    param_spec: tuple[float, float, str] = (0.05, 0.15, "normal")
    ens = m.simulate_ensemble(params={"decay": param_spec}, n=10, seed=42)
    assert len(ens["trajectories"]) == 10


def test_ensemble_with_lognormal_dist():
    m = parse_sysd("""
model 'LN'
  dt 1
  from 0 to 5
  stock 'S': 100
    - 'Out': S * rate
  aux 'rate': 0.1
""")
    param_spec: tuple[float, float, str] = (0.05, 0.2, "lognormal")
    ens = m.simulate_ensemble(params={"rate": param_spec}, n=10, seed=42)
    assert len(ens["trajectories"]) == 10


def test_ensemble_seed_reproducibility():
    m = parse_sysd("""
model 'RP'
  dt 1
  from 0 to 5
  stock 'X': 0
    + 'In': rate
  aux 'rate': growth
""")
    ens1 = m.simulate_ensemble(params={"growth": (0.5, 1.5)}, n=5, seed=42)
    ens2 = m.simulate_ensemble(params={"growth": (0.5, 1.5)}, n=5, seed=42)
    assert ens1["mean"]["X"][-1] == ens2["mean"]["X"][-1]
