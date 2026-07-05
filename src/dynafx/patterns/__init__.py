"""Pattern library — reusable cross-paradigm simulation factories.

Each pattern is a Python class whose ``.build(params)`` method returns a
wired ``SysdModel`` ready to simulate.
"""

from dynafx.patterns.disruption_cascade import DisruptionCascade
from dynafx.patterns.signal_chain import SignalChain

__all__ = ["DisruptionCascade", "SignalChain"]
