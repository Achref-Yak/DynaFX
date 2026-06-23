"""Templates — reusable model builders for common structural patterns.

Each template is a class that constructs a SysdModel directly via the
Python API (no .sysd file needed). Every template can also be expressed
as a .sysd file using include/submodel for the DSL side.
"""

from cognitive_engine.templates.signal_chain import SignalChain

__all__ = [
    "SignalChain",
]
