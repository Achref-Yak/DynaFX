"""Deprecated import shim — use ``dynafx.patterns.SignalChain`` instead."""

import warnings as _warnings

from dynafx.patterns.signal_chain import SignalChain  # noqa: F401

_warnings.warn(
    "Import SignalChain from dynafx.patterns instead of dynafx.dynamics",
    DeprecationWarning, stacklevel=2,
)
