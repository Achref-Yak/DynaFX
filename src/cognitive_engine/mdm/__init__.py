"""Multiple Domain Matrix (MDM) module.

NumPy-based matrix operations for deducing indirect dependencies
across multiple domains.

Based on:
    - DSM (Design Structure Matrix) theory
    - DMM (Domain Mapping Matrix) theory
    - MDM (Multiple Domain Matrix) theory
    - Lindemann et al. (2008) "Structural Complexity Management"
"""

from cognitive_engine.mdm.matrix import (
    DomainMatrix,
    DomainMappingMatrix,
    MultipleDomainMatrix,
)

__all__ = [
    "DomainMatrix",
    "DomainMappingMatrix",
    "MultipleDomainMatrix",
]
