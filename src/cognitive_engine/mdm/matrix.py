"""MDM Matrix Operations — NumPy-based dependency deduction.

Implements:
    - DomainMatrix (DSM): relationships within a single domain
    - DomainMappingMatrix (DMM): relationships between two domains
    - MultipleDomainMatrix (MDM): combines DSMs and DMMs

The core power of MDM is deducing indirect dependencies via matrix
multiplication. For example, if Function A depends on Part B, and
Part B is used in Process C, then Function A indirectly depends on
Process C. Matrix multiplication discovers this automatically.

Reference: Lindemann et al. (2008) "Structural Complexity Management"
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DomainMatrix:
    """DSM: Design Structure Matrix — relationships within a single domain.

    A square matrix where rows and columns represent elements of the
    same domain. Matrix[i][j] > 0 means element i has a relationship
    to element j with strength equal to the matrix value.

    Example:
        DSM for "Functions":
            F1  F2  F3
        F1 [ 0   1   0 ]  ← F1 depends on F2
        F2 [ 0   0   1 ]  ← F2 depends on F3
        F3 [ 1   0   0 ]  ← F3 depends on F1 (feedback loop!)
    """

    def __init__(self, domain: str, elements: list[str]):
        """Initialize a Domain Matrix.

        Args:
            domain: Domain name (e.g., "Functions", "Parts", "Failures").
            elements: List of element names in this domain.
        """
        self.domain = domain
        self.elements = list(elements)
        self.element_index: dict[str, int] = {
            name: i for i, name in enumerate(elements)
        }
        n = len(elements)
        self.matrix = np.zeros((n, n), dtype=np.float64)

    def add_relation(self, source: str, target: str, weight: float = 1.0) -> None:
        """Add a relationship from source to target element.

        Args:
            source: Source element name.
            target: Target element name.
            weight: Relationship strength (0.0 to 1.0).
        """
        i = self.element_index[source]
        j = self.element_index[target]
        self.matrix[i][j] = weight

    def get_relation(self, source: str, target: str) -> float:
        """Get the relationship weight from source to target."""
        i = self.element_index[source]
        j = self.element_index[target]
        return float(self.matrix[i][j])

    def get_dependencies(self, element: str) -> list[tuple[str, float]]:
        """Get all elements that depend on this element (incoming edges).

        Returns:
            List of (element_name, weight) tuples.
        """
        i = self.element_index[element]
        deps: list[tuple[str, float]] = []
        for j, name in enumerate(self.elements):
            if self.matrix[j][i] > 0:
                deps.append((name, float(self.matrix[j][i])))
        return deps

    def get_dependees(self, element: str) -> list[tuple[str, float]]:
        """Get all elements that this element depends on (outgoing edges).

        Returns:
            List of (element_name, weight) tuples.
        """
        i = self.element_index[element]
        deps: list[tuple[str, float]] = []
        for j, name in enumerate(self.elements):
            if self.matrix[i][j] > 0:
                deps.append((name, float(self.matrix[i][j])))
        return deps

    def find_feedback_loops(self) -> list[list[str]]:
        """Find feedback loops in the DSM.

        Returns:
            List of loops, where each loop is a list of element names.
        """
        loops: list[list[str]] = []
        n = len(self.elements)

        def _dfs(node: int, path: list[int], visited: set[int]) -> None:
            visited.add(node)
            path.append(node)
            for j in range(n):
                if self.matrix[node][j] > 0:
                    if j in path:
                        # Found a loop
                        loop_start = path.index(j)
                        loop = [self.elements[k] for k in path[loop_start:]]
                        loops.append(loop)
                    elif j not in visited:
                        _dfs(j, path, visited)
            path.pop()
            visited.discard(node)

        for i in range(n):
            _dfs(i, [], set())

        return loops

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "domain": self.domain,
            "elements": self.elements,
            "matrix": self.matrix.tolist(),
        }


class DomainMappingMatrix:
    """DMM: Domain Mapping Matrix — relationships between two domains.

    A rectangular matrix where rows represent elements of the source
    domain and columns represent elements of the target domain.
    Matrix[i][j] > 0 means source element i maps to target element j.

    Example:
        DMM from "Functions" to "Parts":
            P1  P2  P3
        F1 [ 1   0   1 ]  ← F1 uses P1 and P3
        F2 [ 0   1   0 ]  ← F2 uses P2
        F3 [ 1   1   0 ]  ← F3 uses P1 and P2
    """

    def __init__(
        self,
        source_domain: str,
        target_domain: str,
        source_elements: list[str],
        target_elements: list[str],
    ):
        """Initialize a Domain Mapping Matrix.

        Args:
            source_domain: Source domain name.
            target_domain: Target domain name.
            source_elements: List of source element names.
            target_elements: List of target element names.
        """
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.source_elements = list(source_elements)
        self.target_elements = list(target_elements)
        self.source_index: dict[str, int] = {
            name: i for i, name in enumerate(source_elements)
        }
        self.target_index: dict[str, int] = {
            name: i for i, name in enumerate(target_elements)
        }
        self.matrix = np.zeros(
            (len(source_elements), len(target_elements)), dtype=np.float64
        )

    def add_mapping(self, source: str, target: str, weight: float = 1.0) -> None:
        """Add a mapping from source to target element.

        Args:
            source: Source element name.
            target: Target element name.
            weight: Mapping strength (0.0 to 1.0).
        """
        i = self.source_index[source]
        j = self.target_index[target]
        self.matrix[i][j] = weight

    def get_mappings(self, source: str) -> list[tuple[str, float]]:
        """Get all target elements mapped from a source element."""
        i = self.source_index[source]
        mappings: list[tuple[str, float]] = []
        for j, name in enumerate(self.target_elements):
            if self.matrix[i][j] > 0:
                mappings.append((name, float(self.matrix[i][j])))
        return mappings

    def get_mapped_by(self, target: str) -> list[tuple[str, float]]:
        """Get all source elements that map to a target element."""
        j = self.target_index[target]
        mappings: list[tuple[str, float]] = []
        for i, name in enumerate(self.source_elements):
            if self.matrix[i][j] > 0:
                mappings.append((name, float(self.matrix[i][j])))
        return mappings

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "source_elements": self.source_elements,
            "target_elements": self.target_elements,
            "matrix": self.matrix.tolist(),
        }


class MultipleDomainMatrix:
    """MDM: Multiple Domain Matrix — combines DSMs and DMMs.

    The MDM is a block matrix where:
        - Diagonal blocks are DSMs (within-domain relationships)
        - Off-diagonal blocks are DMMs (between-domain relationships)

    Power of MDM: Matrix multiplication deduces indirect dependencies.

    Example:
        Domains: Functions (Fu), Parts (Pa), Failures (Fa)

        MDM block structure:
            Fu      Pa      Fa
        Fu [ DSM_Fu  DMM_FuPa  DMM_FuFa ]
        Pa [ DMM_PaFu  DSM_Pa  DMM_PaFa ]
        Fa [ DMM_FaFu  DMM_FaPa  DSM_Fa  ]

        Indirect dependencies:
        [FaFa] = [FuFa]^T · [FlFu] · [FuFl] · [FuFa]

    Reference: Lindemann et al. (2008)
    """

    def __init__(self) -> None:
        self.domains: dict[str, list[str]] = {}
        self.dsms: dict[str, DomainMatrix] = {}
        self.dmms: dict[tuple[str, str], DomainMappingMatrix] = {}

    def add_domain(self, domain: str, elements: list[str]) -> DomainMatrix:
        """Add a domain with its elements.

        Args:
            domain: Domain name.
            elements: List of element names in this domain.

        Returns:
            The created DomainMatrix.
        """
        self.domains[domain] = list(elements)
        dsm = DomainMatrix(domain, elements)
        self.dsms[domain] = dsm
        return dsm

    def get_dsm(self, domain: str) -> Optional[DomainMatrix]:
        """Get the DSM for a domain."""
        return self.dsms.get(domain)

    def add_dmm(
        self,
        source_domain: str,
        target_domain: str,
    ) -> DomainMappingMatrix:
        """Add a DMM between two domains.

        Args:
            source_domain: Source domain name.
            target_domain: Target domain name.

        Returns:
            The created DomainMappingMatrix.
        """
        key = (source_domain, target_domain)
        if key not in self.dmms:
            self.dmms[key] = DomainMappingMatrix(
                source_domain,
                target_domain,
                self.domains[source_domain],
                self.domains[target_domain],
            )
        return self.dmms[key]

    def get_dmm(
        self,
        source_domain: str,
        target_domain: str,
    ) -> Optional[DomainMappingMatrix]:
        """Get the DMM between two domains."""
        return self.dmms.get((source_domain, target_domain))

    def deduce_dependencies(
        self,
        source_domain: str,
        target_domain: str,
        max_depth: int = 3,
    ) -> Optional[np.ndarray]:
        """Deduce indirect dependencies via matrix multiplication.

        This is the core power of MDM: finding hidden failure paths
        where a small error in one component cascades through the network.

        Args:
            source_domain: Starting domain.
            target_domain: Ending domain.
            max_depth: Maximum number of intermediate hops.

        Returns:
            Matrix of indirect dependencies, or None if no path exists.
        """
        # Direct mapping
        key = (source_domain, target_domain)
        if key in self.dmms:
            result = self.dmms[key].matrix.copy()
            # Try to strengthen via indirect paths
            for depth in range(2, max_depth + 1):
                indirect = self._find_indirect_path(
                    source_domain, target_domain, depth
                )
                if indirect is not None:
                    result = np.maximum(result, indirect)
            return result

        # Try indirect paths
        for depth in range(2, max_depth + 1):
            indirect = self._find_indirect_path(
                source_domain, target_domain, depth
            )
            if indirect is not None:
                return indirect

        return None

    def _find_indirect_path(
        self,
        source_domain: str,
        target_domain: str,
        depth: int,
    ) -> Optional[np.ndarray]:
        """Find an indirect path of given depth between two domains.

        Uses breadth-first search through domain graph.
        """
        from collections import deque

        # BFS to find all paths of given depth
        queue: deque[tuple[str, np.ndarray]] = deque()
        queue.append((source_domain, np.eye(len(self.domains[source_domain]))))

        for _ in range(depth):
            next_queue: deque[tuple[str, np.ndarray]] = deque()
            while queue:
                current_domain, current_matrix = queue.popleft()
                for (src, tgt), dmm in self.dmms.items():
                    if src == current_domain:
                        # Multiply: current_matrix @ dmm.matrix
                        try:
                            result = current_matrix @ dmm.matrix
                            next_queue.append((tgt, result))
                        except ValueError:
                            # Matrix dimensions don't match
                            continue
            queue = next_queue

        # Check if we reached the target
        for domain, matrix in queue:
            if domain == target_domain:
                return matrix

        return None

    def get_full_matrix(self) -> np.ndarray:
        """Get the full MDM as a single block matrix.

        Returns:
            Square numpy array representing the complete MDM.
        """
        domain_names = list(self.domains.keys())
        n = sum(len(self.domains[d]) for d in domain_names)
        full = np.zeros((n, n), dtype=np.float64)

        offset_i = 0
        for domain_i in domain_names:
            len_i = len(self.domains[domain_i])
            offset_j = 0
            for domain_j in domain_names:
                len_j = len(self.domains[domain_j])
                if domain_i == domain_j:
                    # DSM on diagonal
                    full[offset_i:offset_i + len_i,
                         offset_j:offset_j + len_j] = self.dsms[domain_i].matrix
                else:
                    # DMM off diagonal
                    key = (domain_i, domain_j)
                    if key in self.dmms:
                        full[offset_i:offset_i + len_i,
                             offset_j:offset_j + len_j] = self.dmms[key].matrix
                offset_j += len_j
            offset_i += len_i

        return full

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "domains": self.domains,
            "dsms": {name: dsm.to_dict() for name, dsm in self.dsms.items()},
            "dmms": {
                f"{src}->{tgt}": dmm.to_dict()
                for (src, tgt), dmm in self.dmms.items()
            },
        }
