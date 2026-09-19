"""Adaptive mesh refinement driven by the a posteriori estimator (Petersen
2022, Section 7): Solve -> Estimate -> Mark -> Refine, using Doerfler
(bulk-chasing) marking.
"""
import numpy as np
from dolfinx import mesh as dmesh


def doerfler_mark(eta2: np.ndarray, theta: float = 0.5) -> np.ndarray:
    """Mark the smallest set of cells whose eta_T^2 sum to >= theta * total."""
    order = np.argsort(eta2)[::-1]
    cumulative = np.cumsum(eta2[order])
    threshold = theta * eta2.sum()
    cutoff = int(np.searchsorted(cumulative, threshold)) + 1
    return np.sort(order[:cutoff])


def refine_marked(domain: dmesh.Mesh, marked_cells: np.ndarray) -> dmesh.Mesh:
    """Bisect every edge of each marked cell and return the refined mesh."""
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, tdim - 1)
    cell_to_edge = domain.topology.connectivity(tdim, tdim - 1)
    marked_edges = np.unique(
        np.concatenate([cell_to_edge.links(int(c)) for c in marked_cells])
    ).astype(np.int32)
    result = dmesh.refine(domain, marked_edges)
    return result[0] if isinstance(result, tuple) else result
