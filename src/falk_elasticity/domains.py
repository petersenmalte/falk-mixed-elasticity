"""Domain geometries beyond the unit square, Petersen (2022) Section 7.

dolfinx.mesh has no built-in non-convex domain, so this builds a structured
triangulation directly from explicit vertex/cell arrays.
"""
import basix
import basix.ufl
import numpy as np
import ufl
from dolfinx import mesh as dmesh


def create_l_shaped_mesh(comm, n: int) -> dmesh.Mesh:
    """Omega = (-1,1)x(0,1) U (-1,0)x(-1,0) (Petersen 2022, Section 7.2): the
    square (-1,1)^2 with its bottom-right quadrant removed, giving a
    re-entrant corner at the origin. Built as a uniform h=1/n grid over the
    bounding box with the excluded quadrant's cells dropped.
    """
    h = 1.0 / n
    ncells = 2 * n

    vertex_index: dict[tuple[int, int], int] = {}
    points: list[tuple[float, float]] = []
    cells: list[list[int]] = []

    def vertex(i: int, j: int) -> int:
        key = (i, j)
        if key not in vertex_index:
            vertex_index[key] = len(points)
            points.append((-1.0 + i * h, -1.0 + j * h))
        return vertex_index[key]

    for i in range(ncells):
        for j in range(ncells):
            cx, cy = -1.0 + (i + 0.5) * h, -1.0 + (j + 0.5) * h
            if cx > 0.0 and cy < 0.0:
                continue  # bottom-right quadrant: not part of the domain
            v00, v10 = vertex(i, j), vertex(i + 1, j)
            v01, v11 = vertex(i, j + 1), vertex(i + 1, j + 1)
            cells.append([v00, v10, v11])
            cells.append([v00, v11, v01])

    x = np.array(points, dtype=np.float64)
    c = np.array(cells, dtype=np.int64)
    coord_element = basix.ufl.element("Lagrange", basix.CellType.triangle, 1, shape=(2,))
    return dmesh.create_mesh(comm, c, ufl.Mesh(coord_element), x)
