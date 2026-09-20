#!/usr/bin/env python3
"""Ask the real FEniCSx element library the questions this guide asks.

``visualize_elements.py`` builds four finite element spaces by hand.  This
script checks that story against Basix -- the element library that FEniCSx and
DOLFINx actually use -- in two ways:

1.  It prints, for each element family, how many degrees of freedom sit on
    vertices, on edges and strictly inside the cell, together with the Sobolev
    space Basix reports.  That table is the central claim of the guide in
    machine-readable form: *where the dofs live determines what is continuous*.

2.  It verifies that the lowest-order Raviart-Thomas and Nedelec spaces built
    by hand in ``visualize_elements.py`` span exactly the same polynomials that
    Basix calls ``RT`` degree 1 and ``N1E`` degree 1.

Only ``fenics-basix`` is needed -- ``pip install fenics-basix``.  The full
DOLFINx stack (PETSc, MPI) is *not* required.

Usage
-----
    python3 basix_elements.py
"""

from __future__ import annotations

import sys

import numpy as np

try:
    import basix
except ImportError:                                    # pragma: no cover
    sys.exit("This script needs Basix.  Install it with:\n"
             "    pip install fenics-basix")

import visualize_elements as demo


# Basix reports one of these for every element; it is exactly the classification
# the guide is built around.
CONTINUITY = {
    "H1": "the whole value",
    "HDiv": "the normal component",
    "HCurl": "the tangential component",
    "L2": "nothing",
}

FAMILIES = [
    ("Lagrange P1", basix.ElementFamily.P, 1, False),
    ("Lagrange P2", basix.ElementFamily.P, 2, False),
    ("Lagrange P3", basix.ElementFamily.P, 3, False),
    ("discontinuous P1", basix.ElementFamily.P, 1, True),
    ("RT (lowest)", basix.ElementFamily.RT, 1, False),
    ("RT (next)", basix.ElementFamily.RT, 2, False),
    ("BDM (lowest)", basix.ElementFamily.BDM, 1, False),
    ("BDM (next)", basix.ElementFamily.BDM, 2, False),
    ("Nedelec 1st kind", basix.ElementFamily.N1E, 1, False),
    ("Nedelec 2nd kind", basix.ElementFamily.N2E, 1, False),
    ("Crouzeix-Raviart", basix.ElementFamily.CR, 1, False),
]


def dof_layout_table():
    """Print where the degrees of freedom sit for each family."""
    print("Element degrees of freedom on a triangle, according to Basix")
    print("(Basix counts degrees from 1, so 'RT degree 1' is the element the")
    print(" guide calls RT_0.  The elements are the same.)\n")

    head = (f"{'element':<19}{'dim':>4}{'vert':>6}{'edge':>6}{'cell':>6}"
            f"   {'space':<7}{'continuous across an edge'}")
    print(head)
    print("-" * len(head))

    for label, family, degree, disc in FAMILIES:
        # Lagrange above degree 2 has to say where its nodes go; equispaced is
        # the variant drawn in the guide's degree-of-freedom pictures.  The
        # other families do not accept a Lagrange variant at all.
        extra = ({"lagrange_variant": basix.LagrangeVariant.equispaced}
                 if family is basix.ElementFamily.P else {})
        e = basix.create_element(family, basix.CellType.triangle, degree,
                                 discontinuous=disc, **extra)
        per_vertex, per_edge, per_cell = (e.num_entity_dofs[0][0],
                                          e.num_entity_dofs[1][0],
                                          e.num_entity_dofs[2][0])
        space = e.sobolev_space.name
        print(f"{label:<19}{e.dim:>4}{per_vertex:>6}{per_edge:>6}{per_cell:>6}"
              f"   {space:<7}{CONTINUITY.get(space, '(see the guide)')}")

    print("\nRead the table by rows: an element whose dofs all sit strictly")
    print("inside the cell shares nothing with its neighbours and lands in L2;")
    print("dofs on edges buy one component; dofs on vertices buy the value.")
    print("\nOne row needs a caveat.  Crouzeix-Raviart does share a dof per")
    print("edge, but a single number cannot pin down a linear trace, so the")
    print("two cells agree only at the edge midpoint.  That is not enough for")
    print("H1, and L2 is the largest space Basix can honestly report -- the")
    print("element is nonconforming, exactly as section 8.1 of the guide says.")


def reference_samples(n=6):
    """Sample points strictly inside the Basix reference triangle."""
    pts = []
    for i in range(1, n):
        for j in range(1, n - i):
            pts.append([i / n, j / n])
    return np.array(pts)


def hand_built_matrix(basis_fn, pts):
    """Rows = basis functions, columns = their components at every sample point."""
    tri = basix.geometry(basix.CellType.triangle)
    # visualize_elements works with global vertex/cell tables, so temporarily
    # point it at the Basix reference triangle.
    saved_v, saved_c = demo.VERTICES, demo.CELLS
    demo.VERTICES = np.asarray(tri, dtype=float)
    demo.CELLS = np.array([[0, 1, 2]])
    try:
        rows = [np.concatenate([basis_fn(demo.VERTICES, demo.CELLS[0], p)[i]
                                for p in pts])
                for i in range(3)]
    finally:
        demo.VERTICES, demo.CELLS = saved_v, saved_c
    return np.array(rows)


def basix_matrix(family, degree, pts):
    e = basix.create_element(family, basix.CellType.triangle, degree)
    tab = e.tabulate(0, pts)[0]            # (npoints, ndofs, value_size)
    return np.array([np.concatenate([tab[p, i, :] for p in range(len(pts))])
                     for i in range(e.dim)])


def same_span(A, B, tol=1e-10):
    """Do the rows of A and of B span the same subspace?"""
    ra = np.linalg.matrix_rank(A, tol=tol)
    rb = np.linalg.matrix_rank(B, tol=tol)
    rboth = np.linalg.matrix_rank(np.vstack([A, B]), tol=tol)
    return ra == rb == rboth, (ra, rb, rboth)


def cross_check():
    """Confirm the hand-built RT_0 and N_0 spaces are the Basix ones."""
    print("\n\nCross-check against the hand-built bases in "
          "visualize_elements.py")
    print("-" * 62)
    pts = reference_samples()

    for label, basis_fn, family in [
            ("RT_0    ", demo.rt0_basis, basix.ElementFamily.RT),
            ("Nedelec ", demo.n0_basis, basix.ElementFamily.N1E)]:
        A = hand_built_matrix(basis_fn, pts)
        B = basix_matrix(family, 1, pts)
        ok, ranks = same_span(A, B)
        print(f"  {label}: hand-built space == Basix space?  "
              f"{'yes' if ok else 'NO'}   "
              f"(ranks {ranks[0]}, {ranks[1]}, stacked {ranks[2]})")
        if not ok:
            return False

    print("\nSame span, so the two constructions describe the same finite")
    print("element space -- the bases differ only by how each library")
    print("normalises and orients its degrees of freedom.")
    return True


def main():
    dof_layout_table()
    if not cross_check():
        raise SystemExit("cross-check failed")


if __name__ == "__main__":
    main()
