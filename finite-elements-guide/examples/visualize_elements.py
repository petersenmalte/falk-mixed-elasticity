#!/usr/bin/env python3
"""See for yourself what each finite element family glues together.

This is the runnable companion to ``finite_elements_guide.pdf``.  It builds
four finite element spaces *by hand* on a two-triangle mesh -- continuous
Lagrange, discontinuous Galerkin, Raviart-Thomas and Nedelec -- and draws what
each one does at the edge the two triangles share.

Nothing is hidden inside a framework: every basis function below is three or
four lines of numpy, written straight from its definition.  Only numpy and
matplotlib are required.  (For a cross-check against the real FEniCSx element
library, see ``basix_elements.py`` next to this file.)

Usage
-----
    python3 visualize_elements.py                 # writes element_comparison.png
    python3 visualize_elements.py --show          # opens a window instead
    python3 visualize_elements.py -o /tmp/out.png # choose the output file
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

import matplotlib.pyplot as plt

# ===========================================================================
# 1.  The mesh: two triangles sharing one vertical edge
# ===========================================================================
#
#          v2 (1,1)
#          /|\
#         / | \
#   v0 --  |  -- v3          the shared edge is v1--v2, the segment x = 1.
#  (0,.5)\ | /  (2,.5)       Its unit normal is n = (1,0) and its unit
#         \|/                tangent is t = (0,1).
#          v1 (1,0)

VERTICES = np.array([[0.0, 0.5],
                     [1.0, 0.0],
                     [1.0, 1.0],
                     [2.0, 0.5]])

# Both cells are listed counter-clockwise.
CELLS = np.array([[0, 1, 2],
                  [1, 3, 2]])

SHARED = (1, 2)          # the global vertices of the shared edge


def local_edges(cell):
    """Local edge i of a triangle is the one *opposite* local vertex i."""
    return [(cell[1], cell[2]), (cell[2], cell[0]), (cell[0], cell[1])]


def build_edge_numbering(cells):
    """Give every edge of the mesh one global index and one global direction.

    The global direction runs from the lower to the higher vertex number.  Both
    cells that share an edge therefore agree on its tangent t and its normal n
    -- which is exactly what makes a single shared degree of freedom meaningful.
    """
    numbering = {}
    for cell in cells:
        for a, b in local_edges(cell):
            key = (min(a, b), max(a, b))
            if key not in numbering:
                numbering[key] = len(numbering)
    return numbering


EDGES = build_edge_numbering(CELLS)


def edge_frame(key):
    """Unit tangent and unit normal of a global edge, plus its length."""
    a, b = key
    d = VERTICES[b] - VERTICES[a]
    length = np.linalg.norm(d)
    t = d / length
    n = np.array([t[1], -t[0]])       # t rotated by -90 degrees
    return t, n, length


# ===========================================================================
# 2.  Barycentric coordinates -- the raw material for every basis below
# ===========================================================================

def barycentric(tri):
    """Return a function x -> (lambda_0, lambda_1, lambda_2) and the gradients.

    The barycentric coordinate lambda_i is the linear function that equals 1 at
    vertex i and 0 at the other two.  These *are* the P1 Lagrange basis
    functions on the cell, and every other element here is assembled from them.
    """
    M = np.vstack([tri.T, np.ones(3)])          # rows: x, y, 1
    Minv = np.linalg.inv(M)

    def lam(x):
        return Minv @ np.array([x[0], x[1], 1.0])

    grads = Minv[:, :2]                          # row i is grad(lambda_i)
    return lam, grads


def area(tri):
    d1, d2 = tri[1] - tri[0], tri[2] - tri[0]
    return 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])


# ===========================================================================
# 3.  The four element families
# ===========================================================================

# --- (a) Lagrange P1 -------------------------------------------------------
#
# Degrees of freedom: the value at each vertex.  Neighbouring cells share the
# vertices of the edge between them, so they are forced to use the same two
# numbers there -- and two numbers determine a linear function along the edge.
# Hence the traces agree: the space sits inside H^1.

def p1_value(tri, coeffs, x):
    lam, _ = barycentric(tri)
    return float(np.dot(coeffs, lam(x)))


# --- (b) Discontinuous Galerkin P1 ----------------------------------------
#
# Exactly the same polynomials.  The only change is bookkeeping: each cell owns
# its own three numbers and shares nothing.  The space sits in L^2 and nowhere
# smaller.  (p1_value is reused verbatim -- that is the whole point.)


# --- (c) Raviart-Thomas RT_0 ----------------------------------------------
#
# On a triangle,  RT_0 = { v(x) = a + b x }, three-dimensional.  The basis
# function belonging to edge i is
#
#       phi_i(x) = sigma_i / (2 |K|) * (x - a_i),
#
# where a_i is the vertex opposite edge i and sigma_i = +-1 orients the cell's
# outward normal with the global edge normal.
#
# Two facts make this work, and both are worth checking on paper:
#   * for x on edge j (j != i) the vector x - a_i runs *along* that edge, so
#     phi_i . n_j = 0 there -- the basis function is invisible to other edges;
#   * on its own edge phi_i . n is the constant 1 / |e_i|, so the flux
#     integral over that edge is exactly 1.
# The normal component on an edge is therefore a single number, shared by both
# cells: the space sits inside H(div).

def rt0_basis(tri, cell, x):
    """The three RT_0 basis vectors of one cell, evaluated at x."""
    A = area(tri)
    centre = tri.mean(axis=0)
    out = []
    for i, (a, b) in enumerate(local_edges(cell)):
        opposite = tri[i]                        # vertex opposite edge i
        key = (min(a, b), max(a, b))
        _, n_global, _ = edge_frame(key)
        mid = 0.5 * (VERTICES[a] + VERTICES[b])
        # does the global normal point out of this cell?
        sigma = 1.0 if np.dot(n_global, mid - centre) > 0 else -1.0
        out.append(sigma / (2.0 * A) * (np.asarray(x) - opposite))
    return np.array(out)


# --- (d) Nedelec N_0, the Whitney edge element ----------------------------
#
# For the edge running from vertex a to vertex b,
#
#       phi_ab = lambda_a grad(lambda_b) - lambda_b grad(lambda_a).
#
# Its circulation along its own edge is 1 and along the other two edges is 0.
# The tangential component on an edge is a single shared number, so the space
# sits inside H(curl).  In 2D this is precisely RT_0 rotated by 90 degrees.

def n0_basis(tri, cell, x):
    """The three Nedelec (Whitney) basis vectors of one cell, evaluated at x."""
    lam, grads = barycentric(tri)
    lam_x = lam(x)
    local_of = {int(g): k for k, g in enumerate(cell)}
    out = []
    for a, b in local_edges(cell):
        lo, hi = (a, b) if a < b else (b, a)     # global direction: low -> high
        ka, kb = local_of[lo], local_of[hi]
        out.append(lam_x[ka] * grads[kb] - lam_x[kb] * grads[ka])
    return np.array(out)


# ===========================================================================
# 4.  Self-check: are these really the dual bases we claimed?
# ===========================================================================

def line_integral(f, p0, p1, direction, n_quad=200):
    """Integrate f(x) . direction along the segment p0 -> p1."""
    s = (np.arange(n_quad) + 0.5) / n_quad
    pts = p0 + np.outer(s, p1 - p0)
    length = np.linalg.norm(p1 - p0)
    return float(np.mean([np.dot(f(p), direction) for p in pts]) * length)


def check_duality(verbose=True):
    """Verify dof_j(phi_i) = delta_ij for RT_0 and for Nedelec, on both cells.

    This is the numerical statement that the degrees of freedom really are
    'flux through edge j' and 'circulation along edge j'.
    """
    ok = True
    for family, basis, which in [("RT_0     ", rt0_basis, "n"),
                                 ("Nedelec  ", n0_basis, "t")]:
        for c, cell in enumerate(CELLS):
            tri = VERTICES[cell]
            M = np.zeros((3, 3))
            for i in range(3):
                for j, (a, b) in enumerate(local_edges(cell)):
                    key = (min(a, b), max(a, b))
                    t, n, _ = edge_frame(key)
                    direction = n if which == "n" else t
                    M[j, i] = line_integral(
                        lambda x, i=i: basis(tri, cell, x)[i],
                        VERTICES[key[0]], VERTICES[key[1]], direction)
            good = np.allclose(M, np.eye(3), atol=1e-10)
            ok = ok and good
            if verbose:
                print(f"  {family} cell {c}:  dof_j(phi_i) = identity?  "
                      f"{'yes' if good else 'NO'}   (max error "
                      f"{np.abs(M - np.eye(3)).max():.2e})")
    return ok


# ===========================================================================
# 5.  Degree-of-freedom values used for the pictures
# ===========================================================================

# Continuous Lagrange: one number per *global vertex*, shared by both cells.
P1_GLOBAL = {0: 0.22, 1: 0.90, 2: 0.38, 3: 0.80}

# Discontinuous: one number per (cell, local vertex).  The two cells are given
# deliberately different values at the two shared vertices.
DG_LOCAL = {0: [0.22, 0.90, 0.38],
            1: [0.42, 0.86, 0.12]}

# RT_0 and Nedelec: one number per *global edge*.
EDGE_DOFS_RT = {(1, 2): 0.80, (0, 2): -0.25, (0, 1): 0.35,
                (2, 3): -0.40, (1, 3): 0.30}
EDGE_DOFS_N0 = {(1, 2): 0.70, (0, 2): 0.15, (0, 1): -0.35,
                (2, 3): 0.55, (1, 3): -0.42}


def cell_edge_coeffs(cell, table):
    return np.array([table[(min(a, b), max(a, b))]
                     for a, b in local_edges(cell)])


def scalar_field(c, x, discontinuous):
    tri = VERTICES[CELLS[c]]
    if discontinuous:
        coeffs = DG_LOCAL[c]
    else:
        coeffs = [P1_GLOBAL[int(g)] for g in CELLS[c]]
    return p1_value(tri, coeffs, x)


def vector_field(c, x, family):
    cell = CELLS[c]
    tri = VERTICES[cell]
    if family == "rt":
        return cell_edge_coeffs(cell, EDGE_DOFS_RT) @ rt0_basis(tri, cell, x)
    return cell_edge_coeffs(cell, EDGE_DOFS_N0) @ n0_basis(tri, cell, x)


# ===========================================================================
# 6.  Drawing
# ===========================================================================

C_H1, C_L2, C_HDIV, C_HCURL = "#2f5fa8", "#1a7f64", "#c0562a", "#7a4fa3"
C_INK, C_DIM, C_WARN = "#22242a", "#6b6f7a", "#b3243c"

LABEL_BOX = dict(facecolor="white", edgecolor="none", alpha=0.80, pad=1.2)

EDGE_SAMPLES = np.linspace(0.0, 1.0, 120)

SHARED_KEY = (min(SHARED), max(SHARED))


def edge_points():
    """Points along the shared edge, from (1,0) up to (1,1)."""
    a, b = VERTICES[SHARED[0]], VERTICES[SHARED[1]]
    return a + np.outer(EDGE_SAMPLES, b - a)


def edge_owners():
    """Map each global edge to the cells that contain it."""
    owners = {key: [] for key in EDGES}
    for c, cell in enumerate(CELLS):
        for a, b in local_edges(cell):
            owners[(min(a, b), max(a, b))].append(c)
    return owners


OWNERS = edge_owners()


def draw_mesh(ax):
    for cell in CELLS:
        ax.add_patch(plt.Polygon(VERTICES[cell], closed=True, fill=False,
                                 edgecolor=C_INK, lw=1.3, zorder=5))
    a, b = VERTICES[SHARED[0]], VERTICES[SHARED[1]]
    ax.plot([a[0], b[0]], [a[1], b[1]], color=C_WARN, lw=2.4, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(-0.45, 1.80)
    ax.axis("off")


def row_heading(ax, title, subtitle):
    ax.text(-0.30, 1.70, title, fontsize=11.5, color=C_INK, ha="left",
            va="center")
    ax.text(-0.30, 1.50, subtitle, fontsize=8.4, color=C_DIM, ha="left",
            va="center")


def panel_scalar(ax_field, ax_trace, discontinuous):
    colour = C_L2 if discontinuous else C_H1
    vmin, vmax = 0.05, 0.95

    for c, cell in enumerate(CELLS):
        tri = VERTICES[cell]
        vals = (DG_LOCAL[c] if discontinuous
                else [P1_GLOBAL[int(g)] for g in cell])
        ax_field.tripcolor(tri[:, 0], tri[:, 1], np.array([[0, 1, 2]]),
                           np.array(vals, dtype=float), shading="gouraud",
                           cmap="YlGnBu", vmin=vmin, vmax=vmax, zorder=1)
        # the degrees of freedom themselves
        centre = tri.mean(axis=0)
        mesh_centre = VERTICES.mean(axis=0)
        sideways = np.array([-1.0, 0.0]) if c == 0 else np.array([1.0, 0.0])
        for k in range(3):
            p = tri[k]
            if discontinuous:                     # pull the marker inside
                p = p + 0.14 * (centre - p) / np.linalg.norm(centre - p)
                label_at = p + 0.22 * sideways
            else:                                 # push the label out of the mesh
                out = (p - mesh_centre) / np.linalg.norm(p - mesh_centre)
                label_at = p + 0.17 * out
            ax_field.scatter([p[0]], [p[1]], s=40, color=colour,
                             edgecolor="white", linewidths=0.8, zorder=8)
            ax_field.text(label_at[0], label_at[1], f"{vals[k]:.2f}",
                          ha="center", va="center", fontsize=7.4,
                          color=colour, zorder=9, bbox=LABEL_BOX)
    draw_mesh(ax_field)

    # the two one-sided traces along the shared edge
    pts = edge_points()
    left = [scalar_field(0, p, discontinuous) for p in pts]
    right = [scalar_field(1, p, discontinuous) for p in pts]
    ax_trace.plot(left, EDGE_SAMPLES, color=C_H1, lw=2.0, label="from $K^-$")
    ax_trace.plot(right, EDGE_SAMPLES, color=C_L2, lw=2.0, ls=(0, (4, 2.6)),
                  label="from $K^+$")
    ax_trace.set_xlim(0.0, 1.0)
    ax_trace.set_ylim(0.0, 1.0)
    ax_trace.set_xlabel("value on the shared edge", fontsize=8.5)
    ax_trace.set_ylabel("position along the edge", fontsize=8.5)
    ax_trace.tick_params(labelsize=7.5)
    ax_trace.spines[["top", "right"]].set_visible(False)
    ax_trace.legend(fontsize=7.6, loc="upper right", frameon=True,
                    framealpha=0.85, edgecolor="none")

    gap = float(np.max(np.abs(np.array(left) - np.array(right))))
    ax_trace.text(0.03, 0.96, f"largest jump: {gap:.3f}",
                  transform=ax_trace.transAxes, fontsize=8,
                  color=C_WARN if gap > 1e-12 else C_H1, va="top",
                  bbox=LABEL_BOX)
    return gap


def panel_vector(ax_field, ax_trace, family):
    colour = C_HDIV if family == "rt" else C_HCURL
    table = EDGE_DOFS_RT if family == "rt" else EDGE_DOFS_N0

    for c, cell in enumerate(CELLS):
        tri = VERTICES[cell]
        pts, vecs = [], []
        for i in range(1, 5):                     # interior sample points
            for j in range(1, 5 - i):
                k = 5 - i - j
                if k >= 1:
                    p = (i * tri[0] + j * tri[1] + k * tri[2]) / 5
                    pts.append(p)
                    vecs.append(vector_field(c, p, family))
        pts, vecs = np.array(pts), np.array(vecs)
        ax_field.quiver(pts[:, 0], pts[:, 1], vecs[:, 0], vecs[:, 1],
                        color=colour, angles="xy", scale_units="xy",
                        scale=3.0, width=0.008, zorder=7)

    # the degrees of freedom: one arrow per *global* edge, across it or along it
    for key, cells_here in OWNERS.items():
        t, n, _ = edge_frame(key)
        mid = 0.5 * (VERTICES[key[0]] + VERTICES[key[1]])
        d = n if family == "rt" else t
        ax_field.annotate("", xy=mid + 0.13 * d, xytext=mid - 0.13 * d,
                          arrowprops=dict(arrowstyle="-|>", color=C_DIM,
                                          lw=1.1), zorder=9)
        if key == SHARED_KEY:
            label_at = np.array([mid[0], 1.18])   # above the shared edge
            col = C_WARN
        else:                                     # push outside the mesh
            centre = VERTICES[CELLS[cells_here[0]]].mean(axis=0)
            outward = n if np.dot(n, mid - centre) > 0 else -n
            label_at = mid + 0.26 * outward
            col = C_DIM
        ax_field.text(label_at[0], label_at[1], f"{table[key]:+.2f}",
                      ha="center", va="center", fontsize=7.4, color=col,
                      zorder=10)
    draw_mesh(ax_field)

    # normal and tangential components of both one-sided traces
    pts = edge_points()
    t, n, _ = edge_frame(SHARED)
    comps = {}
    for side, c in [("$K^-$", 0), ("$K^+$", 1)]:
        vals = np.array([vector_field(c, p, family) for p in pts])
        comps[(side, "n")] = vals @ n
        comps[(side, "t")] = vals @ t

    for (side, comp), style in [(("$K^-$", "n"), "-"), (("$K^+$", "n"), (0, (4, 2.6))),
                                (("$K^-$", "t"), "-"), (("$K^+$", "t"), (0, (4, 2.6)))]:
        col = C_HDIV if comp == "n" else C_HCURL
        ax_trace.plot(comps[(side, comp)], EDGE_SAMPLES, color=col, lw=2.0,
                      ls=style,
                      label=f"$v\\cdot {comp}$ from {side}")

    ax_trace.set_ylim(0.0, 1.0)
    ax_trace.set_xlabel("component value", fontsize=8.5)
    ax_trace.set_ylabel("position along the edge", fontsize=8.5)
    ax_trace.tick_params(labelsize=7.5)
    ax_trace.spines[["top", "right"]].set_visible(False)
    ax_trace.legend(fontsize=6.8, loc="lower right", handlelength=1.9,
                    labelspacing=0.3, frameon=True, framealpha=0.85,
                    edgecolor="none")

    jump_n = float(np.max(np.abs(comps[("$K^-$", "n")] - comps[("$K^+$", "n")])))
    jump_t = float(np.max(np.abs(comps[("$K^-$", "t")] - comps[("$K^+$", "t")])))
    ax_trace.text(0.03, 0.96,
                  f"jump in $v\\cdot n$: {jump_n:.3f}\n"
                  f"jump in $v\\cdot t$: {jump_t:.3f}",
                  transform=ax_trace.transAxes, fontsize=8, va="top",
                  color=C_INK, bbox=LABEL_BOX)
    return jump_n, jump_t


TITLES = [
    ("Continuous Lagrange $P_1$", "one value per shared vertex $\\Rightarrow$ "
     "the traces agree $\\Rightarrow$ $H^1$"),
    ("Discontinuous $P_1$", "nothing shared $\\Rightarrow$ the traces differ "
     "$\\Rightarrow$ only $L^2$"),
    ("Raviart–Thomas $RT_0$", "one flux per shared edge $\\Rightarrow$ "
     "$v\\cdot n$ agrees $\\Rightarrow$ $H(\\mathrm{div})$"),
    ("Nédélec $N_0$", "one circulation per shared edge $\\Rightarrow$ "
     "$v\\cdot t$ agrees $\\Rightarrow$ $H(\\mathrm{curl})$"),
]


def build_figure():
    fig, axes = plt.subplots(4, 2, figsize=(8.6, 12.0),
                             gridspec_kw={"width_ratios": [1.3, 1.0]})
    results = {}
    results["lagrange"] = panel_scalar(axes[0, 0], axes[0, 1], discontinuous=False)
    results["dg"] = panel_scalar(axes[1, 0], axes[1, 1], discontinuous=True)
    results["rt"] = panel_vector(axes[2, 0], axes[2, 1], "rt")
    results["nedelec"] = panel_vector(axes[3, 0], axes[3, 1], "nedelec")

    for row, (title, subtitle) in enumerate(TITLES):
        row_heading(axes[row, 0], title, subtitle)

    fig.suptitle("What each finite element family glues together",
                 fontsize=13.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig, results


# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", default=None,
                        help="where to write the figure "
                             "(default: element_comparison.png next to this file)")
    parser.add_argument("--show", action="store_true",
                        help="open an interactive window instead of writing a file")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    print("Checking that the degrees of freedom are dual to the basis "
          "functions ...")
    if not check_duality():
        raise SystemExit("duality check failed -- the bases are wrong")

    fig, results = build_figure()

    print("\nWhat happens on the shared edge:")
    print(f"  continuous P1  : largest jump in the value      "
          f"= {results['lagrange']:.2e}")
    print(f"  discontinuous  : largest jump in the value      "
          f"= {results['dg']:.3f}")
    print(f"  RT_0           : largest jump in v.n            "
          f"= {results['rt'][0]:.2e}")
    print(f"                   largest jump in v.t            "
          f"= {results['rt'][1]:.3f}")
    print(f"  Nedelec N_0    : largest jump in v.n            "
          f"= {results['nedelec'][0]:.3f}")
    print(f"                   largest jump in v.t            "
          f"= {results['nedelec'][1]:.2e}")

    if args.show:
        plt.show()
    else:
        out = args.output or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "element_comparison.png")
        fig.savefig(out, dpi=150)
        print(f"\nfigure written to {out}")


if __name__ == "__main__":
    main()
