"""Generate every figure used by ``finite_elements_guide.tex``.

Run ``python3 make_figures.py`` from any directory; the PDFs are written next
to this file.  Only numpy and matplotlib are required.

Every diagram that shows inter-element behaviour reuses the same two-triangle
mesh so that the element families can be compared directly: two triangles
meeting along one vertical edge, with normal n = (1,0) and tangent t = (0,1)
on that edge.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
import matplotlib.tri as mtri

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# House style
# --------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
    }
)

# One colour per Sobolev space, used consistently in every figure and keyed to
# the same colours in the LaTeX source.
C_H1 = "#2f5fa8"       # H^1      - Lagrange
C_HCURL = "#7a4fa3"    # H(curl)  - Nedelec
C_HDIV = "#c0562a"     # H(div)   - Raviart-Thomas / BDM
C_L2 = "#1a7f64"       # L^2      - discontinuous Galerkin
C_CR = "#8a6d1f"       # Crouzeix-Raviart (nonconforming)
C_INK = "#22242a"
C_DIM = "#6b6f7a"
C_GREY = "#8d93a1"
C_LINE = "#b9bdc7"
C_FILL = "#f2f0eb"
C_FILL2 = "#eceef4"
C_WARN = "#b3243c"

WHITE_BOX = dict(facecolor="white", edgecolor="none", pad=1.4, alpha=0.88)

# --------------------------------------------------------------------------
# The canonical two-triangle mesh
# --------------------------------------------------------------------------

V_L = np.array([0.0, 0.5])   # apex of the left triangle
V_P = np.array([1.0, 0.0])   # bottom of the shared edge
V_Q = np.array([1.0, 1.0])   # top of the shared edge
V_R = np.array([2.0, 0.5])   # apex of the right triangle

T_LEFT = np.array([V_L, V_P, V_Q])
T_RIGHT = np.array([V_P, V_R, V_Q])

NORMAL = np.array([1.0, 0.0])
TANGENT = np.array([0.0, 1.0])


def save(fig, name):
    path = os.path.join(HERE, name + ".pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.relpath(path, HERE))


def clean_axes(ax, xlim=None, ylim=None):
    """Equal aspect, no frame, and anchored north so sibling titles line up."""
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.axis("off")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)


def triangle(ax, verts, edge=C_INK, face=None, lw=1.5, ls="-", z=2, alpha=1.0):
    ax.add_patch(
        Polygon(
            verts,
            closed=True,
            fill=face is not None,
            facecolor=face if face else "none",
            edgecolor=edge,
            linewidth=lw,
            linestyle=ls,
            zorder=z,
            alpha=alpha,
            joinstyle="round",
        )
    )


def arrow(ax, start, end, color=C_INK, lw=1.4, z=6, mut=9, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            tuple(start),
            tuple(end),
            arrowstyle="-|>",
            mutation_scale=mut,
            color=color,
            lw=lw,
            linestyle=ls,
            zorder=z,
            shrinkA=0,
            shrinkB=0,
        )
    )


def interior_points(verts, n=5):
    """Barycentric sample points strictly inside a triangle."""
    pts = []
    for i in range(1, n):
        for j in range(1, n - i):
            k = n - i - j
            if k >= 1:
                pts.append((i * verts[0] + j * verts[1] + k * verts[2]) / n)
    return np.array(pts)


# --------------------------------------------------------------------------
# Degree-of-freedom glyphs (a small reference triangle with markers)
# --------------------------------------------------------------------------

GLYPH = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.88]])


def glyph_triangle(ax, origin=(0.0, 0.0), scale=1.0, color=C_INK, dashed=False, lw=1.4):
    verts = GLYPH * scale + np.asarray(origin)
    triangle(ax, verts, edge=color, lw=lw, ls=(0, (3, 2.4)) if dashed else "-")
    return verts


def dof_point(ax, p, color, size=30, z=8, open_marker=False):
    ax.scatter(
        [p[0]],
        [p[1]],
        s=size,
        facecolor="white" if open_marker else color,
        edgecolor=color,
        linewidths=1.2,
        zorder=z,
    )


def outward_normal(verts, idx):
    a, b = verts[idx], verts[(idx + 1) % 3]
    mid = 0.5 * (a + b)
    e = b - a
    nrm = np.array([e[1], -e[0]])
    nrm = nrm / np.linalg.norm(nrm)
    if np.dot(nrm, mid - verts.mean(axis=0)) < 0:
        nrm = -nrm
    return mid, nrm


def dof_normal(ax, verts, idx, color, length=0.21, z=8):
    """Arrow through the midpoint of edge ``idx``, pointing outward."""
    mid, nrm = outward_normal(verts, idx)
    arrow(ax, mid - 0.45 * length * nrm, mid + 0.75 * length * nrm,
          color=color, lw=1.3, z=z)


def dof_tangent(ax, verts, idx, color, frac=0.34, z=8):
    """Arrow along edge ``idx`` (counter-clockwise traversal)."""
    a, b = verts[idx], verts[(idx + 1) % 3]
    mid = 0.5 * (a + b)
    e = (b - a) / np.linalg.norm(b - a)
    half = frac * np.linalg.norm(b - a) * 0.5
    arrow(ax, mid - half * e, mid + half * e, color=color, lw=1.3, z=z)


def draw_glyph(ax, origin, scale, colour, kind, dot=20, nlen=0.10, tfrac=0.42):
    """Draw one of the standard degree-of-freedom glyphs."""
    v = glyph_triangle(ax, origin=origin, scale=scale, color=colour,
                       dashed=(kind == "dg"), lw=1.2)
    if kind == "lagrange":
        for p in v:
            dof_point(ax, p, colour, size=dot)
    elif kind == "curl":
        for i in range(3):
            dof_tangent(ax, v, i, colour, frac=tfrac)
    elif kind == "div":
        for i in range(3):
            dof_normal(ax, v, i, colour, length=nlen)
    elif kind == "dg":
        dof_point(ax, v.mean(axis=0), colour, size=dot + 2)
    return v


# --------------------------------------------------------------------------
# A hand-rolled oblique projection, used instead of mplot3d so that the
# surface pictures stay fully under control.
# --------------------------------------------------------------------------

AXO_DX, AXO_DY, AXO_SZ = 0.40, 0.34, 0.78


def axo(x, y, z):
    return np.array([x + AXO_DX * y, AXO_SZ * z + AXO_DY * y])


def axo_tri(verts, vals):
    return np.array([axo(p[0], p[1], v) for p, v in zip(verts, vals)])


# ==========================================================================
# Figure 1 - a mesh, and the two-triangle patch used throughout
# ==========================================================================

def fig_mesh():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.95))

    # (a) an unstructured mesh of a square domain
    ax = axes[0]
    rng = np.random.default_rng(7)
    xs, ys = [], []
    n = 6
    for i in range(n + 1):
        for j in range(n + 1):
            x, y = i / n, j / n
            if 0 < i < n and 0 < j < n:
                x += rng.uniform(-0.055, 0.055)
                y += rng.uniform(-0.055, 0.055)
            xs.append(x)
            ys.append(y)
    tri = mtri.Triangulation(np.array(xs), np.array(ys))
    ax.triplot(tri, color=C_LINE, lw=0.9, zorder=1)

    hv = np.array([[tri.x[k], tri.y[k]] for k in tri.triangles[38]])
    triangle(ax, hv, edge=C_HDIV, face="#f4dfd2", lw=1.8, z=3)
    cen = hv.mean(axis=0)
    ax.text(cen[0], cen[1], "$K$", ha="center", va="center", color=C_HDIV,
            fontsize=10, zorder=5)
    for p in hv:
        dof_point(ax, p, C_INK, size=16, z=6)

    mid, nrm = outward_normal(hv, 0)
    arrow(ax, mid, mid + 0.17 * nrm, color=C_INK, lw=1.2)
    ax.text(*(mid + 0.245 * nrm), "$n$", ha="center", va="center", fontsize=9.5)

    ax.annotate("vertex", xy=hv[2], xytext=(hv[2][0] - 0.46, hv[2][1] + 0.16),
                fontsize=8.6, color=C_DIM, bbox=WHITE_BOX, va="center",
                arrowprops=dict(arrowstyle="-", color=C_DIM, lw=0.7))
    ax.annotate("edge (facet)", xy=mid, xytext=(mid[0] - 0.10, mid[1] - 0.34),
                fontsize=8.6, color=C_DIM, bbox=WHITE_BOX, ha="center",
                arrowprops=dict(arrowstyle="-", color=C_DIM, lw=0.7))

    clean_axes(ax, (-0.62, 1.14), (-0.42, 1.14))
    ax.set_title("(a)  a mesh of the domain $\\Omega$", fontsize=9.5, pad=5)

    # (b) the canonical patch
    ax = axes[1]
    triangle(ax, T_LEFT, edge=C_INK, face=C_FILL, lw=1.6)
    triangle(ax, T_RIGHT, edge=C_INK, face=C_FILL2, lw=1.6)
    ax.plot([V_P[0], V_Q[0]], [V_P[1], V_Q[1]], color=C_WARN, lw=2.6, zorder=4)

    for p, lab, off in [
        (V_L, "$a_1$", (-0.17, 0.0)),
        (V_P, "$a_2$", (-0.13, -0.10)),
        (V_Q, "$a_3$", (0.0, 0.14)),
        (V_R, "$a_4$", (0.17, 0.0)),
    ]:
        dof_point(ax, p, C_INK, size=22, z=6)
        ax.text(p[0] + off[0], p[1] + off[1], lab, ha="center", va="center", fontsize=9.5)

    ax.text(0.48, 0.5, "$K^-$", ha="center", va="center", fontsize=10.5)
    ax.text(1.52, 0.5, "$K^+$", ha="center", va="center", fontsize=10.5)

    arrow(ax, (1.0, 0.30), (1.27, 0.30), color=C_INK, lw=1.5)
    ax.text(1.34, 0.30, "$n$", color=C_INK, fontsize=10, va="center")
    arrow(ax, (1.0, 0.62), (1.0, 0.90), color=C_INK, lw=1.5)
    ax.text(1.07, 0.82, "$t$", color=C_INK, fontsize=10, va="center")
    ax.text(1.0, -0.33, "shared edge $F$", color=C_WARN, fontsize=9, ha="center")

    clean_axes(ax, (-0.34, 2.34), (-0.50, 1.26))
    ax.set_title("(b)  the patch used in every later figure", fontsize=9.5, pad=5)

    fig.subplots_adjust(wspace=0.05)
    save(fig, "fig_mesh")


# ==========================================================================
# Figure 2 - P1 degrees of freedom and the hat basis function
# ==========================================================================

def fig_p1_dofs():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))

    ax = axes[0]
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.42, 0.9]])
    triangle(ax, verts, edge=C_H1, face="#e6ecf7", lw=1.7)
    for p, lab, off in zip(verts,
                           ["$u(a_1)$", "$u(a_2)$", "$u(a_3)$"],
                           [(-0.17, -0.11), (0.18, -0.11), (0.02, 0.16)]):
        dof_point(ax, p, C_H1, size=44, z=8)
        ax.text(p[0] + off[0], p[1] + off[1], lab, ha="center", va="center",
                fontsize=9.2, color=C_H1)
    ax.text(0.47, 0.33, "$u|_K \\in P_1(K)$\n$u = c_0 + c_1x + c_2y$",
            ha="center", va="center", fontsize=9, color=C_INK)
    clean_axes(ax, (-0.34, 1.36), (-0.30, 1.18))
    ax.set_title("(a)  three values fix a linear function", fontsize=9.5, pad=6)

    # (b) the hat basis function, drawn in the oblique projection
    ax = axes[1]
    ang = np.linspace(0, 2 * np.pi, 7)[:-1]
    outer = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    centre = np.array([0.0, 0.0])

    base = np.array([axo(p[0], p[1], 0.0) for p in outer])
    triangle_base = Polygon(base, closed=True, facecolor="#f4f5f8",
                            edgecolor=C_LINE, lw=1.0, zorder=1)
    ax.add_patch(triangle_base)

    order = np.argsort(-outer[:, 1])           # draw far faces first
    apex = axo(0.0, 0.0, 1.0)
    for i in order:
        j = (i + 1) % 6
        face = np.array([apex,
                         axo(outer[i][0], outer[i][1], 0.0),
                         axo(outer[j][0], outer[j][1], 0.0)])
        ax.add_patch(Polygon(face, closed=True, facecolor="#c9d6ee",
                             edgecolor=C_H1, lw=0.9, zorder=3, alpha=0.96))

    stem_base = axo(centre[0], centre[1], 0.0)
    ax.plot([stem_base[0], apex[0]], [stem_base[1], apex[1]],
            color=C_H1, lw=1.0, ls=":", zorder=4)
    dof_point(ax, apex, C_H1, size=28, z=6)
    ax.text(apex[0] + 0.10, apex[1], "$\\varphi_i = 1$", fontsize=9,
            color=C_H1, va="center")
    ax.text(0.0, -0.62, "$\\varphi_i = 0$ outside the shaded patch",
            fontsize=8.6, color=C_DIM, ha="center")

    clean_axes(ax, (-1.55, 2.05), (-0.85, 1.30))
    ax.set_title("(b)  the hat function $\\varphi_i$", fontsize=9.5, pad=6)

    fig.subplots_adjust(wspace=0.05)
    save(fig, "fig_p1_dofs")


# ==========================================================================
# Figures 3 and 4 - Lagrange continuity vs. a DG jump
# ==========================================================================

def _surface_panel(ax, vals_left, vals_right, show_gap):
    base_quad = np.array([axo(*V_L, 0.0), axo(*V_P, 0.0),
                          axo(*V_R, 0.0), axo(*V_Q, 0.0)])
    ax.add_patch(Polygon(base_quad, closed=True, facecolor="#f4f5f8",
                         edgecolor=C_LINE, lw=1.0, zorder=1))

    # vertical stems at the four mesh vertices
    for p, v in [(V_L, vals_left[0]), (V_P, vals_left[1]),
                 (V_Q, vals_left[2]), (V_R, vals_right[1])]:
        b, t = axo(*p, 0.0), axo(*p, v)
        ax.plot([b[0], t[0]], [b[1], t[1]], color=C_LINE, lw=0.9,
                ls=(0, (2, 2)), zorder=2)

    top_l = axo_tri(T_LEFT, vals_left)
    top_r = axo_tri(T_RIGHT, vals_right)
    ax.add_patch(Polygon(top_l, closed=True, facecolor="#c9d6ee",
                         edgecolor=C_H1, lw=1.4, zorder=4, alpha=0.97))
    ax.add_patch(Polygon(top_r, closed=True, facecolor="#cfe3dc",
                         edgecolor=C_L2, lw=1.4, zorder=4, alpha=0.97))

    if show_gap:
        gap = np.array([axo(*V_P, vals_left[1]), axo(*V_Q, vals_left[2]),
                        axo(*V_Q, vals_right[2]), axo(*V_P, vals_right[0])])
        ax.add_patch(Polygon(gap, closed=True, facecolor=C_WARN, alpha=0.20,
                             edgecolor=C_WARN, lw=1.2, zorder=6))
        ax.annotate("a gap opens up\nalong $F$", xy=gap.mean(axis=0),
                    xytext=(2.55, 1.30), color=C_WARN, fontsize=8.6,
                    ha="center", va="center", zorder=9,
                    arrowprops=dict(arrowstyle="-", color=C_WARN, lw=0.8))
    else:
        a, b = axo(*V_P, vals_left[1]), axo(*V_Q, vals_left[2])
        ax.plot([a[0], b[0]], [a[1], b[1]], color=C_WARN, lw=2.2, zorder=7)
        ax.annotate("one shared trace\nalong $F$", xy=0.5 * (a + b),
                    xytext=(2.55, 1.30), color=C_WARN, fontsize=8.6,
                    ha="center", va="center", zorder=9,
                    arrowprops=dict(arrowstyle="-", color=C_WARN, lw=0.8))

    ax.text(0.30, -0.22, "$K^-$", fontsize=9.6, ha="center", color=C_DIM)
    ax.text(1.72, -0.22, "$K^+$", fontsize=9.6, ha="center", color=C_DIM)
    clean_axes(ax, (-0.30, 3.05), (-0.42, 1.55))


def _profile_panel(ax, left_edge_value, right_edge_value, uL, uR, title):
    """Cross-section along the horizontal line y = 0.5."""
    xs_l = np.linspace(0.0, 1.0, 60)
    xs_r = np.linspace(1.0, 2.0, 60)
    ax.plot(xs_l, np.interp(xs_l, [0.0, 1.0], [uL, left_edge_value]),
            color=C_H1, lw=2.0, label="on $K^-$")
    ax.plot(xs_r, np.interp(xs_r, [1.0, 2.0], [right_edge_value, uR]),
            color=C_L2, lw=2.0, label="on $K^+$")
    ax.axvline(1.0, color=C_WARN, lw=1.1, ls=(0, (4, 2.5)))
    ax.scatter([1.0], [left_edge_value], color=C_H1, s=26, zorder=6)
    ax.scatter([1.0], [right_edge_value], color=C_L2, s=26, zorder=6)
    if abs(left_edge_value - right_edge_value) > 1e-9:
        ax.annotate("", xy=(1.0, left_edge_value), xytext=(1.0, right_edge_value),
                    arrowprops=dict(arrowstyle="<->", color=C_WARN, lw=1.3))
        ax.text(1.10, 0.5 * (left_edge_value + right_edge_value),
                "jump $[u]$", color=C_WARN, fontsize=9, va="center",
                ha="left", bbox=WHITE_BOX)
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("$x$ along the cut $y = 0.5$", fontsize=9)
    ax.set_ylabel("$u_h$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8.2, frameon=False, loc="upper left")
    ax.set_title(title, fontsize=9.5, pad=6)


def fig_lagrange_continuity():
    uL, uP, uQ, uR = 0.22, 0.90, 0.38, 0.78
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    _surface_panel(axes[0], [uL, uP, uQ], [uP, uR, uQ], show_gap=False)
    axes[0].set_title("(a)  one continuous surface", fontsize=9.5, pad=6)
    _profile_panel(axes[1], 0.5 * (uP + uQ), 0.5 * (uP + uQ), uL, uR,
                   "(b)  the two traces agree on $F$")
    fig.subplots_adjust(wspace=0.30)
    save(fig, "fig_lagrange_continuity")


def fig_dg_jump():
    uL, uP, uQ = 0.22, 0.90, 0.38
    vP, vR, vQ = 0.40, 0.88, 0.12
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    _surface_panel(axes[0], [uL, uP, uQ], [vP, vR, vQ], show_gap=True)
    axes[0].set_title("(a)  two independent surfaces", fontsize=9.5, pad=6)
    _profile_panel(axes[1], 0.5 * (uP + uQ), 0.5 * (vP + vQ), uL, vR,
                   "(b)  the traces disagree on $F$")
    fig.subplots_adjust(wspace=0.30)
    save(fig, "fig_dg_jump")


# ==========================================================================
# Vector fields on the patch: RT (normal continuity), Nedelec (tangential)
# ==========================================================================

def rt_field(a, b):
    """Lowest-order Raviart-Thomas shape: v(x) = a + b x."""
    return lambda p: np.asarray(a, dtype=float) + b * np.asarray(p, dtype=float)


def nedelec_field(a, b):
    """Lowest-order Nedelec shape in 2D: v(x) = a + b x^perp."""
    def f(p):
        p = np.asarray(p, dtype=float)
        return np.asarray(a, dtype=float) + b * np.array([-p[1], p[0]])
    return f


# left / right coefficients, chosen so that exactly one component matches
RT_LEFT = rt_field((0.20, -0.50), 0.60)        # v.n = 0.80 on the edge
RT_RIGHT = rt_field((0.50, 0.60), 0.30)        # v.n = 0.80 on the edge
NED_LEFT = nedelec_field((0.30, 0.20), 0.50)   # v.t = 0.70 on the edge
NED_RIGHT = nedelec_field((0.90, 0.50), 0.20)  # v.t = 0.70 on the edge


def _vector_panel(ax, fL, fR, colour, title):
    triangle(ax, T_LEFT, edge=C_INK, face=C_FILL, lw=1.4)
    triangle(ax, T_RIGHT, edge=C_INK, face=C_FILL2, lw=1.4)
    ax.plot([V_P[0], V_Q[0]], [V_P[1], V_Q[1]], color=C_WARN, lw=2.4, zorder=4)

    scale = 0.30
    for verts, f in [(T_LEFT, fL), (T_RIGHT, fR)]:
        for p in interior_points(verts, 5):
            v = f(p)
            arrow(ax, p - 0.5 * scale * v, p + 0.5 * scale * v,
                  color=colour, lw=1.25, mut=8, z=7)
    ax.text(0.55, 1.10, "$v|_{K^-}$", color=colour, fontsize=9.5, ha="center")
    ax.text(1.45, 1.10, "$v|_{K^+}$", color=colour, fontsize=9.5, ha="center")
    clean_axes(ax, (-0.22, 2.22), (-0.18, 1.34))
    ax.set_title(title, fontsize=9.5, pad=6)


def _component_panel(ax, fL, fR, title):
    ys = np.linspace(0.0, 1.0, 80)
    nL = np.array([np.dot(fL([1.0, y]), NORMAL) for y in ys])
    nR = np.array([np.dot(fR([1.0, y]), NORMAL) for y in ys])
    tL = np.array([np.dot(fL([1.0, y]), TANGENT) for y in ys])
    tR = np.array([np.dot(fR([1.0, y]), TANGENT) for y in ys])

    ax.plot(nL, ys, color=C_HDIV, lw=2.0, label="$v\\cdot n$ from $K^-$")
    ax.plot(nR, ys, color=C_HDIV, lw=2.0, ls=(0, (4, 2.6)),
            label="$v\\cdot n$ from $K^+$")
    ax.plot(tL, ys, color=C_HCURL, lw=2.0, label="$v\\cdot t$ from $K^-$")
    ax.plot(tR, ys, color=C_HCURL, lw=2.0, ls=(0, (4, 2.6)),
            label="$v\\cdot t$ from $K^+$")

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.95, 1.55)
    ax.set_ylabel("position along the edge $F$", fontsize=9)
    ax.set_xlabel("component value", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.8, frameon=False, ncol=2, handlelength=2.0,
              columnspacing=1.0, loc="upper center", bbox_to_anchor=(0.46, -0.20))
    ax.set_title(title, fontsize=9.5, pad=6)


def fig_rt_normal():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.25),
                             gridspec_kw={"width_ratios": [1.3, 1.0]})
    _vector_panel(axes[0], RT_LEFT, RT_RIGHT, C_HDIV,
                  "(a)  a Raviart–Thomas field")
    _component_panel(axes[1], RT_LEFT, RT_RIGHT,
                     "(b)  components on the edge")
    fig.subplots_adjust(wspace=0.34)
    save(fig, "fig_rt_normal")


def fig_nedelec_tangential():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.25),
                             gridspec_kw={"width_ratios": [1.3, 1.0]})
    _vector_panel(axes[0], NED_LEFT, NED_RIGHT, C_HCURL,
                  "(a)  a Nédélec field")
    _component_panel(axes[1], NED_LEFT, NED_RIGHT,
                     "(b)  components on the edge")
    fig.subplots_adjust(wspace=0.34)
    save(fig, "fig_nedelec_tangential")


# ==========================================================================
# Figure 7 - normal versus tangential continuity, side by side
# ==========================================================================

def fig_normal_vs_tangential():
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.15))

    base = np.array([1.0, 0.85])
    cases = [
        dict(col=C_HDIV, kind="n",
             vm=(0.78, -0.42), vp=(0.78, 0.50),
             title="$H(\\mathrm{div})$: the normal part is glued",
             sub="$[\\,v\\cdot n\\,] = 0$,   $[\\,v\\cdot t\\,]$ may jump"),
        dict(col=C_HCURL, kind="t",
             vm=(0.34, 0.62), vp=(0.92, 0.62),
             title="$H(\\mathrm{curl})$: the tangential part is glued",
             sub="$[\\,v\\cdot t\\,] = 0$,   $[\\,v\\cdot n\\,]$ may jump"),
    ]

    for ax, c in zip(axes, cases):
        col = c["col"]
        ax.add_patch(Rectangle((-0.02, 0.0), 1.02, 1.95, facecolor=C_FILL,
                               edgecolor="none", zorder=0))
        ax.add_patch(Rectangle((1.0, 0.0), 1.42, 1.95, facecolor=C_FILL2,
                               edgecolor="none", zorder=0))
        ax.plot([1, 1], [0.0, 1.95], color=C_WARN, lw=2.4, zorder=3)
        ax.text(0.45, 1.80, "$K^-$", fontsize=10, ha="center", color=C_DIM)
        ax.text(1.70, 1.80, "$K^+$", fontsize=10, ha="center", color=C_DIM)

        vm, vp = np.array(c["vm"]), np.array(c["vp"])
        tip_m, tip_p = base + vm, base + vp
        arrow(ax, base, tip_m, color=C_INK, lw=2.0, mut=11, z=8)
        arrow(ax, base, tip_p, color=C_GREY, lw=2.0, mut=11, z=8)
        dof_point(ax, base, C_WARN, size=26, z=9)

        ax.text(tip_m[0] + 0.07, tip_m[1] - 0.04, "$v^-$", fontsize=10,
                color=C_INK, va="center")
        ax.text(tip_p[0] + 0.07, tip_p[1] + 0.03, "$v^+$", fontsize=10,
                color=C_GREY, va="center")

        if c["kind"] == "n":
            xg = tip_m[0]
            ax.plot([xg, xg], [tip_m[1] - 0.10, tip_p[1] + 0.10], color=col,
                    lw=1.3, ls=(0, (3.5, 2.5)), zorder=7)
            ax.annotate("", xy=(xg, 0.22), xytext=(1.0, 0.22),
                        arrowprops=dict(arrowstyle="<->", color=col, lw=1.5))
            ax.text(0.5 * (1.0 + xg), 0.06, "one $v\\cdot n$", fontsize=9,
                    color=col, ha="center")
            ax.text(xg + 0.06, 1.62, "both tips reach\nthe same depth",
                    fontsize=8.3, color=col, va="center", ha="left")
        else:
            yg = tip_m[1]
            ax.plot([tip_m[0] - 0.12, tip_p[0] + 0.12], [yg, yg], color=col,
                    lw=1.3, ls=(0, (3.5, 2.5)), zorder=7)
            ax.annotate("", xy=(0.62, yg), xytext=(0.62, base[1]),
                        arrowprops=dict(arrowstyle="<->", color=col, lw=1.5))
            ax.text(0.56, 0.5 * (yg + base[1]), "one $v\\cdot t$", fontsize=9,
                    color=col, ha="right", va="center")
            ax.text(1.20, yg + 0.13, "both tips reach the same height",
                    fontsize=8.3, color=col, ha="center")

        ax.text(1.20, -0.22, c["sub"], fontsize=9.2, color=col, ha="center")
        clean_axes(ax, (-0.05, 2.45), (-0.38, 1.98))
        ax.set_title(c["title"], fontsize=9.4, pad=6)

    fig.subplots_adjust(wspace=0.06)
    save(fig, "fig_normal_vs_tangential")


# ==========================================================================
# Figure 8 - why a jump leaves the Sobolev space
# ==========================================================================

def fig_jump_delta():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35))

    ax = axes[0]
    ax.plot([0, 1], [0.35, 0.35], color=C_L2, lw=2.4)
    ax.plot([1, 2], [0.85, 0.85], color=C_L2, lw=2.4)
    ax.scatter([1, 1], [0.35, 0.85], color=C_L2, s=26, zorder=5)
    ax.axvline(1, color=C_WARN, lw=1.1, ls=(0, (4, 2.5)))
    ax.annotate("", xy=(1.0, 0.85), xytext=(1.0, 0.35),
                arrowprops=dict(arrowstyle="<->", color=C_WARN, lw=1.3))
    ax.text(1.07, 0.60, "jump $= c$", color=C_WARN, fontsize=9.2, va="center")
    ax.set_ylim(0, 1.25)
    ax.set_xlim(0, 2)
    ax.set_xticks([1])
    ax.set_xticklabels(["$F$"])
    ax.set_yticks([])
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("(a)  $u_h$ jumps across the interface", fontsize=9.5, pad=6)

    ax = axes[1]
    ax.plot([0, 1], [0, 0], color=C_INK, lw=2.0)
    ax.plot([1, 2], [0, 0], color=C_INK, lw=2.0)
    arrow(ax, (1.0, 0.0), (1.0, 0.92), color=C_WARN, lw=2.4, mut=13)
    ax.text(1.09, 0.84, "$c\\,\\delta_F$", color=C_WARN, fontsize=10, va="center")
    ax.text(1.09, 0.48, "an infinite spike:\nnot a function in $L^2$",
            color=C_DIM, fontsize=8.4, va="center")
    ax.set_ylim(-0.12, 1.25)
    ax.set_xlim(0, 2)
    ax.set_xticks([1])
    ax.set_xticklabels(["$F$"])
    ax.set_yticks([])
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("(b)  its weak derivative $\\partial_x u_h$", fontsize=9.5, pad=6)

    fig.subplots_adjust(wspace=0.16)
    save(fig, "fig_jump_delta")


# ==========================================================================
# Figure 9 - degree-of-freedom zoo
# ==========================================================================

def fig_dof_zoo():
    fig, axes = plt.subplots(2, 4, figsize=(6.7, 3.75))
    axes = axes.ravel()

    def setup(ax, title, sub, colour):
        clean_axes(ax, (-0.30, 1.30), (-0.34, 1.10))
        ax.set_title(title, fontsize=9.3, pad=3, color=colour)
        ax.text(0.5, -0.28, sub, fontsize=8.0, color=C_DIM, ha="center")

    V = GLYPH
    mids = [0.5 * (V[i] + V[(i + 1) % 3]) for i in range(3)]
    cen = V.mean(axis=0)

    ax = axes[0]
    glyph_triangle(ax, color=C_H1)
    for p in V:
        dof_point(ax, p, C_H1, size=34)
    setup(ax, "$P_1$ Lagrange", "$H^1$  ·  3 dofs", C_H1)

    ax = axes[1]
    glyph_triangle(ax, color=C_H1)
    for p in list(V) + mids:
        dof_point(ax, p, C_H1, size=34)
    setup(ax, "$P_2$ Lagrange", "$H^1$  ·  6 dofs", C_H1)

    ax = axes[2]
    glyph_triangle(ax, color=C_H1)
    for p in V:
        dof_point(ax, p, C_H1, size=30)
    for i in range(3):
        a, b = V[i], V[(i + 1) % 3]
        for s in (1 / 3, 2 / 3):
            dof_point(ax, a + s * (b - a), C_H1, size=30)
    dof_point(ax, cen, C_H1, size=30)
    setup(ax, "$P_3$ Lagrange", "$H^1$  ·  10 dofs", C_H1)

    ax = axes[3]
    glyph_triangle(ax, color=C_CR, dashed=True)
    for p in mids:
        dof_point(ax, p, C_CR, size=34)
    setup(ax, "Crouzeix–Raviart", "nonconforming  ·  3 dofs", C_CR)

    ax = axes[4]
    glyph_triangle(ax, color=C_L2, dashed=True)
    for p in V:
        dof_point(ax, p + 0.15 * (cen - p) / np.linalg.norm(cen - p), C_L2, size=34)
    setup(ax, "$P_1$ discontinuous", "$L^2$  ·  3 dofs, cell-local", C_L2)

    ax = axes[5]
    v = glyph_triangle(ax, color=C_HDIV)
    for i in range(3):
        dof_normal(ax, v, i, C_HDIV)
    setup(ax, "$RT_0$", "$H(\\mathrm{div})$  ·  3 dofs", C_HDIV)

    ax = axes[6]
    v = glyph_triangle(ax, color=C_HDIV)
    for i in range(3):
        a, b = v[i], v[(i + 1) % 3]
        _, nrm = outward_normal(v, i)
        for s in (0.28, 0.72):
            p = a + s * (b - a)
            arrow(ax, p - 0.08 * nrm, p + 0.16 * nrm, color=C_HDIV, lw=1.2)
    setup(ax, "$BDM_1$", "$H(\\mathrm{div})$  ·  6 dofs", C_HDIV)

    ax = axes[7]
    v = glyph_triangle(ax, color=C_HCURL)
    for i in range(3):
        dof_tangent(ax, v, i, C_HCURL)
    setup(ax, "$N_0$ Nédélec", "$H(\\mathrm{curl})$  ·  3 dofs", C_HCURL)

    fig.subplots_adjust(wspace=0.06, hspace=0.46)
    save(fig, "fig_dof_zoo")


# ==========================================================================
# Figure 10 - the de Rham complex
# ==========================================================================

def fig_derham():
    fig, ax = plt.subplots(figsize=(6.7, 3.5))

    xs = [0.0, 1.30, 2.60, 3.90]
    colours = [C_H1, C_HCURL, C_HDIV, C_L2]
    spaces = ["$H^1$", "$H(\\mathrm{curl})$", "$H(\\mathrm{div})$", "$L^2$"]
    families = ["Lagrange\n$P_k$", "Nédélec\n$N_{k-1}$",
                "Raviart–Thomas\n$RT_{k-1}$", "discontinuous\n$P_{k-1}$"]
    quantity = ["potential", "field", "flux", "density"]
    kinds = ["lagrange", "curl", "div", "dg"]
    ops = ["$\\mathrm{grad}$", "$\\mathrm{curl}$", "$\\mathrm{div}$"]

    box_w, box_h = 0.80, 0.44
    y_top = 2.34

    for x, col, sp, fam, q in zip(xs, colours, spaces, families, quantity):
        ax.add_patch(Rectangle((x - box_w / 2, y_top), box_w, box_h,
                               facecolor="white", edgecolor=col, lw=1.6,
                               zorder=3, joinstyle="round"))
        ax.text(x, y_top + box_h / 2, sp, ha="center", va="center",
                fontsize=12, color=col, zorder=4)
        ax.text(x, y_top + box_h + 0.14, q, ha="center", va="center",
                fontsize=8.3, color=C_DIM)
        ax.text(x, 1.66, fam, ha="center", va="center", fontsize=8.8, color=col)

    for i, op in enumerate(ops):
        x0 = xs[i] + box_w / 2 + 0.05
        x1 = xs[i + 1] - box_w / 2 - 0.05
        arrow(ax, (x0, y_top + box_h / 2), (x1, y_top + box_h / 2),
              color=C_INK, lw=1.3, mut=10)
        ax.text(0.5 * (x0 + x1), y_top + box_h / 2 + 0.12, op,
                ha="center", fontsize=9.5)

    gs = 0.44
    for x, col, kind in zip(xs, colours, kinds):
        draw_glyph(ax, (x - gs / 2, 0.70), gs, col, kind, dot=20, nlen=0.11)

    for y, lab in [(y_top + box_h / 2, "continuous\nspaces"),
                   (1.66, "finite element\nfamily"),
                   (0.92, "degrees of\nfreedom")]:
        ax.text(-1.15, y, lab, fontsize=8.6, color=C_DIM, ha="center", va="center")

    ax.text(1.95, 0.20,
            "each map sends one space exactly onto the kernel of the next:   "
            "$\\mathrm{curl}\\,\\mathrm{grad} = 0$,    $\\mathrm{div}\\,\\mathrm{curl} = 0$",
            fontsize=8.8, color=C_INK, ha="center")
    ax.text(1.95, -0.06,
            "degrees of freedom sit on vertices $\\to$ edges $\\to$ faces $\\to$ cells",
            fontsize=8.8, color=C_DIM, ha="center")

    clean_axes(ax, (-1.75, 4.55), (-0.24, 3.10))
    save(fig, "fig_derham")


# ==========================================================================
# Figure 11 - local conservation in a mixed method
# ==========================================================================

def fig_conservation():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.85))

    panels = [
        dict(col=C_HDIV, arrows=[(0.50, 2.4, "$2.4$")],
             title="(a)  an $H(\\mathrm{div})$ flux",
             note="one number lives on the edge:\n"
                  "what leaves $K^-$ is what enters $K^+$"),
        dict(col=C_WARN,
             arrows=[(0.66, 2.4, "$2.4$ in $K^-$"),
                     (0.34, 1.7, "$1.7$ in $K^+$")],
             title="(b)  a flux built from $-K\\nabla p_h$",
             note="the two cells disagree:\n"
                  "$0.7$ units of mass vanish at the interface"),
    ]

    for ax, c in zip(axes, panels):
        triangle(ax, T_LEFT, edge=C_INK, face=C_FILL, lw=1.4)
        triangle(ax, T_RIGHT, edge=C_INK, face=C_FILL2, lw=1.4)
        ax.plot([V_P[0], V_Q[0]], [V_P[1], V_Q[1]], color=C_WARN, lw=2.2, zorder=4)

        col = c["col"]
        for y, val, lab in c["arrows"]:
            length = 0.16 * val
            arrow(ax, (1.0 - 0.5 * length, y), (1.0 + 0.5 * length, y),
                  color=col, lw=2.0, mut=11, z=8)
            ax.text(1.0 + 0.5 * length + 0.07, y, lab, fontsize=8.6,
                    color=col, va="center", ha="left", zorder=9,
                    bbox=dict(facecolor=C_FILL2, edgecolor="none", pad=1.4))

        ax.text(0.45, -0.13, "$K^-$", fontsize=9.6, ha="center", va="center",
                color=C_DIM)
        ax.text(1.55, -0.13, "$K^+$", fontsize=9.6, ha="center", va="center",
                color=C_DIM)
        ax.text(1.0, -0.42, c["note"], fontsize=8.5, color=col, ha="center",
                va="top")

        clean_axes(ax, (-0.32, 2.55), (-0.85, 1.16))
        ax.set_title(c["title"], fontsize=9.3, pad=6, color=col)

    fig.subplots_adjust(wspace=0.06)
    save(fig, "fig_conservation")


# ==========================================================================
# Figure 12 - Taylor-Hood and the checkerboard mode
# ==========================================================================

def fig_taylor_hood():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75))

    ax = axes[0]
    v = glyph_triangle(ax, origin=(0.0, 0.0), scale=1.0, color=C_H1, lw=1.6)
    mids = [0.5 * (v[i] + v[(i + 1) % 3]) for i in range(3)]
    for p in v:                       # pressure ring first, velocity dot on top
        ax.scatter([p[0]], [p[1]], s=150, marker="o", facecolor="none",
                   edgecolor=C_L2, linewidths=1.7, zorder=7)
    for p in list(v) + mids:
        dof_point(ax, p, C_H1, size=46, z=8)

    handles = [
        Line2D([0], [0], linestyle="none", marker="o", markersize=6.5,
               markerfacecolor=C_H1, markeredgecolor=C_H1,
               label="velocity: $P_2$, all 6 nodes"),
        Line2D([0], [0], linestyle="none", marker="o", markersize=10,
               markerfacecolor="none", markeredgecolor=C_L2, markeredgewidth=1.7,
               label="pressure: $P_1$, vertices only"),
    ]
    ax.legend(handles=handles, fontsize=8.4, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.02),
              handletextpad=0.6, labelspacing=0.5)
    clean_axes(ax, (-0.32, 1.32), (-0.72, 1.12))
    ax.set_title("(a)  the Taylor–Hood pair $P_2$/$P_1$", fontsize=9.4, pad=6)

    ax = axes[1]
    n = 4
    for i in range(n):
        for j in range(n):
            plus = (i + j) % 2 == 0
            ax.add_patch(Rectangle((i, j), 1, 1,
                                   facecolor="#d8e2f3" if plus else "#f6ddd2",
                                   edgecolor=C_LINE, lw=0.8))
            ax.text(i + 0.5, j + 0.5, "$+$" if plus else "$-$",
                    ha="center", va="center", fontsize=11,
                    color=C_H1 if plus else C_HDIV)
    clean_axes(ax, (-0.35, 4.35), (-1.55, 4.35))
    ax.set_title("(b)  a checkerboard pressure mode", fontsize=9.4, pad=6)
    ax.text(2.0, -0.55, "an unstable pair such as $P_1$/$P_1$\n"
                        "cannot tell this apart from $p = 0$",
            fontsize=8.4, color=C_DIM, ha="center", va="top")

    fig.subplots_adjust(wspace=0.10)
    save(fig, "fig_taylor_hood")


# ==========================================================================
# Figure 13 - the Falk element of the companion code
# ==========================================================================

def fig_falk():
    fig, ax = plt.subplots(figsize=(6.6, 2.6))

    entries = [
        (0.00, C_HDIV, "$\\sigma$  stress", "$BDM_k$ rows",
         "$H(\\mathrm{div})$", "traction $\\sigma n$ continuous", "div"),
        (1.15, C_L2, "$u$  displacement", "broken $P_{k-1}$",
         "$L^2$", "never differentiated", "dg"),
        (2.30, C_H1, "$\\gamma$  rotation", "$P_k$ Lagrange",
         "$H^1$", "needed for stability", "lagrange"),
    ]

    gs = 0.38
    for x, col, field, space, sob, why, kind in entries:
        ax.add_patch(Rectangle((x - 0.52, 0.04), 1.04, 1.46, facecolor="white",
                               edgecolor=col, lw=1.5, zorder=2))
        ax.text(x, 1.34, field, ha="center", fontsize=10.2, color=col, zorder=4)
        draw_glyph(ax, (x - gs / 2, 0.68), gs, col, kind, dot=18, nlen=0.10)
        ax.text(x, 0.50, space, ha="center", fontsize=8.8, color=C_INK)
        ax.text(x, 0.32, sob, ha="center", fontsize=9.4, color=col)
        ax.text(x, 0.14, why, ha="center", fontsize=7.6, color=C_DIM)

    ax.text(1.15, 1.70, "one mixed element, three different spaces",
            ha="center", fontsize=9.8, color=C_INK)
    clean_axes(ax, (-0.72, 3.02), (0.0, 1.90))
    save(fig, "fig_falk")


# ==========================================================================
# Figure 14 - the cheat sheet
# ==========================================================================

def fig_cheatsheet():
    fig = plt.figure(figsize=(6.7, 3.15))

    rows = [
        (C_H1, "lagrange", "$H^1$", "Lagrange $P_k$", "the whole value",
         "Poisson, elasticity, heat"),
        (C_HCURL, "curl", "$H(\\mathrm{curl})$", "Nédélec $N_k$",
         "tangential part $v\\cdot t$", "Maxwell, eddy currents"),
        (C_HDIV, "div", "$H(\\mathrm{div})$", "$RT_k$,  $BDM_k$",
         "normal part $v\\cdot n$", "Darcy, mixed Poisson"),
        (C_L2, "dg", "$L^2$", "discontinuous $P_k$", "nothing",
         "advection, shocks"),
    ]

    x_space, x_fam, x_glue, x_pde = 0.165, 0.345, 0.575, 0.830
    for x, head in [(x_space, "space"), (x_fam, "element"),
                    (x_glue, "what is glued"), (x_pde, "typical PDE")]:
        fig.text(x, 0.935, head, fontsize=8.6, color=C_DIM, ha="center")
    fig.add_artist(Line2D([0.02, 0.98], [0.905, 0.905], color=C_LINE, lw=0.9))

    row_h = 0.195
    for i, (col, kind, sp, fam, glue, pde) in enumerate(rows):
        y0 = 0.695 - i * row_h
        yc = y0 + 0.075

        axg = fig.add_axes([0.025, y0, 0.085, 0.15])
        axg.set_xlim(-0.12, 1.12)
        axg.set_ylim(-0.10, 1.00)
        axg.set_aspect("equal")
        axg.axis("off")
        draw_glyph(axg, (0.0, 0.0), 1.0, col, kind, dot=16, nlen=0.19, tfrac=0.42)

        fig.text(x_space, yc, sp, fontsize=10, color=col, ha="center", va="center")
        fig.text(x_fam, yc, fam, fontsize=9, color=C_INK, ha="center", va="center")
        fig.text(x_glue, yc, glue, fontsize=9, color=col, ha="center", va="center")
        fig.text(x_pde, yc, pde, fontsize=8.6, color=C_DIM, ha="center", va="center")

        if i < len(rows) - 1:
            fig.add_artist(Line2D([0.02, 0.98], [y0 - 0.022, y0 - 0.022],
                                  color="#e6e8ee", lw=0.8))

    fig.text(0.5, 0.035,
             "reading down the table: less and less is glued together across "
             "element boundaries",
             fontsize=8.6, color=C_DIM, ha="center")
    save(fig, "fig_cheatsheet")


# ==========================================================================

FIGURES = [
    fig_mesh,
    fig_p1_dofs,
    fig_lagrange_continuity,
    fig_dg_jump,
    fig_rt_normal,
    fig_nedelec_tangential,
    fig_normal_vs_tangential,
    fig_jump_delta,
    fig_dof_zoo,
    fig_derham,
    fig_conservation,
    fig_taylor_hood,
    fig_falk,
    fig_cheatsheet,
]


def main():
    for f in FIGURES:
        f()
    print("\n%d figures written to %s" % (len(FIGURES), HERE))


if __name__ == "__main__":
    main()
