# A Field Guide to Finite Element Families

A standalone, visual introduction to the main families of finite elements —
Lagrange, discontinuous Galerkin, Raviart–Thomas, BDM and Nédélec, plus
Crouzeix–Raviart, Argyris and Taylor–Hood — written for someone who knows
calculus and linear algebra but has not studied finite element spaces.

The guide is built around one question: **how much of a function is forced to
match across the boundary between two neighbouring cells?** Every family is a
different answer, and the right answer is dictated by the PDE.

**Read it:** [`finite_elements_guide.pdf`](finite_elements_guide.pdf) (24 pages).

This directory is self-contained and independent of the rest of the
repository. It is a document and a demo, not part of the `falk_elasticity`
package, and nothing here is published to a website.

## Contents

| Path | What it is |
| --- | --- |
| `finite_elements_guide.tex` | LaTeX source of the guide |
| `finite_elements_guide.pdf` | the built document |
| `figures/make_figures.py` | generates every figure the document uses |
| `figures/*.pdf` | the 14 generated figures |
| `examples/visualize_elements.py` | the runnable demonstration (numpy + matplotlib) |
| `examples/basix_elements.py` | optional cross-check against the FEniCSx element library |
| `examples/element_comparison.png` | what the demonstration produces |

## Rebuilding the PDF

You need a LaTeX distribution with `latexmk`, and Python with numpy and
matplotlib for the figures.

```bash
cd figures && python3 make_figures.py && cd ..   # regenerate the 14 figures
latexmk -pdf finite_elements_guide.tex           # build the document
latexmk -c                                       # optional: clean aux files
```

The figure step is only needed if you change `make_figures.py`; the generated
PDFs are committed, so `latexmk -pdf` alone is enough to rebuild the document.

On a Debian or Ubuntu machine the LaTeX packages are:

```bash
sudo apt-get install -y --no-install-recommends \
    texlive-latex-base texlive-latex-recommended texlive-latex-extra \
    texlive-fonts-recommended texlive-pictures latexmk
```

## Running the demonstration

```bash
cd examples
python3 visualize_elements.py          # writes element_comparison.png
python3 visualize_elements.py --show   # opens an interactive window instead
```

`visualize_elements.py` builds four finite element spaces **by hand** on a
two-triangle mesh — continuous Lagrange, discontinuous Galerkin,
Raviart–Thomas and Nédélec — and draws what each one does at the shared edge.
Every basis function is written out in a few lines of numpy straight from its
definition; nothing is hidden inside a framework.

Before drawing, the script verifies numerically that the degrees of freedom
really are dual to the basis functions, and then reports the jumps it measures
across the shared edge:

```
  RT_0      cell 0:  dof_j(phi_i) = identity?  yes   (max error 2.22e-16)
  ...
What happens on the shared edge:
  continuous P1  : largest jump in the value      = 2.22e-16
  discontinuous  : largest jump in the value      = 0.480
  RT_0           : largest jump in v.n            = 0.00e+00
                   largest jump in v.t            = 0.850
  Nedelec N_0    : largest jump in v.n            = 1.100
                   largest jump in v.t            = 1.11e-16
```

That output is the whole guide in six numbers: Lagrange glues the value,
discontinuous Galerkin glues nothing, Raviart–Thomas glues the normal
component only, and Nédélec glues the tangential component only.

## Required Python packages

| Package | Needed for | Required? |
| --- | --- | --- |
| `numpy` | figures and demonstration | yes |
| `matplotlib` | figures and demonstration | yes |
| `fenics-basix` | `examples/basix_elements.py` only | optional |

```bash
pip install numpy matplotlib          # everything except the cross-check
pip install fenics-basix              # to also run basix_elements.py
```

## Is FEniCSx required?

**No.** The guide and the demonstration need only numpy and matplotlib, and
DOLFINx is not used anywhere in this directory.

There is one optional extra. `examples/basix_elements.py` talks to
[Basix](https://github.com/FEniCS/basix), the element library that FEniCSx and
DOLFINx actually use, and does two things with it:

1. prints, for each family, how many degrees of freedom sit on vertices, on
   edges and strictly inside the cell, next to the Sobolev space Basix reports
   — the guide's central claim in machine-readable form;
2. verifies that the Raviart–Thomas and Nédélec spaces built by hand in
   `visualize_elements.py` span exactly the polynomials Basix calls `RT`
   degree 1 and `N1E` degree 1.

Basix installs from PyPI as a self-contained wheel — the full DOLFINx stack
(PETSc, MPI) is **not** needed:

```bash
pip install fenics-basix
cd examples && python3 basix_elements.py
```

```
element             dim  vert  edge  cell   space  continuous across an edge
----------------------------------------------------------------------------
Lagrange P1           3     1     0     0   H1     the whole value
discontinuous P1      3     0     0     3   L2     nothing
RT (lowest)           3     0     1     0   HDiv   the normal component
BDM (lowest)          6     0     2     0   HDiv   the normal component
Nedelec 1st kind      3     0     1     0   HCurl  the tangential component
...
```

## A note on degree conventions

The guide uses the classical mathematical convention in which `RT_0` and `N_0`
are the *lowest-order* spaces. Basix and FEniCSx count from one instead, so
the same elements are called degree 1 there. The elements are identical; only
the label shifts by one. `basix_elements.py` says so in its output too.

## Relation to the rest of this repository

The repository implements the Falk mixed finite element for linear elasticity
with weakly imposed symmetry. That element uses three different spaces at
once — `BDM` rows for the stress, a broken space for the displacement, and
Lagrange for the rotation multiplier — which is exactly the situation this
guide exists to explain. Section 11.1 of the PDF walks through it.
