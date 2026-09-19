# Falk mixed elasticity in FEniCSx

A [DOLFINx](https://github.com/FEniCS/dolfinx) implementation of the **Falk
mixed finite element** for linear elasticity with *weakly imposed* stress
symmetry — the method studied in my master's thesis, *Postprocessing of
Mixed Elastic Eigenvalues* (Petersen, 2022, University of Bonn, advised by
Prof. Dr. Joscha Gedicke).

## The method, briefly

Instead of solving for the displacement `u` alone, the Hellinger–Reissner
mixed formulation solves for the stress `σ` and displacement `u`
simultaneously — stresses come out directly, and the method avoids the
locking that plagues displacement-only elements for (nearly) incompressible
materials. Conservation of angular momentum requires `σ` to be symmetric,
but finite elements that impose this exactly are complex and expensive
(Arnold–Winther). The Falk element instead enforces symmetry only *weakly*,
via a Lagrange multiplier `γ` (an idea going back to Fraijs de Veubeke), and
solves the saddle point problem for `(σ, u, γ)` at once:

```
a(σ, τ) + b(τ, u) + c(γ, τ) = 0        for all τ
b(σ, v)                     = -(f, v)  for all v
c(σ, η)                     = 0        for all η
```

with `a(σ,τ) = ∫ C⁻¹σ:τ dx`, `b(τ,v) = ∫ div(τ)·v dx`, `c(η,τ) = ∫ η:τ dx`,
and `C` the isotropic elasticity tensor built from the Lamé parameters. On
each triangle, the order-`k` Falk element pairs:

| field                        | space                          |
|-------------------------------|----------------------------------|
| stress `σ` (as two rows)       | `BDM_k`, H(div)-conforming       |
| displacement `u`               | discontinuous `P_{k-1}`          |
| skew multiplier `γ = S₂(q)`     | `q` in continuous `P_k`          |

`S₂(q) := [[0, q], [-q, 0]]`. This is exactly what
[`falk_elasticity.elements.falk_function_space`](src/falk_elasticity/elements.py)
builds out of standard [Basix](https://github.com/FEniCS/basix) elements —
no custom finite element needed, just the right combination of existing
ones.

## What's implemented

- [x] **Phase 1 — the source problem.** The Falk element for `k = 1, 2, 3`,
  assembled and solved as a single monolithic saddle-point system on the
  unit square. A manufactured trigonometric solution verifies the a priori
  rates from the thesis (eq. 1):
  `‖σ-σ_h‖₀ + ‖γ-γ_h‖₀ = O(h^{k+1})`, `‖u-u_h‖₀ = O(h^k)`.
  See [`tests/test_convergence.py`](tests/test_convergence.py) and
  [`scripts/convergence_study.py`](scripts/convergence_study.py).
- [x] **Phase 2 — the eigenvalue problem.** The actual subject of the
  thesis: elastic eigenfrequencies via SLEPc shift-and-invert, reproducing
  the reference eigenvalue `κ = 51.294997977322` (third eigenvalue, unit
  square, Dirichlet data) and its `O(h^{2k})` convergence rate for `k=1`.
  The eigenvalue problem reuses the exact same bilinear form as the source
  problem, now paired against a mass form supported only on the
  displacement block — see
  [`src/falk_elasticity/eigenvalue.py`](src/falk_elasticity/eigenvalue.py),
  [`tests/test_eigenvalue.py`](tests/test_eigenvalue.py), and
  [`scripts/eigenvalue_study.py`](scripts/eigenvalue_study.py).
- [x] **Phase 3 — the postprocessing.** The thesis's actual contribution:
  a cheap, element-local solve (eq. 52/53) that turns `(σ_h, u_h, γ_h)` into
  a postprocessed eigenpair `(κ*_h, u*_h)` via the Rayleigh quotient of
  Definition 5.5, improving the eigenvalue rate from `O(h^{2k})` to
  `O(h^{2k+2})` (Theorem 5.7) — with observed *super*convergence beyond that
  for the lowest-order element, `k=1`, reproduced numerically here too. The
  postprocessing is itself block-diagonal (broken trial/test spaces, no
  facet terms), so it's implemented as a single assembly rather than a
  per-triangle loop — see
  [`src/falk_elasticity/postprocessing.py`](src/falk_elasticity/postprocessing.py),
  [`tests/test_postprocessing.py`](tests/test_postprocessing.py), and
  [`scripts/postprocessing_study.py`](scripts/postprocessing_study.py).
- [x] **Phase 4 — a posteriori error estimation & adaptivity, on the
  L-shaped domain.** The reliable/efficient a posteriori estimator `η`
  (Section 6) as a DG0 cellwise indicator, driving Doerfler (bulk-chasing)
  marking and adaptive refinement via `dolfinx.mesh.refine`. On
  `Ω = (-1,1)×(0,1) ∪ (-1,0)×(-1,0)` (Section 7.2), the eigenfunction is
  singular at the re-entrant corner, so uniform refinement alone can't
  recover the full rate — adaptivity is the point. See
  [`src/falk_elasticity/domains.py`](src/falk_elasticity/domains.py),
  [`src/falk_elasticity/estimator.py`](src/falk_elasticity/estimator.py),
  [`src/falk_elasticity/adaptive.py`](src/falk_elasticity/adaptive.py),
  [`tests/test_l_shaped.py`](tests/test_l_shaped.py),
  [`tests/test_adaptive.py`](tests/test_adaptive.py), and
  [`scripts/adaptive_study.py`](scripts/adaptive_study.py) (uniform vs.
  adaptive convergence, reproducing the comparison in Figs. 16–22).
- [ ] **Phase 5 — Cook's membrane.** Deferred: unlike the square and
  L-shaped domains, it's clamped on only one edge and free elsewhere, which
  needs genuine essential boundary conditions on the stress space
  (`Σ_g` with `g=0` strongly enforced on the free edges) — a boundary
  condition path Phases 1–4 don't exercise, since both existing domains use
  pure Dirichlet data everywhere.

## Running it

DOLFINx isn't pip-installable in the usual sense (PETSc/SLEPc/MPI and Basix
need to be compiled together) — use the official Docker image or
conda-forge.

**Docker**
```bash
docker run -it --rm -v $(pwd):/workspace dolfinx/dolfinx:stable
cd /workspace && pip install -e . && pytest -v -s tests/
```

**Conda**
```bash
conda env create -f environment.yml
conda activate falk-elasticity
pip install -e .
pytest -v -s tests/
python scripts/convergence_study.py   # writes results/convergence.{png,csv}
```

CI runs the same test suite inside `dolfinx/dolfinx:stable` on every push
(see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)) and uploads the
convergence plot/data as a build artifact.

## Reference

Malte Petersen, *Postprocessing of Mixed Elastic Eigenvalues*, Master's
thesis, Rheinische Friedrich-Wilhelms-Universität Bonn, 2022. Advisor:
Prof. Dr. Joscha Gedicke, second advisor Prof. Dr. Ira Neitzel.

R. S. Falk, *Finite elements for linear elasticity*, in *Mixed Finite
Elements, Compatibility Conditions, and Applications*, Springer Lecture
Notes in Mathematics, 2008.

D. Boffi, F. Brezzi, M. Fortin, *Mixed Finite Element Methods and
Applications*, Springer, 2013.

---
[petersenmalte.de](https://petersenmalte.de) · [@petersenmalte](https://github.com/petersenmalte)
