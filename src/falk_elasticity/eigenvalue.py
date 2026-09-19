"""Assembly and SLEPc solve of the Falk mixed eigenvalue problem, Petersen
(2022) eq. (36)/(37): find kappa in R and (sigma, u, gamma) in
Sigma_h x U_h x X_h, not identically zero, such that

    a(sigma, tau) + b(tau, u) + c(gamma, tau) = 0            for all tau
    b(sigma, v)                               = -kappa (u,v) for all v
    c(sigma, eta)                             = 0            for all eta

This reuses the exact same combined, symmetric bilinear form A(.,.) from the
source problem (problem.py), now homogeneous, paired against a mass form
B(.,.) supported only on the displacement block:

    A(x, y) = -kappa * B(x, y),   B((sigma,u,gamma),(tau,v,eta)) := (u, v)

so kappa = -mu, where mu ranges over the eigenvalues of the pencil (A, B).
B is positive semi-definite but singular (identically zero on the stress
and multiplier blocks), which formally puts some pencil eigenvalues at
infinity; shift-and-invert handles that robustly since those map far from
any finite target and simply never show up in the converged set.
"""
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix
from petsc4py import PETSc
from slepc4py import SLEPc

from .elements import as_skew, as_stress, falk_function_space
from .materials import C_inv


def assemble_eigenproblem(
    domain: mesh.Mesh,
    k: int,
    lmbda: float,
    mu: float,
    quadrature_degree: int = 16,
):
    """Assemble the pencil (A, B) for the order-k Falk eigenvalue problem."""
    W = falk_function_space(domain, k)
    sigma0, sigma1, u, q = ufl.TrialFunctions(W)
    tau0, tau1, v, p = ufl.TestFunctions(W)

    sigma = as_stress(sigma0, sigma1)
    tau = as_stress(tau0, tau1)
    gamma = as_skew(q)
    eta = as_skew(p)

    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quadrature_degree})

    a_form = (
        ufl.inner(C_inv(sigma, lmbda, mu), tau) * dx
        + ufl.inner(ufl.div(tau), u) * dx
        + ufl.inner(gamma, tau) * dx
        + ufl.inner(ufl.div(sigma), v) * dx
        + ufl.inner(sigma, eta) * dx
    )
    b_form = ufl.inner(u, v) * dx

    A = assemble_matrix(fem.form(a_form))
    A.assemble()
    B = assemble_matrix(fem.form(b_form))
    B.assemble()
    return W, A, B


def solve_eigenproblem(A: "PETSc.Mat", B: "PETSc.Mat", target_kappa: float, nev: int = 8) -> list[float]:
    """Shift-and-invert around target_kappa; return the converged kappas.

    A and B are both symmetric, but B is singular, so this deliberately uses
    the general (non-Hermitian-exploiting) SLEPc pathway rather than GHEP,
    which assumes a positive-definite B.
    """
    E = SLEPc.EPS().create(A.getComm())
    E.setOperators(A, B)
    E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    E.setDimensions(nev)
    E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)

    target_mu = -target_kappa
    E.setTarget(target_mu)

    st = E.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(target_mu)
    ksp = st.getKSP()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")

    E.solve()

    nconv = E.getConverged()
    kappas = [-E.getEigenvalue(i).real for i in range(nconv)]
    return sorted(kappas)
