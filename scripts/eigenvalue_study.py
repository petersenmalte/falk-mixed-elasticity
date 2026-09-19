"""Reproduce Fig. 8 of Petersen (2022): convergence of the third eigenvalue
kappa_h toward the reference value 51.294997977322 on the unit square with
Dirichlet data, k = 1, 2, 3.
"""
import csv
import pathlib

import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import mesh

from falk_elasticity.eigenvalue import assemble_eigenproblem, solve_eigenproblem

REFERENCE_KAPPA = 51.294997977322
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
NS_BY_K = {1: [8, 16, 24, 32], 2: [8, 16, 24], 3: [8, 12, 16]}


def closest_eigenvalue(n: int, k: int, nev: int = 8) -> float:
    domain = mesh.create_unit_square(MPI.COMM_WORLD, n, n, mesh.CellType.triangle)
    _, A, B = assemble_eigenproblem(domain, k, 1.0, 1.0)
    pairs = solve_eigenproblem(A, B, REFERENCE_KAPPA, nev=nev)
    kappa, _ = min(pairs, key=lambda pair: abs(pair[0] - REFERENCE_KAPPA))
    return kappa


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    fig, ax = plt.subplots(figsize=(6, 5))

    for k, ns in NS_BY_K.items():
        h, err = [], []
        for n in ns:
            kappa_h = closest_eigenvalue(n, k)
            e = abs(kappa_h - REFERENCE_KAPPA)
            h.append(1.0 / n)
            err.append(e)
            rows.append({"k": k, "n": n, "h": 1.0 / n, "kappa_h": kappa_h, "error": e})
        ax.loglog(h, err, "o-", label=f"k={k}")

    ax.set_xlabel("mesh size h")
    ax.set_ylabel("|kappa - kappa_h|")
    ax.set_title("Falk element eigenvalue convergence, unit square, Dirichlet")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eigenvalue_convergence.png", dpi=150)

    with open(OUT_DIR / "eigenvalue_convergence.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["k", "n", "h", "kappa_h", "error"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT_DIR / 'eigenvalue_convergence.png'} and {OUT_DIR / 'eigenvalue_convergence.csv'}")


if __name__ == "__main__":
    main()
