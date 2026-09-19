"""Reproduce Petersen (2022) Section 7.2's key point: on the non-convex
L-shaped domain, uniform refinement cannot recover the optimal
postprocessed-eigenvalue rate (the eigenfunction is singular at the
re-entrant corner), but adaptive refinement driven by the estimator eta
does. Lowest-order Falk element, k=1.
"""
import csv
import pathlib

import matplotlib.pyplot as plt
from mpi4py import MPI

from falk_elasticity.adaptive import doerfler_mark, refine_marked
from falk_elasticity.domains import create_l_shaped_mesh
from falk_elasticity.estimator import solve_estimate

LMBDA, MU = 1.0, 1.0
K = 1
REFERENCE_KAPPA = 13.591920213185
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"


def num_cells(domain) -> int:
    tdim = domain.topology.dim
    return domain.topology.index_map(tdim).size_local


def uniform_study(ns):
    rows = []
    for n in ns:
        domain = create_l_shaped_mesh(MPI.COMM_WORLD, n)
        kappa_h, kappa_star_h, eta, _ = solve_estimate(domain, K, LMBDA, MU, REFERENCE_KAPPA)
        rows.append(
            {
                "strategy": "uniform", "step": n, "ncells": num_cells(domain),
                "kappa_h": kappa_h, "kappa_star_h": kappa_star_h, "eta": eta,
                "error": abs(kappa_star_h - REFERENCE_KAPPA),
            }
        )
    return rows


def adaptive_study(n0, nsteps, theta=0.5):
    domain = create_l_shaped_mesh(MPI.COMM_WORLD, n0)
    rows = []
    for step in range(nsteps):
        kappa_h, kappa_star_h, eta, eta2 = solve_estimate(domain, K, LMBDA, MU, REFERENCE_KAPPA)
        rows.append(
            {
                "strategy": "adaptive", "step": step, "ncells": num_cells(domain),
                "kappa_h": kappa_h, "kappa_star_h": kappa_star_h, "eta": eta,
                "error": abs(kappa_star_h - REFERENCE_KAPPA),
            }
        )
        marked = doerfler_mark(eta2.x.array, theta=theta)
        domain = refine_marked(domain, marked)
    return rows


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = uniform_study([4, 6, 8, 10]) + adaptive_study(n0=4, nsteps=6)

    fig, ax = plt.subplots(figsize=(6, 5))
    for strategy, marker in [("uniform", "o--"), ("adaptive", "s-")]:
        pts = sorted((r["ncells"], r["error"]) for r in rows if r["strategy"] == strategy)
        ax.loglog([p[0] for p in pts], [p[1] for p in pts], marker, label=f"{strategy}: |kappa-kappa*_h|")

    ax.set_xlabel("number of cells")
    ax.set_ylabel("postprocessed eigenvalue error")
    ax.set_title("L-shaped domain: uniform vs. adaptive refinement (k=1)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "adaptive_convergence.png", dpi=150)

    with open(OUT_DIR / "adaptive_convergence.csv", "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["strategy", "step", "ncells", "kappa_h", "kappa_star_h", "eta", "error"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT_DIR / 'adaptive_convergence.png'} and {OUT_DIR / 'adaptive_convergence.csv'}")


if __name__ == "__main__":
    main()
