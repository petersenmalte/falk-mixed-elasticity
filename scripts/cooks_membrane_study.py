"""Reproduce Petersen (2022) Section 7.3: Cook's membrane, a tapered panel
clamped on the left edge and free elsewhere. Unlike the square and
L-shaped domains, there's no independent reference eigenvalue here, so
this reports the smallest eigenvalue's self-convergence under refinement,
k = 1, 2, 3.
"""
import csv
import pathlib

import matplotlib.pyplot as plt

from falk_elasticity.domains import cooks_membrane_eigenvalue_pair_at

LMBDA, MU = 1.0, 1.0
TARGET_KAPPA = 0.005
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
NS_BY_K = {1: [4, 6, 8, 10], 2: [4, 6, 8], 3: [4, 6]}


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    fig, ax = plt.subplots(figsize=(6, 5))

    for k, ns in NS_BY_K.items():
        values, values_star = [], []
        for n in ns:
            kappa_h, kappa_star_h = cooks_membrane_eigenvalue_pair_at(n, k, LMBDA, MU, TARGET_KAPPA)
            values.append(kappa_h)
            values_star.append(kappa_star_h)
            rows.append({"k": k, "n": n, "kappa_h": kappa_h, "kappa_star_h": kappa_star_h})
        ax.plot(ns, values, "o--", label=f"k={k}: kappa_h")
        ax.plot(ns, values_star, "s-", label=f"k={k}: kappa*_h")

    ax.set_xlabel("N (mesh resolution)")
    ax.set_ylabel("smallest eigenvalue near target")
    ax.set_title("Cook's membrane: eigenvalue self-convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cooks_membrane_convergence.png", dpi=150)

    with open(OUT_DIR / "cooks_membrane_convergence.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["k", "n", "kappa_h", "kappa_star_h"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT_DIR / 'cooks_membrane_convergence.png'} and {OUT_DIR / 'cooks_membrane_convergence.csv'}")


if __name__ == "__main__":
    main()
