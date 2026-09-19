"""Reproduce the O(h^{k+1}) / O(h^k) rates of Petersen (2022), eq. (1), for
the Falk element source problem, k = 1, 2, 3: a log-log convergence plot plus
the raw numbers -- input data for the convergence-plot explorer planned for
petersenmalte.de.
"""
import csv
import pathlib

import matplotlib.pyplot as plt

from falk_elasticity.convergence import errors_at

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
NS_BY_K = {1: [4, 8, 16, 32, 64], 2: [4, 8, 16, 32], 3: [4, 8, 16]}


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    fig, ax = plt.subplots(figsize=(6, 5))

    for k, ns in NS_BY_K.items():
        data = [errors_at(n, k) for n in ns]
        for n, (h, e_sigma, e_gamma, e_u) in zip(ns, data):
            rows.append(
                {"k": k, "n": n, "h": h, "e_sigma": e_sigma, "e_gamma": e_gamma, "e_u": e_u}
            )
        h = [d[0] for d in data]
        ax.loglog(h, [d[1] for d in data], "o-", label=f"k={k}: ||sigma-sigma_h||_0")
        ax.loglog(h, [d[3] for d in data], "s--", label=f"k={k}: ||u-u_h||_0")

    ax.set_xlabel("mesh size h")
    ax.set_ylabel("L2 error")
    ax.set_title("Falk element, unit square, manufactured solution")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "convergence.png", dpi=150)

    with open(OUT_DIR / "convergence.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["k", "n", "h", "e_sigma", "e_gamma", "e_u"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT_DIR / 'convergence.png'} and {OUT_DIR / 'convergence.csv'}")


if __name__ == "__main__":
    main()
