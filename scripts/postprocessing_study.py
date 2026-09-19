"""Reproduce Fig. 8 of Petersen (2022): the postprocessed eigenvalue
converges faster than the raw eigenvalue approximation, k = 1, 2, 3, on the
unit square with Dirichlet data.
"""
import csv
import pathlib

import matplotlib.pyplot as plt

from falk_elasticity.postprocessing import eigenvalue_pair_at

REFERENCE_KAPPA = 51.294997977322
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
NS_BY_K = {1: [8, 16, 24, 32], 2: [8, 16, 24], 3: [8, 12, 16]}


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    fig, ax = plt.subplots(figsize=(6, 5))

    for k, ns in NS_BY_K.items():
        h, err_raw, err_star = [], [], []
        for n in ns:
            hn, kappa_h, kappa_star_h = eigenvalue_pair_at(n, k, 1.0, 1.0, REFERENCE_KAPPA)
            e_raw = abs(kappa_h - REFERENCE_KAPPA)
            e_star = abs(kappa_star_h - REFERENCE_KAPPA)
            h.append(hn)
            err_raw.append(e_raw)
            err_star.append(e_star)
            rows.append(
                {
                    "k": k, "n": n, "h": hn,
                    "kappa_h": kappa_h, "kappa_star_h": kappa_star_h,
                    "error_raw": e_raw, "error_postprocessed": e_star,
                }
            )
        ax.loglog(h, err_raw, "o--", label=f"k={k}: |kappa-kappa_h|")
        ax.loglog(h, err_star, "s-", label=f"k={k}: |kappa-kappa*_h|")

    ax.set_xlabel("mesh size h")
    ax.set_ylabel("eigenvalue error")
    ax.set_title("Postprocessed vs. raw eigenvalue convergence")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "postprocessing_convergence.png", dpi=150)

    with open(OUT_DIR / "postprocessing_convergence.csv", "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["k", "n", "h", "kappa_h", "kappa_star_h", "error_raw", "error_postprocessed"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT_DIR / 'postprocessing_convergence.png'} and {OUT_DIR / 'postprocessing_convergence.csv'}")


if __name__ == "__main__":
    main()
