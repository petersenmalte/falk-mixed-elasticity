"""Verify Petersen (2022) eq. (1), the Falk element's a priori rates
    ||sigma - sigma_h||_0 + ||gamma - gamma_h||_0 = O(h^{k+1})
    ||u - u_h||_0                                 = O(h^k)
on the unit square with a manufactured trigonometric solution.
"""
import numpy as np
import pytest

from falk_elasticity.convergence import errors_at, loglog_rate

LMBDA, MU = 1.0, 1.0


@pytest.mark.parametrize("k", [1, 2, 3])
def test_falk_element_convergence_rates(k):
    ns = [4, 8, 16]
    results = [errors_at(n, k, LMBDA, MU) for n in ns]
    h = np.array([r[0] for r in results])
    e_sigma = np.array([r[1] for r in results])
    e_gamma = np.array([r[2] for r in results])
    e_u = np.array([r[3] for r in results])

    rate_sigma = loglog_rate(h, e_sigma)
    rate_gamma = loglog_rate(h, e_gamma)
    rate_u = loglog_rate(h, e_u)

    print(
        f"\nk={k}: rate(sigma)={rate_sigma:.2f} (expect {k + 1}), "
        f"rate(gamma)={rate_gamma:.2f} (expect {k + 1}), "
        f"rate(u)={rate_u:.2f} (expect {k})"
    )

    tol = 0.25
    assert rate_sigma > (k + 1) - tol
    assert rate_gamma > (k + 1) - tol
    assert rate_u > k - tol
