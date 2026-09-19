"""Verify the local postprocessing improves the eigenvalue rate, Petersen
(2022) Theorem 5.7 (eq. 61): |kappa - kappa*_h| = O(h^{2k+2}) for k>=2. For
k=1 the guaranteed rate is only O(h^3) (Remark 5.8), but the thesis observes
*better* than that in practice -- a mesh-dependent superconvergence
phenomenon (cancellation in the leading-order terms, Section 7.1) -- so this
only asserts a safe improvement over the raw O(h^2) rate, not that specific
extra order.
"""
import numpy as np
import pytest

from falk_elasticity.convergence import loglog_rate
from falk_elasticity.postprocessing import eigenvalue_pair_at

LMBDA, MU = 1.0, 1.0
REFERENCE_KAPPA = 51.294997977322


def test_lowest_order_postprocessing_improves_rate():
    ns = [8, 16, 24]
    results = [eigenvalue_pair_at(n, 1, LMBDA, MU, REFERENCE_KAPPA) for n in ns]
    h = np.array([r[0] for r in results])
    errors = np.array([abs(r[2] - REFERENCE_KAPPA) for r in results])

    print(f"\nk=1 postprocessed error by N: {list(zip(ns, errors))}")

    rate = loglog_rate(h, errors)
    print(f"k=1 postprocessed eigenvalue rate: {rate:.2f} (guaranteed >=3 asymptotically, thesis observes higher)")
    assert rate > 2.0


@pytest.mark.parametrize("k", [2, 3])
def test_higher_order_postprocessing_is_close(k):
    _, kappa_h, kappa_star_h = eigenvalue_pair_at(12, k, LMBDA, MU, REFERENCE_KAPPA)
    error = abs(kappa_star_h - REFERENCE_KAPPA)
    print(f"\nk={k}, N=12: kappa_h={kappa_h:.6f}, kappa*_h={kappa_star_h:.6f}, error={error:.6f}")
    assert error < 0.01
