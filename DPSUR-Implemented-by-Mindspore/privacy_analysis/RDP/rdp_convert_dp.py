import math
import numpy as np


def compute_eps(orders, rdp, delta):
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if delta <= 0:
        raise ValueError("Privacy failure probability bound delta must be >0.")
    if len(orders_vec) != len(rdp_vec):
        raise ValueError("Input lists must have the same length.")
    eps_vec = []
    for (a, r) in zip(orders_vec, rdp_vec):
        if a < 1:
            raise ValueError("Renyi divergence order must be >=1.")
        if r < 0:
            raise ValueError("Renyi divergence must be >=0.")

        if delta ** 2 + math.expm1(-r) >= 0:
            eps = 0
        elif a > 1.01:
            eps = (r - (np.log(delta) + np.log(a)) / (a - 1) + np.log((a - 1) / a))
        else:
            eps = np.inf
        eps_vec.append(eps)

    idx_opt = np.argmin(eps_vec)
    return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]

def compute_eps2(orders, rdp, delta):
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if len(orders_vec) != len(rdp_vec):
        raise ValueError(
            f"Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
        )

    eps = rdp_vec - math.log(delta) / (orders_vec - 1)

    if np.isnan(eps).all():
        return np.inf, np.nan

    idx_opt = np.nanargmin(eps)
    return eps[idx_opt], orders_vec[idx_opt]
