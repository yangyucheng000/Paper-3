import numpy as np
import math
from scipy import special
from privacy_analysis.RDP.rdp_convert_dp import compute_eps


def compute_rdp(q, noise_multiplier, steps, orders):
    if np.isscalar(orders):
        rdp = _compute_rdp(q, noise_multiplier, orders)
    else:
        rdp = np.array(
            [_compute_rdp(q, noise_multiplier, order) for order in orders])

    return rdp * steps


# 整型
def _compute_log_a_for_int_alpha(q, sigma, alpha):
    assert isinstance(alpha, int)
    rdp = -np.inf

    for i in range(alpha + 1):
        log_b = (
                math.log(special.binom(alpha, i))
                + i * math.log(q)
                + (alpha - i) * math.log(1 - q)
                + (i * i - i) / (2 * (sigma ** 2))
        )
        a, b = min(rdp, log_b), max(rdp, log_b)
        if a == -np.inf:
            rdp = b
        else:
            rdp = math.log(math.exp(
                a - b) + 1) + b

    rdp = float(rdp) / (alpha - 1)
    return rdp


def _log_add(logx: float, logy: float) -> float:
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:
        return b
    return math.log1p(math.exp(a - b)) + b


def _log_sub(logx: float, logy: float) -> float:
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:
        return logx
    if logx == logy:
        return -np.inf

    try:
        return math.log(math.expm1(logx - logy)) + logy
    except OverflowError:
        return logx


def _log_erfc(x: float) -> float:
    return math.log(2) + special.log_ndtr(-x * 2 ** 0.5)


def _compute_log_a_for_frac_alpha(q: float, sigma: float, alpha: float) -> float:
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0

    z0 = sigma ** 2 * math.log(1 / q - 1) + 0.5

    while True:
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma ** 2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma ** 2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1) / (alpha - 1)

def _compute_rdp(q, sigma, alpha):
    if q == 0:
        return 0
    if sigma == 0:
        return np.inf
    if q == 1.:
        return alpha / (
                    2 * sigma ** 2)

    if np.isinf(alpha):
        return np.inf

    if float(alpha).is_integer():
        return _compute_log_a_for_int_alpha(q, sigma, int(alpha))
    else:
        return _compute_log_a_for_frac_alpha(q, sigma, alpha)


def compute_rdp_randomized_response(p, steps, orders, q):
    if np.isscalar(orders):
        rdp = _compute_rdp_randomized_response(p, orders)
    else:
        rdp = np.array([_compute_rdp_randomized_response(p, order) for order in orders])

    return q * rdp * steps


def _compute_rdp_randomized_response(p, alpha):
    item1 = float((p ** alpha) * ((1 - p) ** (1 - alpha)))
    item2 = float(((1 - p) ** alpha) * (p ** (1 - alpha)))
    rdp = float(math.log(item1 + item2)) / (alpha - 1)
    return rdp


if __name__ == '__main__':
    ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 128))
    p = 0.7
    steps = 100
    q = 3000 / 60000
    delta = 1e-5
    rdp = compute_rdp_randomized_response(p, steps, ORDERS, q)
    dp, ord = compute_eps(ORDERS, rdp, delta)

    print("rdp:", dp)
    print("ORDERS:", ord)
