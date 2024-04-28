import numpy as np
import math
from scipy import special
import os
def cartesian_to_polar(x):
    r = np.linalg.norm(x)
    theta = np.arccos(x[0] / r)
    phi = [1. for i in range(len(x) - 1)]
    for i in range(len(phi)):
        phi[i] = np.arctan2(x[i + 1], x[0])
    return np.concatenate(([r, theta], phi))


def polar_to_cartesian(p):
    r = p[0]
    theta = p[1]
    phi = p[2:]
    x = [1. for i in range(len(phi) + 1)]
    x[0] = r * np.cos(theta)
    for i in range(len(phi)):
        x[i + 1] = x[0] * np.tan(phi[i])
    for j in range(len(x)):
        x[j] = round(x[j], 4)
    return x


def vector_to_matrix(vector, shape):
    shape = tuple(shape)
    if len(shape) == 0 or np.prod(shape) != len(vector):
        raise ValueError("Invalid input dimensions")
    matrix = np.zeros(shape)
    strides = [np.prod(shape[i + 1:]) for i in range(len(shape) - 1)] + [1]
    for i in range(len(vector)):
        index = [0] * len(shape)
        for j in range(len(shape)):
            index[j] = (i // strides[j]) % shape[j]
        matrix[tuple(index)] = vector[i]
    return matrix

def cartesian_add_noise(p, sigma1, C1, sigma2):
    r = p[0]
    r += C1 * sigma1 * np.random.normal(0, 1)

    theta = p[1:]
    theta += 2 * math.pi * sigma2 * np.random.normal(0, 1)

    return np.concatenate(([r], theta))


# 划分eps
def devide_epslion(sigma, q, n):
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
    eps, opt_order = apply_dp_sgd_analysis(q, sigma, 1, orders, 10 ** (-5))
    eps_sum = n * eps
    eps1 = eps_sum * 0.000001
    eps2 = eps_sum - eps1
    sigma1 = get_noise_multiplier(target_epsilon=eps1, target_delta=1e-5, sample_rate=512 / 60000, steps=1,
                                  alphas=orders)
    sigma2 = get_noise_multiplier(target_epsilon=eps2, target_delta=1e-5, sample_rate=512 / 60000, steps=1,
                                  alphas=orders)
    return sigma1, sigma2


def get_noise_multiplier(
        target_epsilon: float,
        target_delta: float,
        sample_rate: float,
        steps: int,
        alphas,
        epsilon_tolerance: float = 0.01,
) -> float:
    sigma_low, sigma_high = 0, 1000

    eps_high, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma_high, steps, alphas, target_delta)

    if eps_high > target_epsilon:
        raise ValueError("The target privacy budget is too low. 当前可供搜索的最大的sigma只到100")

    while target_epsilon - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2

        eps, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma, steps, alphas, target_delta)

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return round(sigma_high, 2)


def compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta):
    q = batch_size / n
    if q > 1:
        print('n must be larger than the batch size.')
    orders = (list(range(2, 64)) + [128, 256, 512])
    steps = int(math.ceil(epochs * (n / batch_size)))

    return apply_dp_sgd_analysis(q, noise_multiplier, steps, orders, delta)


def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
    rdp = compute_rdp(q, sigma, steps, orders)

    eps, opt_order = compute_eps(orders, rdp, delta)
    return eps, opt_order


def compute_rdp(q, noise_multiplier, steps, orders):
    if np.isscalar(orders):
        rdp = _compute_rdp(q, noise_multiplier, orders)
    else:
        rdp = np.array(
            [_compute_rdp(q, noise_multiplier, order) for order in orders])

    return rdp * steps


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
        return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx


def _log_erfc(x: float) -> float:
    return math.log(2) + special.log_ndtr(-x * 2 ** 0.5)

ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

def scatter_normalization(train_loader, scattering, K, device,
                          data_size, sample_size,
                          noise_multiplier=1.0, orders=ORDERS, save_dir=None):
    rdp = 0
    epsilon_norm = np.inf
    delta=1e-5
    if noise_multiplier > 0:
        sample_rate = sample_size / (1.0 * data_size)
        rdp = 2*compute_rdp(sample_rate, noise_multiplier, 1, orders)
        epsilon_norm, best_alpha = compute_eps(orders, rdp , delta)

    print(scattering)

    use_scattering = scattering is not None
    assert use_scattering
    mean_path = os.path.join(save_dir, f"mean_bn_{sample_size}_{noise_multiplier}_{use_scattering}.npy")
    var_path = os.path.join(save_dir, f"var_bn_{sample_size}_{noise_multiplier}_{use_scattering}.npy")

    print(f"Using BN stats for {sample_size}/{data_size} samples")
    print(f"With noise_mul={noise_multiplier}, we get ε_norm = {epsilon_norm:.3f}")

    try:
        print(f"loading {mean_path}")
        mean = np.load(mean_path)
        var = np.load(var_path)
        print(mean.shape, var.shape)
    except OSError:
        scatters = []
        mean = 0
        sq_mean = 0
        count = 0
        cnt = 0
        for data in train_loader[0]:
            if scattering is not None:
                print(data.shape)
                data = scattering(data).reshape(-1, K, data.shape[2]//4, data.shape[3]//4)
            if noise_multiplier == 0:
                data = data.reshape(len(data), K, -1).mean(-1)
                mean += data.sum(0).cpu().numpy()
                sq_mean += (data**2).sum(0).cpu().numpy()
            else:
                scatters.append(data)
            count += len(data)
            if count >= sample_size:
                break

        if noise_multiplier > 0:
            scatters = np.concatenate(scatters, axis=0)
            scatters = np.transpose(scatters, (0, 2, 3, 1))

            scatters = scatters[:sample_size]

            scatter_means = np.mean(scatters.reshape(len(scatters), -1, K), axis=1)
            norms = np.linalg.norm(scatter_means, axis=-1)

            thresh_mean = np.quantile(norms, 0.5)
            scatter_means /= np.maximum(norms / thresh_mean, 1).reshape(-1, 1)
            mean = np.mean(scatter_means, axis=0)

            mean += np.random.normal(scale=thresh_mean * noise_multiplier,
                                     size=mean.shape) / sample_size

            scatter_sq_means = np.mean((scatters ** 2).reshape(len(scatters), -1, K),
                                       axis=1)
            norms = np.linalg.norm(scatter_sq_means, axis=-1)

            thresh_var = np.quantile(norms, 0.5)
            print(f"thresh_mean={thresh_mean:.2f}, thresh_var={thresh_var:.2f}")
            scatter_sq_means /= np.maximum(norms / thresh_var, 1).reshape(-1, 1)
            sq_mean = np.mean(scatter_sq_means, axis=0)
            sq_mean += np.random.normal(scale=thresh_var * noise_multiplier,
                                        size=sq_mean.shape) / sample_size
            var = np.maximum(sq_mean - mean ** 2, 0)
        else:
            mean /= count
            sq_mean /= count
            var = np.maximum(sq_mean - mean ** 2, 0)

        if save_dir is not None:
            print(f"saving mean and var: {mean.shape} {var.shape}")
            np.save(mean_path, mean)
            np.save(var_path, var)


    return (mean, var), rdp