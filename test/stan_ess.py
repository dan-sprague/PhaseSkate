"""
Standalone reimplementation of Stan's ESS algorithm for validation.
Matches stan-dev/stan develop branch (stan/analyze/mcmc/ess.hpp).

Usage:
    python stan_ess.py < input.txt
    python stan_ess.py input.txt

Input format: first line "N M", then N*M doubles (column-major).
Output:       "ess_raw ess_bulk ess_tail"
"""

import sys
import math
import numpy as np
from scipy.stats import norm  # for inverse normal CDF


def autocovariance(x):
    """Biased autocovariance (1/N normalization), matching Stan/Geyer 1992."""
    N = len(x)
    x_centered = x - np.mean(x)
    acov = np.zeros(N)
    for lag in range(N):
        acov[lag] = np.dot(x_centered[:N - lag], x_centered[lag:]) / N
    return acov


def ess(chains):
    """
    ESS via Geyer's initial positive sequence + monotone truncation.
    chains: ndarray of shape (N, M) -- N draws per chain, M chains.
    Matches stan/analyze/mcmc/ess.hpp.
    """
    N, M = chains.shape

    if N < 4 or M < 1:
        return float(N * M)

    # Per-chain autocovariance
    acov_per_chain = []
    chain_var = np.zeros(M)
    chain_mean = np.zeros(M)
    for j in range(M):
        ac = autocovariance(chains[:, j])
        acov_per_chain.append(ac)
        chain_var[j] = ac[0] * N / (N - 1.0)  # Bessel correction
        chain_mean[j] = np.mean(chains[:, j])

    W = np.mean(chain_var)

    if M > 1:
        B = np.var(chain_mean, ddof=1)  # var of chain means
        var_plus = W * (N - 1.0) / N + B
    else:
        var_plus = W

    if var_plus <= 0.0:
        return float(N * M)

    # Geyer's initial positive sequence with monotone truncation
    rho_hat = []
    prev_pair_sum = float("inf")

    max_lag = N - 3
    t = 0
    while t < max_lag:
        mean_acov_t = sum(acov_per_chain[j][t] for j in range(M)) / M
        rho_t = 1.0 - (W - mean_acov_t) / var_plus

        if t + 1 >= N:
            rho_hat.append(rho_t)
            break

        mean_acov_t1 = sum(acov_per_chain[j][t + 1] for j in range(M)) / M
        rho_t1 = 1.0 - (W - mean_acov_t1) / var_plus

        pair_sum = rho_t + rho_t1
        if pair_sum < 0.0:
            break

        # Monotone sequence
        if pair_sum > prev_pair_sum:
            rho_t = prev_pair_sum / 2.0
            rho_t1 = prev_pair_sum / 2.0
            pair_sum = prev_pair_sum
        prev_pair_sum = pair_sum

        rho_hat.append(rho_t)
        rho_hat.append(rho_t1)
        t += 2

    tau = -1.0 + 2.0 * sum(rho_hat)

    total_draws = float(N * M)
    floor_val = 1.0 / math.log10(total_draws)
    tau = max(tau, floor_val)

    return total_draws / tau


def split_chains(chains):
    """Split each chain in half. Odd middle draw discarded (matching Stan)."""
    N, M = chains.shape
    nhalf = N // 2
    start2 = (N + 1) // 2
    split = np.zeros((nhalf, 2 * M))
    for j in range(M):
        split[:, 2 * j] = chains[:nhalf, j]
        split[:, 2 * j + 1] = chains[start2 : start2 + nhalf, j]
    return split


def rank_transform(chains):
    """Rank normalization with Blom fractional offset, matching Stan."""
    N, M = chains.shape
    total = N * M
    pooled = chains.ravel(order="F")  # column-major

    # Rank with average ties
    order = np.argsort(pooled, kind="stable")
    ranks = np.empty(total)

    k = 0
    while k < total:
        k_start = k
        while k + 1 < total and pooled[order[k + 1]] == pooled[order[k]]:
            k += 1
        avg_rank = 0.5 * (k_start + k) + 1.0  # 1-based
        for t in range(k_start, k + 1):
            ranks[order[t]] = avg_rank
        k += 1

    # Inverse normal CDF via Blom offset
    denom = total + 0.25
    z = norm.ppf((ranks - 0.375) / denom)
    return z.reshape((N, M), order="F")


def quantile_val(data, prob):
    """Linear interpolation quantile, matching Stan."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    index = prob * (n - 1)
    lo = int(math.floor(index))
    hi = min(lo + 1, n - 1)
    frac = index - lo
    return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac


def split_rank_normalized_ess(chains):
    """
    Split-rank-normalized bulk and tail ESS, matching Stan.
    Returns (ess_bulk, ess_tail).
    """
    split = split_chains(chains)

    # Bulk ESS
    ranked = rank_transform(split)
    ess_bulk = ess(ranked)

    # Tail ESS
    pooled = split.ravel(order="F")
    q05 = quantile_val(pooled, 0.05)
    q95 = quantile_val(pooled, 0.95)

    lo_ind = (split <= q05).astype(float)
    hi_ind = (split >= q95).astype(float)

    ess_lo = ess(lo_ind)
    ess_hi = ess(hi_ind)

    if math.isnan(ess_lo):
        ess_tail = ess_hi
    elif math.isnan(ess_hi):
        ess_tail = ess_lo
    else:
        ess_tail = min(ess_lo, ess_hi)

    return ess_bulk, ess_tail


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    tokens = text.split()
    N = int(tokens[0])
    M = int(tokens[1])

    data = np.array([float(x) for x in tokens[2:]])
    assert len(data) == N * M, f"Expected {N*M} values, got {len(data)}"

    # Column-major reshape
    chains = data.reshape((N, M), order="F")

    ess_raw = ess(chains)
    ess_bulk, ess_tail = split_rank_normalized_ess(chains)

    print(f"{ess_raw:.10f} {ess_bulk:.10f} {ess_tail:.10f}")


if __name__ == "__main__":
    main()
