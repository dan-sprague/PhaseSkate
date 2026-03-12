// Standalone reimplementation of Stan's ESS algorithm for validation.
// Implements: autocovariance (direct), split_chains, rank_normalization,
// ess, split_rank_normalized_ess -- matching stan-dev/stan develop branch.
//
// Input (stdin):  first line: "N M" (draws_per_chain, num_chains)
//                 then N*M doubles (column-major: all draws for chain 1, then chain 2, ...)
// Output (stdout): "ess_raw  ess_bulk  ess_tail"
//
// Compile: g++ -std=c++17 -O2 -o stan_ess stan_ess.cpp

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

// ── Matrix helper ───────────────────────────────────────────────────────────

struct Matrix {
    std::vector<double> data;
    int rows, cols;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c) : data(r * c, 0.0), rows(r), cols(c) {}

    double& operator()(int i, int j) { return data[j * rows + i]; }
    double  operator()(int i, int j) const { return data[j * rows + i]; }
    const double* col_ptr(int j) const { return data.data() + j * rows; }
};

// ── Autocovariance (biased 1/N normalization, matching Stan/Geyer 1992) ─────

static void autocovariance(const double* x, int N, std::vector<double>& acov) {
    double mean = 0.0;
    for (int i = 0; i < N; i++) mean += x[i];
    mean /= N;

    acov.resize(N);
    for (int lag = 0; lag < N; lag++) {
        double s = 0.0;
        for (int i = 0; i < N - lag; i++)
            s += (x[i] - mean) * (x[i + lag] - mean);
        acov[lag] = s / N;
    }
}

// ── ESS (Geyer's initial positive sequence + monotone truncation) ───────────
// Matches stan/analyze/mcmc/ess.hpp

static double ess(const Matrix& chains) {
    int N = chains.rows;
    int M = chains.cols;

    if (N < 4 || M < 1) return static_cast<double>(N * M);

    // Per-chain autocovariance
    std::vector<std::vector<double>> acov_per_chain(M);
    std::vector<double> chain_mean(M);
    std::vector<double> chain_var(M);

    for (int j = 0; j < M; j++) {
        autocovariance(chains.col_ptr(j), N, acov_per_chain[j]);
        chain_var[j] = acov_per_chain[j][0] * N / (N - 1.0);
        double s = 0.0;
        for (int i = 0; i < N; i++) s += chains(i, j);
        chain_mean[j] = s / N;
    }

    double W = 0.0;
    for (int j = 0; j < M; j++) W += chain_var[j];
    W /= M;

    double var_plus;
    if (M > 1) {
        double mean_of_means = 0.0;
        for (int j = 0; j < M; j++) mean_of_means += chain_mean[j];
        mean_of_means /= M;
        double B = 0.0;
        for (int j = 0; j < M; j++) {
            double d = chain_mean[j] - mean_of_means;
            B += d * d;
        }
        B /= (M - 1.0);
        var_plus = W * (N - 1.0) / N + B;
    } else {
        var_plus = W;
    }

    if (var_plus <= 0.0) return static_cast<double>(N * M);

    // Geyer's initial positive sequence with monotone truncation
    std::vector<double> rho_hat;
    double prev_pair_sum = std::numeric_limits<double>::infinity();

    int max_lag = N - 3;
    int t = 0;
    while (t < max_lag) {
        double mean_acov_t = 0.0;
        for (int j = 0; j < M; j++)
            mean_acov_t += acov_per_chain[j][t];
        mean_acov_t /= M;
        double rho_t = 1.0 - (W - mean_acov_t) / var_plus;

        if (t + 1 >= N) {
            rho_hat.push_back(rho_t);
            break;
        }

        double mean_acov_t1 = 0.0;
        for (int j = 0; j < M; j++)
            mean_acov_t1 += acov_per_chain[j][t + 1];
        mean_acov_t1 /= M;
        double rho_t1 = 1.0 - (W - mean_acov_t1) / var_plus;

        double pair_sum = rho_t + rho_t1;

        if (pair_sum < 0.0) break;

        // Monotone sequence: enforce pair_sum <= prev_pair_sum
        if (pair_sum > prev_pair_sum) {
            rho_t  = prev_pair_sum / 2.0;
            rho_t1 = prev_pair_sum / 2.0;
            pair_sum = prev_pair_sum;
        }
        prev_pair_sum = pair_sum;

        rho_hat.push_back(rho_t);
        rho_hat.push_back(rho_t1);
        t += 2;
    }

    // tau = -1 + 2 * sum(rho_hat)   (rho[0] included in sum)
    double tau = -1.0;
    for (double r : rho_hat) tau += 2.0 * r;

    // Safety floor
    double total_draws = static_cast<double>(N * M);
    double floor_val = 1.0 / std::log10(total_draws);
    if (tau < floor_val) tau = floor_val;

    return total_draws / tau;
}

// ── Split chains ────────────────────────────────────────────────────────────

static Matrix split_chains(const Matrix& chains) {
    int N = chains.rows;
    int M = chains.cols;
    int nhalf = N / 2;
    int start2 = (N + 1) / 2;

    Matrix split(nhalf, 2 * M);
    for (int j = 0; j < M; j++) {
        for (int i = 0; i < nhalf; i++) {
            split(i, 2 * j)     = chains(i, j);
            split(i, 2 * j + 1) = chains(start2 + i, j);
        }
    }
    return split;
}

// ── Inverse normal CDF (Acklam's rational approximation) ───────────────────

static double inv_normal_cdf(double p) {
    static const double a[] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    };
    static const double b[] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    };
    static const double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    static const double d[] = {
        7.784695709041462e-03,  3.224671290700398e-01,
        2.445134137142996e+00,  3.754408661907416e+00
    };

    if (p < 0.02425) {
        double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= 0.97575) {
        double q = p - 0.5;
        double r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}

// ── Rank normalization (Blom fractional offset, matching Stan) ──────────────

static Matrix rank_transform(const Matrix& chains) {
    int N = chains.rows;
    int M = chains.cols;
    int total = N * M;

    std::vector<std::pair<double, int>> vals(total);
    for (int j = 0; j < M; j++)
        for (int i = 0; i < N; i++)
            vals[j * N + i] = {chains(i, j), j * N + i};

    std::sort(vals.begin(), vals.end());

    // Average ranks for ties
    std::vector<double> ranks(total);
    int k = 0;
    while (k < total) {
        int k_start = k;
        while (k + 1 < total && vals[k + 1].first == vals[k].first)
            k++;
        double avg_rank = 0.5 * (k_start + k) + 1.0;
        for (int t = k_start; t <= k; t++)
            ranks[vals[t].second] = avg_rank;
        k++;
    }

    Matrix result(N, M);
    double denom = total + 0.25;
    for (int j = 0; j < M; j++)
        for (int i = 0; i < N; i++) {
            double p = (ranks[j * N + i] - 0.375) / denom;
            result(i, j) = inv_normal_cdf(p);
        }
    return result;
}

// ── Quantile (linear interpolation) ─────────────────────────────────────────

static double quantile_val(const double* data, int n, double prob) {
    std::vector<double> sorted(data, data + n);
    std::sort(sorted.begin(), sorted.end());
    double index = prob * (n - 1);
    int lo = static_cast<int>(std::floor(index));
    int hi = lo + 1;
    double frac = index - lo;
    if (hi >= n) return sorted[n - 1];
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

// ── Split-rank-normalized ESS (bulk and tail) ───────────────────────────────

static void split_rank_normalized_ess(const Matrix& chains,
                                       double& ess_bulk, double& ess_tail) {
    Matrix split = split_chains(chains);

    // Bulk ESS
    Matrix ranked = rank_transform(split);
    ess_bulk = ess(ranked);

    // Tail ESS
    int N = split.rows;
    int M = split.cols;
    int total = N * M;

    std::vector<double> pooled(total);
    for (int j = 0; j < M; j++)
        for (int i = 0; i < N; i++)
            pooled[j * N + i] = split(i, j);

    double q05 = quantile_val(pooled.data(), total, 0.05);
    double q95 = quantile_val(pooled.data(), total, 0.95);

    Matrix lo_ind(N, M), hi_ind(N, M);
    for (int j = 0; j < M; j++)
        for (int i = 0; i < N; i++) {
            lo_ind(i, j) = split(i, j) <= q05 ? 1.0 : 0.0;
            hi_ind(i, j) = split(i, j) >= q95 ? 1.0 : 0.0;
        }

    double ess_lo = ess(lo_ind);
    double ess_hi = ess(hi_ind);

    if (std::isnan(ess_lo)) ess_tail = ess_hi;
    else if (std::isnan(ess_hi)) ess_tail = ess_lo;
    else ess_tail = std::min(ess_lo, ess_hi);
}

// ── Main ────────────────────────────────────────────────────────────────────

int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) {
        fprintf(stderr, "Expected: N M on first line\n");
        return 1;
    }

    Matrix chains(N, M);
    for (int j = 0; j < M; j++)
        for (int i = 0; i < N; i++)
            if (scanf("%lf", &chains(i, j)) != 1) {
                fprintf(stderr, "Failed to read data at (%d, %d)\n", i, j);
                return 1;
            }

    double ess_raw = ess(chains);

    double ess_bulk, ess_tail;
    split_rank_normalized_ess(chains, ess_bulk, ess_tail);

    printf("%.10f %.10f %.10f\n", ess_raw, ess_bulk, ess_tail);
    return 0;
}
