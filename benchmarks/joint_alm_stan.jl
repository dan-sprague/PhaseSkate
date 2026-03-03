# Joint ALM Model — Stan version via StanSample.jl
# Mirror of joint_alm.jl for performance comparison
using Pkg; Pkg.activate(@__DIR__)
using StanSample
using Random
using Statistics
using Distributions: Weibull, Exponential, BetaBinomial

# ─────────────────────────────────────────────────────────────────────────────
# Stan model
# ─────────────────────────────────────────────────────────────────────────────

stan_model = "
data {
    int<lower=1> n1;
    int<lower=1> n2;
    int<lower=1> p;
    int<lower=1> n_countries;
    int<lower=1> MRC_MAX;

    // Tier 1 survival
    vector[n1] tier1_times;
    matrix[n1, p] tier1_X;
    array[n1] int<lower=1, upper=n_countries> tier1_country_ids;
    int<lower=0> n1_obs;
    int<lower=0> n1_cens;
    array[n1_obs] int<lower=1, upper=n1> tier1_obs_idx;
    array[n1_cens] int<lower=1, upper=n1> tier1_cens_idx;

    // Tier 2 survival
    vector[n2] tier2_times;
    matrix[n2, p] tier2_X;
    array[n2] int<lower=1, upper=n_countries> tier2_country_ids;
    int<lower=0> n2_obs;
    int<lower=0> n2_cens;
    array[n2_obs] int<lower=1, upper=n2> tier2_obs_idx;
    array[n2_cens] int<lower=1, upper=n2> tier2_cens_idx;

    // MRC longitudinal
    int<lower=0> total_mrc_obs;
    array[total_mrc_obs] int<lower=0, upper=MRC_MAX> mrc_scores_flat;
    vector[total_mrc_obs] mrc_times_flat;
    array[total_mrc_obs] int<lower=1, upper=n2> mrc_patient_ids;
}

parameters {
    real log_shape;
    real log_scale;
    vector[p] beta_s;
    vector[p] beta_k;
    real<lower=0> sigma_country_k;
    real<lower=0> sigma_country_s;
    vector[n_countries] mu_country_k;
    vector[n_countries] mu_country_s;
    real mu_k;
    real<lower=0> omega_k;
    real gamma_k;
    real<lower=1> gamma_hill;
    real<lower=0, upper=1> EC50;
    real log_phi;
    real<lower=0, upper=1> P0;
    vector[n2] z_k;
}

transformed parameters {
    real shape = exp(log_shape);
    real inv_shape = 1.0 / shape;
    real phi = exp(log_phi);
    real log_P0_ratio = log1p(-P0) - log(P0);
    real log_EC50g = gamma_hill * log(EC50);

    // Tier 2: log_k and log_eff_scale (vectorized)
    vector[n2] log_k_2 = mu_k + tier2_X * beta_k + mu_country_k[tier2_country_ids] + omega_k * z_k;
    vector[n2] log_eff_scale_2 = log_scale - (tier2_X * beta_s + mu_country_s[tier2_country_ids] + gamma_k * log_k_2) * inv_shape;

    // Tier 1: log_k and log_eff_scale (vectorized)
    vector[n1] log_k_1 = mu_k + tier1_X * beta_k + mu_country_k[tier1_country_ids];
    vector[n1] log_eff_scale_1 = log_scale - (tier1_X * beta_s + mu_country_s[tier1_country_ids] + gamma_k * log_k_1) * inv_shape;
}

model {
    // Priors
    log_shape ~ normal(0.2, 0.5);
    log_scale ~ normal(2.5, 1.0);
    beta_s ~ normal(0, 2);
    beta_k ~ normal(0, 0.5);
    sigma_country_k ~ cauchy(0, 0.5);
    sigma_country_s ~ cauchy(0, 1);
    mu_country_k ~ normal(0, sigma_country_k);
    mu_country_s ~ normal(0, sigma_country_s);
    mu_k ~ normal(log(0.08), 0.5);
    omega_k ~ cauchy(0, 0.5);
    gamma_k ~ normal(1.0, 0.5);
    gamma_hill ~ normal(3.0, 1.0);
    EC50 ~ beta(4.0, 6.0);
    log_phi ~ normal(log(15.0), 0.5);
    P0 ~ beta(2.0, 8.0);
    z_k ~ std_normal();

    // Tier 2 survival likelihood (vectorized)
    tier2_times[tier2_obs_idx] ~ weibull(shape, exp(log_eff_scale_2[tier2_obs_idx]));
    target += weibull_lccdf(tier2_times[tier2_cens_idx] | shape, exp(log_eff_scale_2[tier2_cens_idx]));

    // Tier 1 survival likelihood (vectorized)
    tier1_times[tier1_obs_idx] ~ weibull(shape, exp(log_eff_scale_1[tier1_obs_idx]));
    target += weibull_lccdf(tier1_times[tier1_cens_idx] | shape, exp(log_eff_scale_1[tier1_cens_idx]));

    // MRC longitudinal (beta-binomial, vectorized)
    {
        vector[total_mrc_obs] k_vec = exp(log_k_2[mrc_patient_ids]);
        vector[total_mrc_obs] P_t = inv_logit(k_vec .* mrc_times_flat - log_P0_ratio);
        vector[total_mrc_obs] log_Pg = gamma_hill * log(fmax(P_t, 1e-9));
        vector[total_mrc_obs] mu_mrc = fmin(fmax(inv_logit(log_Pg - log_EC50g), 1e-6), 1.0 - 1e-6);
        vector[total_mrc_obs] a_mrc = mu_mrc * phi;
        vector[total_mrc_obs] b_mrc = (1.0 - mu_mrc) * phi;
        for (i in 1:total_mrc_obs)
            target += beta_binomial_lpmf(mrc_scores_flat[i] | MRC_MAX, a_mrc[i], b_mrc[i]);
    }
}
";

# ─────────────────────────────────────────────────────────────────────────────
# Ground truth parameters (same as joint_alm.jl)
# ─────────────────────────────────────────────────────────────────────────────

Random.seed!(42)

const TRUE = (
    log_shape      = 0.2,
    log_scale      = 2.5,
    beta_s         = [0.3, -0.2, 0.1, -0.15],
    beta_k         = [0.4, -0.3, 0.15, -0.1],
    sigma_country_k = 0.1,
    sigma_country_s = 0.5,
    mu_country_k   = [0.05, -0.08, 0.03, 0.0],
    mu_country_s   = [0.2, -0.3, 0.1, 0.0],
    mu_k           = log(0.08),
    omega_k        = 0.3,
    gamma_k        = 1.0,
    gamma_hill     = 3.0,
    EC50           = 0.4,
    log_phi        = log(15.0),
    P0             = 0.2,
)

n1, n2 = 3500, 150
p = 4
n_countries = 4
MRC_MAX = 20

# ─────────────────────────────────────────────────────────────────────────────
# Simulate data (identical to joint_alm.jl)
# ─────────────────────────────────────────────────────────────────────────────

shape_true = exp(TRUE.log_shape)
scale_true = exp(TRUE.log_scale)
phi_true   = exp(TRUE.log_phi)

true_z_k  = randn(n2)
_ = randn(n1)  # consume RNG state to keep data identical

tier1_X = randn(n1, p)
tier1_country_ids = rand(1:n_countries, n1)
tier2_X = randn(n2, p)
tier2_country_ids = rand(1:n_countries, n2)

# Tier 2 survival
tier2_times = Vector{Float64}(undef, n2)
true_log_k_2 = Vector{Float64}(undef, n2)
for i in 1:n2
    xbk = sum(tier2_X[i, j] * TRUE.beta_k[j] for j in 1:p)
    xbs = sum(tier2_X[i, j] * TRUE.beta_s[j] for j in 1:p)
    ce_k = TRUE.mu_country_k[tier2_country_ids[i]]
    ce_s = TRUE.mu_country_s[tier2_country_ids[i]]
    true_log_k_2[i] = TRUE.mu_k + xbk + ce_k + TRUE.omega_k * true_z_k[i]
    log_eff_scale = TRUE.log_scale - (xbs + ce_s + TRUE.gamma_k * true_log_k_2[i]) / shape_true
    tier2_times[i] = rand(Weibull(shape_true, exp(log_eff_scale)))
end
cens_times_2 = rand(Exponential(median(tier2_times) * 1.5), n2)
tier2_observed = tier2_times .<= cens_times_2
tier2_times .= min.(tier2_times, cens_times_2)
tier2_obs_idx  = findall(tier2_observed)
tier2_cens_idx = findall(.!tier2_observed)

# Tier 1 survival
tier1_times = Vector{Float64}(undef, n1)
for i in 1:n1
    xbk = sum(tier1_X[i, j] * TRUE.beta_k[j] for j in 1:p)
    xbs = sum(tier1_X[i, j] * TRUE.beta_s[j] for j in 1:p)
    ce_k = TRUE.mu_country_k[tier1_country_ids[i]]
    ce_s = TRUE.mu_country_s[tier1_country_ids[i]]
    log_k_i = TRUE.mu_k + xbk + ce_k
    log_eff_scale = TRUE.log_scale - (xbs + ce_s + TRUE.gamma_k * log_k_i) / shape_true
    tier1_times[i] = rand(Weibull(shape_true, exp(log_eff_scale)))
end
cens_times_1 = rand(Exponential(median(tier1_times) * 1.5), n1)
tier1_observed = tier1_times .<= cens_times_1
tier1_times .= min.(tier1_times, cens_times_1)
tier1_obs_idx  = findall(tier1_observed)
tier1_cens_idx = findall(.!tier1_observed)

# MRC scores (longitudinal beta-binomial)
obs_per_patient = 5
total_mrc_obs = n2 * obs_per_patient
mrc_patient_ids = repeat(1:n2, inner=obs_per_patient)
mrc_times_flat  = Float64[rand() * tier2_times[mrc_patient_ids[i]] for i in 1:total_mrc_obs]

log_P0_ratio_true = log1p(-TRUE.P0) - log(TRUE.P0)
log_EC50g_true    = TRUE.gamma_hill * log(TRUE.EC50)

mrc_scores_flat = Vector{Int}(undef, total_mrc_obs)
for i in 1:total_mrc_obs
    pid   = mrc_patient_ids[i]
    k_i   = exp(true_log_k_2[pid])
    P_t   = 1.0 / (1.0 + exp(log_P0_ratio_true - k_i * mrc_times_flat[i]))
    log_Pg = TRUE.gamma_hill * log(max(P_t, 1e-9))
    mu_mrc = clamp(1.0 / (1.0 + exp(log_Pg - log_EC50g_true)), 1e-6, 1.0 - 1e-6)
    a = mu_mrc * phi_true
    b = (1.0 - mu_mrc) * phi_true
    mrc_scores_flat[i] = rand(BetaBinomial(MRC_MAX, a, b))
end

println("Data generated from ground truth.")
println("  Tier 1: $(length(tier1_obs_idx)) observed, $(length(tier1_cens_idx)) censored")
println("  Tier 2: $(length(tier2_obs_idx)) observed, $(length(tier2_cens_idx)) censored")
println("  MRC obs: $total_mrc_obs")

# ─────────────────────────────────────────────────────────────────────────────
# Run Stan
# ─────────────────────────────────────────────────────────────────────────────

stan_data = Dict(
    "n1" => n1,
    "n2" => n2,
    "p" => p,
    "n_countries" => n_countries,
    "MRC_MAX" => MRC_MAX,
    "tier1_times" => tier1_times,
    "tier1_X" => tier1_X,
    "tier1_country_ids" => tier1_country_ids,
    "n1_obs" => length(tier1_obs_idx),
    "n1_cens" => length(tier1_cens_idx),
    "tier1_obs_idx" => tier1_obs_idx,
    "tier1_cens_idx" => tier1_cens_idx,
    "tier2_times" => tier2_times,
    "tier2_X" => tier2_X,
    "tier2_country_ids" => tier2_country_ids,
    "n2_obs" => length(tier2_obs_idx),
    "n2_cens" => length(tier2_cens_idx),
    "tier2_obs_idx" => tier2_obs_idx,
    "tier2_cens_idx" => tier2_cens_idx,
    "total_mrc_obs" => total_mrc_obs,
    "mrc_scores_flat" => mrc_scores_flat,
    "mrc_times_flat" => mrc_times_flat,
    "mrc_patient_ids" => mrc_patient_ids,
)

sm = SampleModel("joint_alm_stan", stan_model)
sm.num_chains = 4
sm.num_samples = 2000
sm.num_warmups = 1000
sm.max_depth = 8
sm.seed = 42

println("\nCompiling and sampling Stan model...")
@time rc = stan_sample(sm; data=stan_data)

if success(rc)
    df = read_samples(sm, :dataframe)
    println("Stan sampling complete — $(nrow(df)) total draws")

    # ─────────────────────────────────────────────────────────────────────────
    # Posterior vs ground truth
    # ─────────────────────────────────────────────────────────────────────────

    function post_summary_stan(name, col_name, truth, df)
        vals = df[!, col_name]
        m = mean(vals)
        sorted = sort(vals)
        N = length(sorted)
        lo = sorted[max(1, round(Int, 0.025 * N))]
        hi = sorted[min(N, round(Int, 0.975 * N))]
        covered = lo <= truth <= hi
        bias = m - truth
        return (; name, truth, mean=m, lo, hi, covered, bias)
    end

    results = []
    push!(results, post_summary_stan("log_shape", "log_shape", TRUE.log_shape, df))
    push!(results, post_summary_stan("log_scale", "log_scale", TRUE.log_scale, df))
    for k in 1:p
        push!(results, post_summary_stan("beta_s[$k]", "beta_s.$k", TRUE.beta_s[k], df))
    end
    for k in 1:p
        push!(results, post_summary_stan("beta_k[$k]", "beta_k.$k", TRUE.beta_k[k], df))
    end
    push!(results, post_summary_stan("sigma_country_k", "sigma_country_k", TRUE.sigma_country_k, df))
    push!(results, post_summary_stan("sigma_country_s", "sigma_country_s", TRUE.sigma_country_s, df))
    for k in 1:n_countries
        push!(results, post_summary_stan("mu_country_k[$k]", "mu_country_k.$k", TRUE.mu_country_k[k], df))
    end
    for k in 1:n_countries
        push!(results, post_summary_stan("mu_country_s[$k]", "mu_country_s.$k", TRUE.mu_country_s[k], df))
    end
    push!(results, post_summary_stan("mu_k", "mu_k", TRUE.mu_k, df))
    push!(results, post_summary_stan("omega_k", "omega_k", TRUE.omega_k, df))
    push!(results, post_summary_stan("gamma_k", "gamma_k", TRUE.gamma_k, df))
    push!(results, post_summary_stan("gamma_hill", "gamma_hill", TRUE.gamma_hill, df))
    push!(results, post_summary_stan("EC50", "EC50", TRUE.EC50, df))
    push!(results, post_summary_stan("log_phi", "log_phi", TRUE.log_phi, df))
    push!(results, post_summary_stan("P0", "P0", TRUE.P0, df))

    println("\n", "=" ^ 82)
    println(rpad("Parameter", 18), rpad("Truth", 10), rpad("Mean", 10),
            rpad("95% CI", 22), rpad("Bias", 10), "Cover")
    println("-" ^ 82)
    n_covered = 0
    for r in results
        ci = "[$(round(r.lo; digits=3)), $(round(r.hi; digits=3))]"
        mark = r.covered ? "  ✓" : "  ✗"
        n_covered += r.covered
        println(rpad(r.name, 18),
                rpad(round(r.truth; digits=4), 10),
                rpad(round(r.mean; digits=4), 10),
                rpad(ci, 22),
                rpad(round(r.bias; digits=4), 10),
                mark)
    end
    println("-" ^ 82)
    pct = round(100 * n_covered / length(results); digits=1)
    println("Coverage: $n_covered / $(length(results)) ($pct%) of 95% CIs contain the truth")
else
    println("Stan sampling failed!")
end
