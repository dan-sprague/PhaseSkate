"""
NumPyro / BlackJAX implementation of the two-tier survival + MRC longitudinal model.
Translated from the Stan / PhaseSkate Julia version.

Usage:
    python joint_alm_numpyro.py                  # NumPyro NUTS (default)
    python joint_alm_numpyro.py --sampler blackjax  # BlackJAX NUTS
"""

import argparse
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import initialize_model
from scipy.stats import weibull_min, betabinom as sp_betabinom

# ─────────────────────────────────────────────────────────────────────────────
# Weibull helpers (work directly with log-scale for numerical stability)
# ─────────────────────────────────────────────────────────────────────────────


def weibull_logscale_lpdf(t, shape, log_scale):
    """Weibull log-pdf parameterised by log(scale).  Vectorised."""
    log_t = jnp.log(t)
    return (
        jnp.log(shape)
        - shape * log_scale
        + (shape - 1.0) * log_t
        - jnp.exp(shape * (log_t - log_scale))
    )


def weibull_logscale_lccdf(t, shape, log_scale):
    """Weibull log-CCDF (log survival) parameterised by log(scale)."""
    return -jnp.exp(shape * (jnp.log(t) - log_scale))


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth parameters
# ─────────────────────────────────────────────────────────────────────────────

TRUE_PARAMS = dict(
    log_shape=0.2,
    log_scale=2.5,
    beta_s=np.array([0.3, -0.2, 0.1, -0.15]),
    beta_k=np.array([0.4, -0.3, 0.15, -0.1]),
    sigma_country_k=0.1,
    sigma_country_s=0.5,
    mu_country_k=np.array([0.05, -0.08, 0.03, 0.0]),
    mu_country_s=np.array([0.2, -0.3, 0.1, 0.0]),
    mu_k=np.log(0.08),
    omega_k=0.3,
    gamma_k=1.0,
    gamma_hill=3.0,
    EC50=0.4,
    log_phi=np.log(15.0),
    P0=0.2,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data generation  (mirrors the Julia code, different RNG stream)
# ─────────────────────────────────────────────────────────────────────────────


def generate_data(seed=42):
    rng = np.random.RandomState(seed)

    n1, n2, p, n_countries, MRC_MAX = 3500, 150, 4, 4, 20
    T = TRUE_PARAMS

    shape = np.exp(T["log_shape"])
    phi = np.exp(T["log_phi"])

    z_k = rng.randn(n2)
    _ = rng.randn(n1)  # consume RNG state (matches Julia script)

    tier1_X = rng.randn(n1, p)
    tier1_country_ids = rng.randint(0, n_countries, n1)  # 0-based
    tier2_X = rng.randn(n2, p)
    tier2_country_ids = rng.randint(0, n_countries, n2)

    # -- Tier 2 survival --
    tier2_times = np.empty(n2)
    true_log_k_2 = np.empty(n2)
    for i in range(n2):
        xbk = tier2_X[i] @ T["beta_k"]
        xbs = tier2_X[i] @ T["beta_s"]
        ce_k = T["mu_country_k"][tier2_country_ids[i]]
        ce_s = T["mu_country_s"][tier2_country_ids[i]]
        true_log_k_2[i] = T["mu_k"] + xbk + ce_k + T["omega_k"] * z_k[i]
        log_eff_scale = T["log_scale"] - (
            xbs + ce_s + T["gamma_k"] * true_log_k_2[i]
        ) / shape
        tier2_times[i] = weibull_min.rvs(
            shape, scale=np.exp(log_eff_scale), random_state=rng
        )

    cens_times_2 = rng.exponential(np.median(tier2_times) * 1.5, n2)
    tier2_observed = tier2_times <= cens_times_2
    tier2_times = np.minimum(tier2_times, cens_times_2)
    tier2_obs_idx = np.where(tier2_observed)[0]
    tier2_cens_idx = np.where(~tier2_observed)[0]

    # -- Tier 1 survival --
    tier1_times = np.empty(n1)
    for i in range(n1):
        xbk = tier1_X[i] @ T["beta_k"]
        xbs = tier1_X[i] @ T["beta_s"]
        ce_k = T["mu_country_k"][tier1_country_ids[i]]
        ce_s = T["mu_country_s"][tier1_country_ids[i]]
        log_k_i = T["mu_k"] + xbk + ce_k
        log_eff_scale = T["log_scale"] - (
            xbs + ce_s + T["gamma_k"] * log_k_i
        ) / shape
        tier1_times[i] = weibull_min.rvs(
            shape, scale=np.exp(log_eff_scale), random_state=rng
        )

    cens_times_1 = rng.exponential(np.median(tier1_times) * 1.5, n1)
    tier1_observed = tier1_times <= cens_times_1
    tier1_times = np.minimum(tier1_times, cens_times_1)
    tier1_obs_idx = np.where(tier1_observed)[0]
    tier1_cens_idx = np.where(~tier1_observed)[0]

    # -- MRC longitudinal (beta-binomial) --
    obs_per_patient = 5
    total_mrc_obs = n2 * obs_per_patient
    mrc_patient_ids = np.repeat(np.arange(n2), obs_per_patient)  # 0-based
    mrc_times_flat = np.array(
        [rng.uniform(0, tier2_times[mrc_patient_ids[i]]) for i in range(total_mrc_obs)]
    )

    log_P0_ratio = np.log1p(-T["P0"]) - np.log(T["P0"])
    log_EC50g = T["gamma_hill"] * np.log(T["EC50"])

    mrc_scores_flat = np.empty(total_mrc_obs, dtype=int)
    for i in range(total_mrc_obs):
        pid = mrc_patient_ids[i]
        k_i = np.exp(true_log_k_2[pid])
        P_t = 1.0 / (1.0 + np.exp(log_P0_ratio - k_i * mrc_times_flat[i]))
        log_Pg = T["gamma_hill"] * np.log(max(P_t, 1e-9))
        mu_mrc = np.clip(
            1.0 / (1.0 + np.exp(log_EC50g - log_Pg)), 1e-6, 1 - 1e-6
        )
        a = mu_mrc * phi
        b = (1.0 - mu_mrc) * phi
        mrc_scores_flat[i] = sp_betabinom.rvs(MRC_MAX, a, b, random_state=rng)

    print("Data generated from ground truth.")
    print(f"  Tier 1: {len(tier1_obs_idx)} observed, {len(tier1_cens_idx)} censored")
    print(f"  Tier 2: {len(tier2_obs_idx)} observed, {len(tier2_cens_idx)} censored")
    print(f"  MRC obs: {total_mrc_obs}")

    # Return JAX arrays for model consumption
    return dict(
        n1=n1,
        n2=n2,
        p=p,
        n_countries=n_countries,
        MRC_MAX=MRC_MAX,
        tier1_times=jnp.array(tier1_times),
        tier1_X=jnp.array(tier1_X),
        tier1_country_ids=jnp.array(tier1_country_ids),
        tier1_obs_idx=jnp.array(tier1_obs_idx),
        tier1_cens_idx=jnp.array(tier1_cens_idx),
        tier2_times=jnp.array(tier2_times),
        tier2_X=jnp.array(tier2_X),
        tier2_country_ids=jnp.array(tier2_country_ids),
        tier2_obs_idx=jnp.array(tier2_obs_idx),
        tier2_cens_idx=jnp.array(tier2_cens_idx),
        total_mrc_obs=total_mrc_obs,
        mrc_scores_flat=jnp.array(mrc_scores_flat),
        mrc_times_flat=jnp.array(mrc_times_flat),
        mrc_patient_ids=jnp.array(mrc_patient_ids),
    )


# ─────────────────────────────────────────────────────────────────────────────
# NumPyro model
# ─────────────────────────────────────────────────────────────────────────────


def model(
    n1,
    n2,
    p,
    n_countries,
    MRC_MAX,
    tier1_times,
    tier1_X,
    tier1_country_ids,
    tier1_obs_idx,
    tier1_cens_idx,
    tier2_times,
    tier2_X,
    tier2_country_ids,
    tier2_obs_idx,
    tier2_cens_idx,
    total_mrc_obs,
    mrc_scores_flat,
    mrc_times_flat,
    mrc_patient_ids,
):
    # ── Priors ────────────────────────────────────────────────────────────────
    log_shape = numpyro.sample("log_shape", dist.Normal(0.2, 0.5))
    log_scale = numpyro.sample("log_scale", dist.Normal(2.5, 1.0))
    beta_s = numpyro.sample("beta_s", dist.Normal(0, 2).expand([p]))
    beta_k = numpyro.sample("beta_k", dist.Normal(0, 0.5).expand([p]))

    sigma_country_k = numpyro.sample("sigma_country_k", dist.HalfCauchy(0.5))
    sigma_country_s = numpyro.sample("sigma_country_s", dist.HalfCauchy(1.0))
    mu_country_k = numpyro.sample(
        "mu_country_k", dist.Normal(0, sigma_country_k).expand([n_countries])
    )
    mu_country_s = numpyro.sample(
        "mu_country_s", dist.Normal(0, sigma_country_s).expand([n_countries])
    )

    mu_k = numpyro.sample("mu_k", dist.Normal(jnp.log(0.08), 0.5))
    omega_k = numpyro.sample("omega_k", dist.HalfCauchy(0.5))
    gamma_k = numpyro.sample("gamma_k", dist.Normal(1.0, 0.5))
    gamma_hill = numpyro.sample(
        "gamma_hill", dist.TruncatedNormal(3.0, 1.0, low=1.0)
    )
    EC50 = numpyro.sample("EC50", dist.Beta(4.0, 6.0))
    log_phi = numpyro.sample("log_phi", dist.Normal(jnp.log(15.0), 0.5))
    P0 = numpyro.sample("P0", dist.Beta(2.0, 8.0))
    z_k = numpyro.sample("z_k", dist.Normal(0, 1).expand([n2]))

    # ── Transformed parameters ────────────────────────────────────────────────
    shape = jnp.exp(log_shape)
    inv_shape = 1.0 / shape
    phi = jnp.exp(log_phi)
    log_P0_ratio = jnp.log1p(-P0) - jnp.log(P0)
    log_EC50g = gamma_hill * jnp.log(EC50)

    # Tier 2: log_k and log effective scale
    log_k_2 = (
        mu_k
        + tier2_X @ beta_k
        + mu_country_k[tier2_country_ids]
        + omega_k * z_k
    )
    log_eff_scale_2 = log_scale - (
        tier2_X @ beta_s
        + mu_country_s[tier2_country_ids]
        + gamma_k * log_k_2
    ) * inv_shape

    # Tier 1: log_k and log effective scale
    log_k_1 = mu_k + tier1_X @ beta_k + mu_country_k[tier1_country_ids]
    log_eff_scale_1 = log_scale - (
        tier1_X @ beta_s
        + mu_country_s[tier1_country_ids]
        + gamma_k * log_k_1
    ) * inv_shape

    # ── Tier 2 survival likelihood ────────────────────────────────────────────
    numpyro.factor(
        "tier2_obs_ll",
        jnp.sum(
            weibull_logscale_lpdf(
                tier2_times[tier2_obs_idx], shape, log_eff_scale_2[tier2_obs_idx]
            )
        ),
    )
    numpyro.factor(
        "tier2_cens_ll",
        jnp.sum(
            weibull_logscale_lccdf(
                tier2_times[tier2_cens_idx], shape, log_eff_scale_2[tier2_cens_idx]
            )
        ),
    )

    # ── Tier 1 survival likelihood ────────────────────────────────────────────
    numpyro.factor(
        "tier1_obs_ll",
        jnp.sum(
            weibull_logscale_lpdf(
                tier1_times[tier1_obs_idx], shape, log_eff_scale_1[tier1_obs_idx]
            )
        ),
    )
    numpyro.factor(
        "tier1_cens_ll",
        jnp.sum(
            weibull_logscale_lccdf(
                tier1_times[tier1_cens_idx], shape, log_eff_scale_1[tier1_cens_idx]
            )
        ),
    )

    # ── MRC longitudinal (beta-binomial) ──────────────────────────────────────
    k_vec = jnp.exp(log_k_2[mrc_patient_ids])
    P_t = jsp.expit(k_vec * mrc_times_flat - log_P0_ratio)
    log_Pg = gamma_hill * jnp.log(jnp.maximum(P_t, 1e-9))
    mu_mrc = jnp.clip(jsp.expit(log_Pg - log_EC50g), 1e-6, 1.0 - 1e-6)
    a_mrc = mu_mrc * phi
    b_mrc = (1.0 - mu_mrc) * phi

    numpyro.sample(
        "mrc_obs",
        dist.BetaBinomial(a_mrc, b_mrc, total_count=MRC_MAX),
        obs=mrc_scores_flat,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sampler: NumPyro NUTS
# ─────────────────────────────────────────────────────────────────────────────


def run_numpyro_nuts(data, num_warmup=1000, num_samples=1000, num_chains=4, seed=0):
    kernel = NUTS(model, max_tree_depth=10, target_accept_prob=0.8)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(jax.random.PRNGKey(seed), **data)
    mcmc.print_summary()
    return mcmc.get_samples(group_by_chain=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sampler: BlackJAX NUTS
# ─────────────────────────────────────────────────────────────────────────────


def run_blackjax_nuts(data, num_warmup=1000, num_samples=1000, num_chains=4, seed=0):
    import blackjax
    from blackjax.util import run_inference_algorithm
    import time

    rng_key = jax.random.PRNGKey(seed)
    init_key, run_key = jax.random.split(rng_key)

    # Extract log-density and transforms from the NumPyro model
    init_params, potential_fn, postprocess_fn, _ = initialize_model(
        init_key, model, model_kwargs=data,
    )

    logdensity_fn = lambda params: -potential_fn(params)
    initial_position = init_params.z

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        target_acceptance_rate=0.8,
    )

    # --- warmup (sequential — each chain gets its own adapted params) ----------

    chain_keys = jax.random.split(run_key, num_chains)
    warmup_states = []
    warmup_params = []
    for i in range(num_chains):
        warmup_key, _ = jax.random.split(chain_keys[i])
        print(f"  Chain {i + 1}/{num_chains} warmup ...")
        (state, parameters), _ = warmup.run(
            warmup_key, initial_position, num_steps=num_warmup
        )
        warmup_states.append(state)
        warmup_params.append(parameters)

    # --- sampling (vmapped across chains) --------------------------------------

    # All chains share the same adapted step_size/inverse_mass_matrix from chain 0
    # (they should be similar after warmup on the same model)
    shared_params = warmup_params[0]
    sampler = blackjax.nuts(logdensity_fn, **shared_params)

    init_states = jax.tree.map(lambda *xs: jnp.stack(xs), *warmup_states)
    sample_keys = jax.random.split(chain_keys[0], num_chains)

    @jax.jit
    def _sample_all_chains(init_states, sample_keys):
        def _one_chain(state, key):
            def step_fn(carry, rng_key):
                state, info = sampler.step(rng_key, carry)
                return state, (state, info)

            keys = jax.random.split(key, num_samples)
            _, (states, infos) = jax.lax.scan(step_fn, state, keys)
            return states, infos

        return jax.vmap(_one_chain)(init_states, sample_keys)

    print(f"  Sampling {num_samples} x {num_chains} chains (vmapped) ...")
    t0 = time.time()
    all_states, all_infos = _sample_all_chains(init_states, sample_keys)
    all_states.position  # force materialization
    elapsed = time.time() - t0
    print(f"  Sampling took {elapsed:.1f}s")

    # --- constrain back to model space  (num_chains, num_samples, ...) ---------

    def _postprocess_chain(positions):
        return jax.vmap(postprocess_fn)(positions)

    constrained_per_chain = jax.vmap(_postprocess_chain)(all_states.position)
    samples = constrained_per_chain  # already (num_chains, num_samples, ...)

    # Per-chain diagnostics
    for i in range(num_chains):
        n_div = int(jnp.sum(all_infos.is_divergent[i]))
        accept = float(jnp.mean(all_infos.acceptance_rate[i]))
        print(f"  Chain {i + 1}: {n_div} divergences, acceptance rate {accept:.3f}")

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────


def _rhat(chains):
    """Split-Rhat for array of shape (num_chains, num_samples)."""
    num_chains, num_samples = chains.shape
    # Split each chain in half
    half = num_samples // 2
    split = jnp.concatenate([chains[:, :half], chains[:, half:2*half]], axis=0)
    m = split.shape[0]
    n = split.shape[1]
    chain_means = jnp.mean(split, axis=1)
    grand_mean = jnp.mean(chain_means)
    B = n / (m - 1.0) * jnp.sum((chain_means - grand_mean) ** 2)
    chain_vars = jnp.var(split, axis=1, ddof=1)
    W = jnp.mean(chain_vars)
    var_hat = (n - 1.0) / n * W + B / n
    return jnp.sqrt(var_hat / W)


def _ess(chains):
    """Bulk ESS estimate for array of shape (num_chains, num_samples)."""
    num_chains, num_samples = chains.shape
    # FFT-based autocorrelation per chain, then average
    def _chain_ess(x):
        x = x - jnp.mean(x)
        n = x.shape[0]
        fft_len = 2 * n
        fft_x = jnp.fft.rfft(x, n=fft_len)
        acf = jnp.fft.irfft(fft_x * jnp.conj(fft_x), n=fft_len)[:n]
        acf = acf / acf[0]
        # Sum consecutive pairs; stop when sum is negative
        pairs = acf[1:n-1].reshape(-1, 2).sum(axis=1)
        # Find first negative pair
        neg_mask = pairs < 0
        first_neg = jnp.argmax(neg_mask)
        # If no negative pair found, use all
        first_neg = jnp.where(jnp.any(neg_mask), first_neg, pairs.shape[0])
        tau = -1.0 + 2.0 * jnp.sum(jnp.where(jnp.arange(pairs.shape[0]) < first_neg, pairs, 0.0))
        return n / jnp.maximum(tau, 1.0)

    return jnp.sum(jax.vmap(_chain_ess)(chains))


def print_summary(samples, true_params=TRUE_PARAMS):
    scalar_params = [
        "log_shape",
        "log_scale",
        "mu_k",
        "omega_k",
        "gamma_k",
        "gamma_hill",
        "EC50",
        "log_phi",
        "P0",
        "sigma_country_k",
        "sigma_country_s",
    ]
    vector_params = ["beta_s", "beta_k", "mu_country_k", "mu_country_s"]

    has_chains = next(iter(samples.values())).ndim >= 2 and next(iter(samples.values())).shape[0] > 1

    header = f"{'Parameter':<22s} {'True':>8s} {'Mean':>8s} {'Std':>8s} {'2.5%':>8s} {'97.5%':>8s}"
    if has_chains:
        header += f" {'Rhat':>6s} {'ESS':>7s}"
    print(f"\n{header}")
    print("─" * len(header))

    def _row(name, chains_2d, true_val):
        """chains_2d: (num_chains, num_samples) or (1, num_samples)."""
        s = chains_2d.flatten()
        true_str = f"{true_val:8.3f}" if true_val is not None else "     N/A"
        lo, hi = jnp.percentile(s, 2.5), jnp.percentile(s, 97.5)
        line = f"{name:<22s} {true_str} {s.mean():8.3f} {s.std():8.3f} {lo:8.3f} {hi:8.3f}"
        if has_chains:
            r = float(_rhat(chains_2d))
            e = float(_ess(chains_2d))
            line += f" {r:6.3f} {e:7.0f}"
        print(line)

    for name in scalar_params:
        if name in samples:
            # shape: (num_chains, num_samples, ...)
            _row(name, samples[name].reshape(samples[name].shape[0], -1), true_params.get(name))

    for name in vector_params:
        if name in samples:
            s = samples[name]
            true_vec = true_params.get(name)
            for j in range(s.shape[-1]):
                sj = s[..., j]  # (num_chains, num_samples)
                tv = float(true_vec[j]) if true_vec is not None else None
                _row(f"{name}[{j}]", sj, tv)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampler",
        choices=["numpyro", "blackjax"],
        default="numpyro",
        help="Which NUTS backend to use (default: numpyro)",
    )
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    numpyro.set_host_device_count(args.chains)

    data = generate_data(seed=42)

    print(f"\nSampling with {args.sampler} NUTS "
          f"({args.warmup} warmup, {args.samples} samples, {args.chains} chains)\n")

    if args.sampler == "numpyro":
        samples = run_numpyro_nuts(
            data,
            num_warmup=args.warmup,
            num_samples=args.samples,
            num_chains=args.chains,
            seed=args.seed,
        )
    else:
        samples = run_blackjax_nuts(
            data,
            num_warmup=args.warmup,
            num_samples=args.samples,
            num_chains=args.chains,
            seed=args.seed,
        )

    print_summary(samples)
