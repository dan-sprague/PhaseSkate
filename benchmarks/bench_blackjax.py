"""
benchmarks/bench_blackjax.py
Benchmark BlackJAX NUTS on the JointALM model using shared data from generate_data.jl.

Usage:
    JAX_PLATFORMS=cpu python benchmarks/bench_blackjax.py
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import initialize_model

# ── Configuration ────────────────────────────────────────────────────────────

NUM_CHAINS = 4
NUM_WARMUP = 1000
NUM_SAMPLES = 2000
SEED = 42

numpyro.set_host_device_count(NUM_CHAINS)

BENCH_DIR = Path(__file__).parent
DATA_PATH = BENCH_DIR / "data" / "joint_alm_data.json"
RESULT_PATH = BENCH_DIR / "results" / "blackjax_results.json"

# ── Weibull helpers ──────────────────────────────────────────────────────────


def weibull_logscale_lpdf(t, shape, log_scale):
    log_t = jnp.log(t)
    return (
        jnp.log(shape)
        - shape * log_scale
        + (shape - 1.0) * log_t
        - jnp.exp(shape * (log_t - log_scale))
    )


def weibull_logscale_lccdf(t, shape, log_scale):
    return -jnp.exp(shape * (jnp.log(t) - log_scale))


# ── NumPyro model (used for log-density extraction) ─────────────────────────


def numpyro_model(
    n1, n2, p, n_countries, MRC_MAX,
    tier1_times, tier1_X, tier1_country_ids, tier1_obs_idx, tier1_cens_idx,
    tier2_times, tier2_X, tier2_country_ids, tier2_obs_idx, tier2_cens_idx,
    total_mrc_obs, mrc_scores_flat, mrc_times_flat, mrc_patient_ids,
):
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

    shape = jnp.exp(log_shape)
    inv_shape = 1.0 / shape
    phi = jnp.exp(log_phi)
    log_P0_ratio = jnp.log1p(-P0) - jnp.log(P0)
    log_EC50g = gamma_hill * jnp.log(EC50)

    log_k_2 = (
        mu_k + tier2_X @ beta_k + mu_country_k[tier2_country_ids] + omega_k * z_k
    )
    log_eff_scale_2 = log_scale - (
        tier2_X @ beta_s + mu_country_s[tier2_country_ids] + gamma_k * log_k_2
    ) * inv_shape

    log_k_1 = mu_k + tier1_X @ beta_k + mu_country_k[tier1_country_ids]
    log_eff_scale_1 = log_scale - (
        tier1_X @ beta_s + mu_country_s[tier1_country_ids] + gamma_k * log_k_1
    ) * inv_shape

    numpyro.factor(
        "tier2_obs_ll",
        jnp.sum(weibull_logscale_lpdf(
            tier2_times[tier2_obs_idx], shape, log_eff_scale_2[tier2_obs_idx]
        )),
    )
    numpyro.factor(
        "tier2_cens_ll",
        jnp.sum(weibull_logscale_lccdf(
            tier2_times[tier2_cens_idx], shape, log_eff_scale_2[tier2_cens_idx]
        )),
    )
    numpyro.factor(
        "tier1_obs_ll",
        jnp.sum(weibull_logscale_lpdf(
            tier1_times[tier1_obs_idx], shape, log_eff_scale_1[tier1_obs_idx]
        )),
    )
    numpyro.factor(
        "tier1_cens_ll",
        jnp.sum(weibull_logscale_lccdf(
            tier1_times[tier1_cens_idx], shape, log_eff_scale_1[tier1_cens_idx]
        )),
    )

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


# ── ESS / Rhat helpers ──────────────────────────────────────────────────────


def split_rhat(chains):
    """Split-Rhat for array (num_chains, num_samples)."""
    num_chains, num_samples = chains.shape
    half = num_samples // 2
    split = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
    m, n = split.shape
    chain_means = split.mean(axis=1)
    grand_mean = chain_means.mean()
    B = n / (m - 1.0) * np.sum((chain_means - grand_mean) ** 2)
    W = np.mean(np.var(split, axis=1, ddof=1))
    var_hat = (n - 1.0) / n * W + B / n
    return np.sqrt(var_hat / W) if W > 0 else float("nan")


def bulk_ess(chains):
    """Bulk ESS for array (num_chains, num_samples)."""
    total = 0.0
    for chain in chains:
        x = chain - chain.mean()
        n = len(x)
        fft_x = np.fft.rfft(x, n=2 * n)
        acf = np.fft.irfft(fft_x * np.conj(fft_x), n=2 * n)[:n]
        acf = acf / acf[0]
        pairs = acf[1:n - 1].reshape(-1, 2).sum(axis=1)
        neg = np.where(pairs < 0)[0]
        first_neg = neg[0] if len(neg) > 0 else len(pairs)
        tau = -1.0 + 2.0 * np.sum(pairs[:first_neg])
        total += n / max(tau, 1.0)
    return total


# ── Load data ────────────────────────────────────────────────────────────────


def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)

    # Convert index arrays from 1-based (Julia/Stan) to 0-based (Python)
    return dict(
        n1=raw["n1"],
        n2=raw["n2"],
        p=raw["p"],
        n_countries=raw["n_countries"],
        MRC_MAX=raw["MRC_MAX"],
        tier1_times=jnp.array(raw["tier1_times"]),
        tier1_X=jnp.array(raw["tier1_X"]),
        tier1_country_ids=jnp.array([x - 1 for x in raw["tier1_country_ids"]]),
        tier1_obs_idx=jnp.array([x - 1 for x in raw["tier1_obs_idx"]]),
        tier1_cens_idx=jnp.array([x - 1 for x in raw["tier1_cens_idx"]]),
        tier2_times=jnp.array(raw["tier2_times"]),
        tier2_X=jnp.array(raw["tier2_X"]),
        tier2_country_ids=jnp.array([x - 1 for x in raw["tier2_country_ids"]]),
        tier2_obs_idx=jnp.array([x - 1 for x in raw["tier2_obs_idx"]]),
        tier2_cens_idx=jnp.array([x - 1 for x in raw["tier2_cens_idx"]]),
        total_mrc_obs=raw["total_mrc_obs"],
        mrc_scores_flat=jnp.array(raw["mrc_scores_flat"]),
        mrc_times_flat=jnp.array(raw["mrc_times_flat"]),
        mrc_patient_ids=jnp.array([x - 1 for x in raw["mrc_patient_ids"]]),
    )


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import blackjax
    from blackjax.util import run_inference_algorithm

    print(f"JAX platforms: {jax.devices()}")
    print(f"Loading data from: {DATA_PATH}")
    data = load_data()

    # Initialize from NumPyro model
    rng_key = jax.random.PRNGKey(SEED)
    init_key, run_key = jax.random.split(rng_key)

    init_params, potential_fn, postprocess_fn, _ = initialize_model(
        init_key, numpyro_model, model_kwargs=data,
    )

    logdensity_fn = lambda params: -potential_fn(params)
    initial_position = init_params.z

    # ── Warmup (sequential per chain) ────────────────────────────────────────

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        target_acceptance_rate=0.8,
    )

    chain_keys = jax.random.split(run_key, NUM_CHAINS)
    warmup_states = []
    warmup_params = []

    print(f"\nWarmup: {NUM_WARMUP} steps x {NUM_CHAINS} chains")
    t_warmup_start = time.time()
    for i in range(NUM_CHAINS):
        warmup_key, _ = jax.random.split(chain_keys[i])
        print(f"  Chain {i + 1}/{NUM_CHAINS} warmup ...")
        (state, parameters), _ = warmup.run(
            warmup_key, initial_position, num_steps=NUM_WARMUP
        )
        warmup_states.append(state)
        warmup_params.append(parameters)
    t_warmup = time.time() - t_warmup_start

    # ── Sampling (vmapped) ───────────────────────────────────────────────────

    shared_params = warmup_params[0]
    sampler = blackjax.nuts(logdensity_fn, **shared_params)

    init_states = jax.tree.map(lambda *xs: jnp.stack(xs), *warmup_states)
    sample_keys = jax.random.split(chain_keys[0], NUM_CHAINS)

    @jax.jit
    def _sample_all_chains(init_states, sample_keys):
        def _one_chain(state, key):
            def step_fn(carry, rng_key):
                state, info = sampler.step(rng_key, carry)
                return state, (state, info)

            keys = jax.random.split(key, NUM_SAMPLES)
            _, (states, infos) = jax.lax.scan(step_fn, state, keys)
            return states, infos

        return jax.vmap(_one_chain)(init_states, sample_keys)

    print(f"\nSampling: {NUM_SAMPLES} draws x {NUM_CHAINS} chains (vmapped)")
    t_sample_start = time.time()
    all_states, all_infos = _sample_all_chains(init_states, sample_keys)
    all_states.position  # force materialization
    t_sample = time.time() - t_sample_start
    t_total = t_warmup + t_sample

    # ── Diagnostics ──────────────────────────────────────────────────────────

    constrained = jax.vmap(
        lambda pos: jax.vmap(postprocess_fn)(pos)
    )(all_states.position)

    n_divergent = int(jnp.sum(all_infos.is_divergent))

    # Compute ESS and Rhat on scalar parameters
    scalar_params = ["log_shape", "log_scale", "mu_k", "omega_k", "gamma_k",
                     "gamma_hill", "EC50", "log_phi", "P0",
                     "sigma_country_k", "sigma_country_s"]
    vector_params = ["beta_s", "beta_k", "mu_country_k", "mu_country_s", "z_k"]

    min_ess = float("inf")
    max_rhat_val = 0.0

    for name in scalar_params:
        if name in constrained:
            chains_2d = np.array(constrained[name]).reshape(NUM_CHAINS, -1)
            min_ess = min(min_ess, bulk_ess(chains_2d))
            max_rhat_val = max(max_rhat_val, split_rhat(chains_2d))

    for name in vector_params:
        if name in constrained:
            s = np.array(constrained[name])
            for j in range(s.shape[-1]):
                chains_2d = s[..., j].reshape(NUM_CHAINS, -1)
                min_ess = min(min_ess, bulk_ess(chains_2d))
                max_rhat_val = max(max_rhat_val, split_rhat(chains_2d))

    for i in range(NUM_CHAINS):
        n_div_i = int(jnp.sum(all_infos.is_divergent[i]))
        acc_i = float(jnp.mean(all_infos.acceptance_rate[i]))
        print(f"  Chain {i + 1}: {n_div_i} divergences, acceptance rate {acc_i:.3f}")

    print(f"\n{'── Results ':─<50}")
    print(f"  Total wall time:  {t_total:.1f} s")
    print(f"  Warmup time:      {t_warmup:.1f} s")
    print(f"  Sampling time:    {t_sample:.1f} s")
    print(f"  Min ESS (bulk):   {min_ess:.0f}")
    print(f"  Max Rhat:         {max_rhat_val:.4f}")
    print(f"  ESS/s:            {min_ess / t_sample:.1f}")
    print(f"  Divergences:      {n_divergent}")

    # ── Save results ─────────────────────────────────────────────────────────

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "backend": "BlackJAX",
        "num_chains": NUM_CHAINS,
        "num_warmup": NUM_WARMUP,
        "num_samples": NUM_SAMPLES,
        "total_time_s": round(t_total, 2),
        "warmup_time_s": round(t_warmup, 2),
        "sampling_time_s": round(t_sample, 2),
        "min_ess_bulk": round(min_ess, 1),
        "max_rhat": round(max_rhat_val, 4),
        "ess_per_sec": round(min_ess / t_sample, 1),
        "divergences": n_divergent,
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULT_PATH}")
