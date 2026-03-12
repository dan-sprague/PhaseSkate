
# Samplers {#Samplers}

PhaseSkate provides three MCMC samplers, all using Enzyme for gradient computation.

## NUTS {#NUTS}

The No-U-Turn Sampler with dual-averaging step size adaptation.

```julia
ch = sample(model, num_samples;
    warmup = 1000,
    chains = 4,
    ϵ = 0.1,           # initial step size (adapted)
    max_depth = 10,     # max tree depth
    ad = :auto,         # :auto, :forward, or :reverse
    seed = nothing,     # RNG seed
    δ = 0.8,            # target acceptance probability
)
```


**When to use:** General purpose. Well-understood diagnostics (divergences, tree depth). Good default choice.

## MCLMC {#MCLMC}

Microcanonical Langevin Monte Carlo ([Robnik &amp; Seljak, 2023](https://arxiv.org/abs/2212.08549)). Uses equal-speed Hamiltonian dynamics on the energy level set with partial momentum refreshment.

```julia
ch = sample_mclmc(model, num_samples;
    warmup = 1000,
    chains = 4,
    ad = :auto,
    seed = nothing,
    diagonal_preconditioning = true,
    Lfactor = 0.4,      # trajectory length factor
)
```


**When to use:** High-dimensional models where NUTS is slow. Note: MCLMC has discretization bias (no MH correction).

## Adjusted MCLMC (MAMS) {#Adjusted-MCLMC-MAMS}

MCLMC with Metropolis-Hastings correction, removing discretization bias while retaining the efficiency of microcanonical dynamics.

```julia
ch = sample_adjusted_mclmc(model, num_samples;
    warmup = 1000,
    chains = 4,
    ad = :auto,
    seed = nothing,
    diagonal_preconditioning = true,
    target_acceptance = 0.65,  # MH acceptance target
    tuning_factor = 1.3,       # trajectory length scale
    L_proposal_factor = Inf,   # momentum refresh rate (Inf = no refresh)
    num_windows = 2,           # adaptation rounds
)
```


**When to use:** High-dimensional models where you need exact (unbiased) inference. Best of both worlds: MCLMC speed with NUTS correctness.

## Autodiff Mode Selection {#Autodiff-Mode-Selection}

All samplers accept an `ad` keyword:

|      Value |              Mode |                               Used when |
| ----------:| -----------------:| ---------------------------------------:|
|    `:auto` |         Automatic | Forward for dim ≤ 20, reverse otherwise |
| `:forward` |   Batched forward |            Small models (low dimension) |
| `:reverse` | Reverse (adjoint) |           Large models (high dimension) |


The threshold is `dim = 20` — below this, forward mode is typically faster because it avoids the overhead of reverse-mode tape construction.

## Diagnostics {#Diagnostics}

All samplers return a `Chains` object with built-in diagnostics:

```julia
ch = sample(m, 2000; warmup=1000, chains=4)

# Summary table (printed automatically)
display(ch)

# Individual diagnostics
min_ess(ch)          # minimum bulk ESS across all parameters
mean(ch, :mu)        # posterior mean
ci(ch, :mu)          # 95% credible interval
samples(ch, :mu)     # raw samples array
thin(ch, 500)        # thin to 500 draws per chain
```


The summary table includes split-R̂, bulk ESS, and tail ESS for each parameter. Parameters with R̂ &gt; 1.01 or ESS &lt; 400 are highlighted in yellow.
