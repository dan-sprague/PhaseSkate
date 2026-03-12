
# API Reference {#API-Reference}

## Model Definition {#Model-Definition}
<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.@skate' href='#PhaseSkate.@skate'><span class="jlbinding">PhaseSkate.@skate</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@skate ModelName begin
    @constants begin ... end
    @params begin ... end
    @logjoint begin ... end
end
```


Define a Bayesian model. Generates:
- `ModelNameData` struct for holding constants/data
  
- `make(data::ModelNameData) → ModelLogDensity` to build the compiled model
  

**Blocks:**
- `@constants` — Declare data fields with types (e.g. `N::Int`, `X::Matrix{Float64}`)
  
- `@params` — Declare parameters to sample. Scalars use `name::Float64`, constrained params use `name = param(Float64; lower=0.0)`, vectors/matrices use `name = param(Vector{Float64}, K)`. Supports `simplex=true`, `ordered=true`.
  
- `@logjoint` — The log-joint density. Accumulate via `target += lpdf(...)`. Use `@for begin ... end` for zero-allocation broadcast-to-loop unrolling.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/lang.jl#L1145-L1163" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Core Types {#Core-Types}
<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.ModelLogDensity' href='#PhaseSkate.ModelLogDensity'><span class="jlbinding">PhaseSkate.ModelLogDensity</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ModelLogDensity
```


Compiled model holding the log-density function, parameter dimension, and a constraining function. Created by `make(data)`.

Fields:
- `dim::Int` — Number of unconstrained parameters.
  
- `ℓ` — Log-density closure `q::Vector{Float64} → Float64`.
  
- `constrain` — Maps unconstrained vector to named parameter tuple.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/safe_grads.jl#L1-L11" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.log_prob' href='#PhaseSkate.log_prob'><span class="jlbinding">PhaseSkate.log_prob</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
log_prob(model, q) → Float64
```


Evaluate the log-density of `model` at unconstrained parameter vector `q`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/safe_grads.jl#L18-L22" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Samplers {#Samplers}
<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.sample' href='#PhaseSkate.sample'><span class="jlbinding">PhaseSkate.sample</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
sample(model, num_samples; warmup=1000, chains=4, ϵ=0.1, max_depth=10, ad=:auto, seed=nothing, δ=0.8, metric=:auto) → Chains
```


Run NUTS (No-U-Turn Sampler) on a compiled model. Returns a `Chains` object.

**Arguments**
- `model`: A `ModelLogDensity` from `make(data)`.
  
- `num_samples`: Number of post-warmup draws per chain.
  
- `warmup`: Number of warmup/adaptation steps per chain.
  
- `chains`: Number of parallel chains.
  
- `ϵ`: Initial step size (adapted during warmup).
  
- `max_depth`: Maximum tree depth for NUTS.
  
- `ad`: Autodiff mode — `:auto`, `:forward`, or `:reverse`.
  
- `seed`: RNG seed for reproducibility.
  
- `δ`: Target acceptance probability for step size adaptation.
  
- `metric`: Mass matrix type — `:auto` (dense if dim ≤ 500), `:dense`, or `:diagonal`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/hmc.jl#L703-L719" target="_blank" rel="noreferrer">source</a></Badge>

</details>


::: warning Missing docstring.

Missing docstring for `sample_mclmc`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `sample_adjusted_mclmc`. Check Documenter&#39;s build log for details.

:::

## Chains &amp; Diagnostics {#Chains-and-Diagnostics}
<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.Chains' href='#PhaseSkate.Chains'><span class="jlbinding">PhaseSkate.Chains</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Chains(chain_results)
```


Build a `Chains` object from the output of multi-chain sampling. `chain_results` is a `Vector` of per-chain sample vectors, where each sample is a `NamedTuple` of constrained parameter values. Single-chain input (a plain `Vector{<:NamedTuple}`) is wrapped automatically. `raw_chains` is a `Vector{Matrix{Float64}}` where each matrix is `(dim × nsamples)`, and `constrain` transforms an unconstrained vector into a `NamedTuple`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/chains.jl#L15-L24" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.samples' href='#PhaseSkate.samples'><span class="jlbinding">PhaseSkate.samples</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
samples(c::Chains, name::Symbol)
```


Return raw samples for parameter `name`.   scalar  → (nsamples, nchains)   (K,)    → (nsamples, K, nchains)   (K,D)   → (nsamples, K, D, nchains)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/chains.jl#L70-L77" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Statistics.mean-Tuple{Chains, Symbol}' href='#Statistics.mean-Tuple{Chains, Symbol}'><span class="jlbinding">Statistics.mean</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
mean(c::Chains, name::Symbol)
```


Posterior mean of parameter `name`, averaged over all samples and chains. Returns an array matching the parameter&#39;s original shape (or a scalar).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/chains.jl#L89-L94" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.ci' href='#PhaseSkate.ci'><span class="jlbinding">PhaseSkate.ci</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
ci(c::Chains, name::Symbol; level=0.95)
```


Element-wise credible interval for parameter `name`, pooling all chains. Returns `(lower, upper)`, each matching the parameter&#39;s original shape.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/chains.jl#L107-L112" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.thin' href='#PhaseSkate.thin'><span class="jlbinding">PhaseSkate.thin</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
thin(c::Chains, M::Int) → Chains
```


Thin a Chains object to M evenly-spaced draws per chain.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/chains.jl#L178-L182" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.min_ess' href='#PhaseSkate.min_ess'><span class="jlbinding">PhaseSkate.min_ess</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
min_ess(c::Chains) → Float64
```


Minimum bulk ESS across all scalar parameter elements and chains.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/chains.jl#L138-L142" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Simulation-Based Calibration {#Simulation-Based-Calibration}
<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.sbc' href='#PhaseSkate.sbc'><span class="jlbinding">PhaseSkate.sbc</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
sbc(simulate; N=100, M=200, num_samples=1000, ...)
```


Run Simulation-Based Calibration.

`simulate()` takes no arguments and returns `(theta_true::NamedTuple, model::ModelLogDensity)`. The function should draw parameters from the prior, simulate data, build the model, and return both.

Posterior draws are thinned to M roughly independent samples using ESS-based thinning. If the effective sample size is less than M, the sampler doubles `num_samples` (up to 4 times) until min ESS ≥ M.

Returns an `SBCResult` with rank statistics and chi-squared uniformity p-values.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/sbc.jl#L114-L127" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.calibrated' href='#PhaseSkate.calibrated'><span class="jlbinding">PhaseSkate.calibrated</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
calibrated(result; alpha=0.01)
```


Return `true` if all parameters pass the chi-squared uniformity test at level `alpha`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/sbc.jl#L194-L198" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Constraint Transforms {#Constraint-Transforms}

::: warning Missing docstring.

Missing docstring for `IdentityConstraint`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `LowerBounded`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `UpperBounded`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `Bounded`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `SimplexConstraint`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `OrderedConstraint`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `simplex_transform`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `ordered_transform`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `corr_cholesky_transform`. Check Documenter&#39;s build log for details.

:::

## Log-Density Functions {#Log-Density-Functions}

### Continuous Univariate {#Continuous-Univariate}
<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.normal_lpdf' href='#PhaseSkate.normal_lpdf'><span class="jlbinding">PhaseSkate.normal_lpdf</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



Log-density of Normal(μ, σ) evaluated at x. Pure arithmetic — Enzyme-safe.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/lpdfs.jl#L145" target="_blank" rel="noreferrer">source</a></Badge>

</details>


::: warning Missing docstring.

Missing docstring for `cauchy_lpdf`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `exponential_lpdf`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `gamma_lpdf`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `beta_lpdf`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `lognormal_lpdf`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `student_t_lpdf`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `uniform_lpdf`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `laplace_lpdf`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `logistic_lpdf`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `weibull_lpdf`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `weibull_logsigma_lpdf`. Check Documenter&#39;s build log for details.

:::
<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.weibull_logsigma_lccdf' href='#PhaseSkate.weibull_logsigma_lccdf'><span class="jlbinding">PhaseSkate.weibull_logsigma_lccdf</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
weibull_logsigma_lccdf(x, α, log_σ)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/lpdfs.jl#L245-L247" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Discrete {#Discrete}

::: warning Missing docstring.

Missing docstring for `poisson_lpdf`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `binomial_lpdf`. Check Documenter&#39;s build log for details.

:::
<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.bernoulli_logit_lpdf' href='#PhaseSkate.bernoulli_logit_lpdf'><span class="jlbinding">PhaseSkate.bernoulli_logit_lpdf</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
bernoulli_logit_lpdf(y, α)
```


Stan-style Bernoulli log-density using the logit-link linear predictor α. α is typically (intercept + X * beta).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/lpdfs.jl#L217-L222" target="_blank" rel="noreferrer">source</a></Badge>

</details>


::: warning Missing docstring.

Missing docstring for `binomial_logit_lpdf`. Check Documenter&#39;s build log for details.

:::
<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.neg_binomial_2_lpdf' href='#PhaseSkate.neg_binomial_2_lpdf'><span class="jlbinding">PhaseSkate.neg_binomial_2_lpdf</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
neg_binomial_2_lpdf(y, μ, ϕ)
```


Stan-style Negative Binomial log-density. μ: Mean ϕ: Dispersion (smaller ϕ = more variance/overdispersion)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/lpdfs.jl#L199-L205" target="_blank" rel="noreferrer">source</a></Badge>

</details>


::: warning Missing docstring.

Missing docstring for `beta_binomial_lpdf`. Check Documenter&#39;s build log for details.

:::

::: warning Missing docstring.

Missing docstring for `categorical_logit_lpdf`. Check Documenter&#39;s build log for details.

:::
<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.dirichlet_lpdf' href='#PhaseSkate.dirichlet_lpdf'><span class="jlbinding">PhaseSkate.dirichlet_lpdf</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
dirichlet_lpdf(x, K::Float64)
```


Symmetric Dirichlet with concentration α. Equivalent to `dirichlet_lpdf(x, fill(α, length(x)))` but zero-allocation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/lpdfs.jl#L368-L373" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Multivariate {#Multivariate}

::: warning Missing docstring.

Missing docstring for `multi_normal_diag_lpdf`. Check Documenter&#39;s build log for details.

:::
<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.multi_normal_cholesky_lpdf' href='#PhaseSkate.multi_normal_cholesky_lpdf'><span class="jlbinding">PhaseSkate.multi_normal_cholesky_lpdf</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
multi_normal_cholesky_lpdf(x, μ, L)
```


Stan-style log-density for MVN. L is the Lower-Triangular Cholesky factor of the covariance matrix.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/lpdfs.jl#L21-L26" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.multi_normal_cholesky_scaled_lpdf' href='#PhaseSkate.multi_normal_cholesky_scaled_lpdf'><span class="jlbinding">PhaseSkate.multi_normal_cholesky_scaled_lpdf</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
multi_normal_cholesky_scaled_lpdf(x, μ, log_sigma_row, L_corr)
```


Fused MVN Cholesky log-density with log-scale parameterization. Equivalent to `multi_normal_cholesky_lpdf(x, μ, diag_pre_multiply(exp.(log_sigma), L_corr))` but computes `exp()` per-element inline — no broadcast allocation, no `diag_pre_multiply` temp.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/lpdfs.jl#L117-L123" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.lkj_corr_cholesky_lpdf' href='#PhaseSkate.lkj_corr_cholesky_lpdf'><span class="jlbinding">PhaseSkate.lkj_corr_cholesky_lpdf</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
lkj_corr_cholesky_lpdf(L, η)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/lpdfs.jl#L307-L309" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.diag_pre_multiply' href='#PhaseSkate.diag_pre_multiply'><span class="jlbinding">PhaseSkate.diag_pre_multiply</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



Construct the Cholesky factor of a covariance matrix from scales and correlation Cholesky. Equivalent to `Diagonal(sigma) * L_corr`, i.e. `sigma .* L_corr`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/lpdfs.jl#L113-L115" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Utilities {#Utilities}
<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.log_sum_exp' href='#PhaseSkate.log_sum_exp'><span class="jlbinding">PhaseSkate.log_sum_exp</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
log_sum_exp(x)
```


The &#39;Bare Metal&#39; stability trick for Logit/Softmax math.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/utilities.jl#L2-L6" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='PhaseSkate.log_mix' href='#PhaseSkate.log_mix'><span class="jlbinding">PhaseSkate.log_mix</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
log_mix(weights, f)
```


Zero-allocation log-mixture-density (branchless). `f(j)` returns the log-density of component `j`.

Usage with `do` syntax:     log_mix(theta) do j         normal_lpdf(x[i], mus[j], sigma)     end


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/utilities.jl#L17-L27" target="_blank" rel="noreferrer">source</a></Badge>



```julia
log_mix(a, b)
```


Log-sum-exp of elementwise `a .+ b`, zero-allocation. `a` and `b` are vectors of log-values (e.g. log-weights and log-likelihoods).

```
log_mix(log_theta, log_phi) == log(sum(exp.(log_theta .+ log_phi)))
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/utilities.jl#L39-L46" target="_blank" rel="noreferrer">source</a></Badge>



```julia
log_mix(a, b, offset)
```


`log_sum_exp(a .+ offset .+ b)`, zero-allocation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/dan-sprague/PhaseSkate/blob/727fa4fcbb781e297d69f1666778356b7168d164/src/utilities.jl#L58-L62" target="_blank" rel="noreferrer">source</a></Badge>

</details>

