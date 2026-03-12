## ── Parameter metadata ──────────────────────────────────────────────────────
struct ParamInfo
    name::Symbol
    shape::Tuple            # () for scalar, (K,) for vector, (K,D) for matrix, …
    cols::UnitRange{Int}    # column range in the flat data array
end

## ── Chains ──────────────────────────────────────────────────────────────────
struct Chains
    data::Array{Float64, 3}               # (nsamples, total_flat_params, nchains)
    params::Vector{ParamInfo}
    name_map::Dict{Symbol, ParamInfo}
end

"""
    Chains(chain_results)

Build a `Chains` object from the output of multi-chain sampling.
`chain_results` is a `Vector` of per-chain sample vectors, where each sample
is a `NamedTuple` of constrained parameter values.
Single-chain input (a plain `Vector{<:NamedTuple}`) is wrapped automatically.
`raw_chains` is a `Vector{Matrix{Float64}}` where each matrix is `(dim × nsamples)`,
and `constrain` transforms an unconstrained vector into a `NamedTuple`.
"""
function Chains(raw_chains::Vector{<:Matrix{Float64}}, constrain::Function)
    nchains  = length(raw_chains)
    nsamples = size(raw_chains[1], 2)

    # ── discover layout from first constrained sample ────────────────────
    first_nt = constrain(@view raw_chains[1][:, 1])
    params = ParamInfo[]
    col = 1
    for name in keys(first_nt)
        val = first_nt[name]
        if val isa Float64
            shape = ()
            ncols = 1
        else
            shape = size(val)
            ncols = length(val)
        end
        push!(params, ParamInfo(name, shape, col:col+ncols-1))
        col += ncols
    end
    total_cols = col - 1

    # ── constrain + copy directly into the 3-D array ─────────────────────
    data = Array{Float64, 3}(undef, nsamples, total_cols, nchains)
    for c in 1:nchains
        raw = raw_chains[c]
        for i in 1:nsamples
            nt = constrain(@view raw[:, i])
            @inbounds for p in params
                val = nt[p.name]
                if val isa Float64
                    data[i, p.cols[1], c] = val
                else
                    data[i, p.cols, c] .= vec(val)
                end
            end
        end
    end

    name_map = Dict(p.name => p for p in params)
    return Chains(data, params, name_map)
end

## ── Accessors ───────────────────────────────────────────────────────────────

"""
    samples(c::Chains, name::Symbol)

Return raw samples for parameter `name`.
  scalar  → (nsamples, nchains)
  (K,)    → (nsamples, K, nchains)
  (K,D)   → (nsamples, K, D, nchains)
"""
function samples(c::Chains, name::Symbol)
    info = c.name_map[name]
    raw  = @view c.data[:, info.cols, :]
    ns, nc = size(c.data, 1), size(c.data, 3)
    if info.shape == ()
        return reshape(raw, ns, nc)
    else
        return reshape(raw, ns, info.shape..., nc)
    end
end

"""
    mean(c::Chains, name::Symbol)

Posterior mean of parameter `name`, averaged over all samples and chains.
Returns an array matching the parameter's original shape (or a scalar).
"""
function mean(c::Chains, name::Symbol)
    info = c.name_map[name]
    flat = vec(Statistics.mean(c.data[:, info.cols, :]; dims=(1, 3)))
    if info.shape == ()
        return flat[1]
    elseif length(info.shape) == 1
        return flat
    else
        return reshape(flat, info.shape)
    end
end

"""
    ci(c::Chains, name::Symbol; level=0.95)

Element-wise credible interval for parameter `name`, pooling all chains.
Returns `(lower, upper)`, each matching the parameter's original shape.
"""
function ci(c::Chains, name::Symbol; level::Float64=0.95)
    (0 < level < 1) || throw(ArgumentError("ci: level must be in (0, 1), got $level"))
    info = c.name_map[name]
    α = (1 - level) / 2
    ncols = length(info.cols)

    lower = Vector{Float64}(undef, ncols)
    upper = Vector{Float64}(undef, ncols)
    for (j, col) in enumerate(info.cols)
        pooled = vec(@view c.data[:, col, :])
        lower[j] = quantile(pooled, α)
        upper[j] = quantile(pooled, 1 - α)
    end

    if info.shape == ()
        return (lower[1], upper[1])
    elseif length(info.shape) == 1
        return (lower, upper)
    else
        return (reshape(lower, info.shape), reshape(upper, info.shape))
    end
end

## ── Diagnostics accessors ────────────────────────────────────────────────────

"""
    min_ess(c::Chains) → Float64

Minimum bulk ESS across all scalar parameter elements and chains.
"""
function min_ess(c::Chains)
    ess_min = Inf
    for p in c.params
        for col in p.cols
            x = @view c.data[:, col, :]
            _, essb, _ = _col_diagnostics(x)
            ess_min = min(ess_min, essb)
        end
    end
    return ess_min
end

"""
    diagnostics(c::Chains) → (names, rhat, ess_bulk, ess_tail)

Per-element diagnostic vectors: parameter labels, R̂, bulk ESS, tail ESS.
"""
function diagnostics(c::Chains)
    labels = String[]
    rhat_vec = Float64[]
    essb_vec = Float64[]
    esst_vec = Float64[]
    for p in c.params
        for (k, col) in enumerate(p.cols)
            x = @view c.data[:, col, :]
            rhat, essb, esst = _col_diagnostics(x)
            push!(labels, _elem_label(p.name, p.shape, k))
            push!(rhat_vec, rhat)
            push!(essb_vec, essb)
            push!(esst_vec, esst)
        end
    end
    return (names=labels, rhat=rhat_vec, ess_bulk=essb_vec, ess_tail=esst_vec)
end

"""
    thin(c::Chains, M::Int) → Chains

Thin a Chains object to M evenly-spaced draws per chain.
"""
function thin(c::Chains, M::Int)
    M > 0 || throw(ArgumentError("thin: M must be > 0, got $M"))
    ns = size(c.data, 1)
    M_per_chain = min(M, ns)
    idx = round.(Int, range(1, ns, length=M_per_chain))
    thinned = c.data[idx, :, :]
    return Chains(thinned, c.params, c.name_map)
end

## ── MCMC Diagnostics ────────────────────────────────────────────────────────

"""
    _split_rhat(x::AbstractMatrix{Float64}) → Float64

Split-R̂ (BDA3). Splits each of m chains in half → 2m chains,
computes between/within variance ratio. No autocovariance needed.
"""
function _split_rhat(x::AbstractMatrix{Float64})
    n, m = size(x)
    nhalf = n ÷ 2
    m2 = 2m

    # compute split-chain means and variances without allocating a split matrix
    means = Vector{Float64}(undef, m2)
    vars  = Vector{Float64}(undef, m2)

    @inbounds for j in 1:m
        # first half
        s = 0.0
        @simd for i in 1:nhalf
            s += x[i, j]
        end
        μ = s / nhalf
        v = 0.0
        @simd for i in 1:nhalf
            v += (x[i, j] - μ) * (x[i, j] - μ)
        end
        means[2j - 1] = μ
        vars[2j - 1]  = v / (nhalf - 1)

        # second half
        s = 0.0
        @simd for i in (nhalf + 1):(2nhalf)
            s += x[i, j]
        end
        μ = s / nhalf
        v = 0.0
        @simd for i in (nhalf + 1):(2nhalf)
            v += (x[i, j] - μ) * (x[i, j] - μ)
        end
        means[2j] = μ
        vars[2j]  = v / (nhalf - 1)
    end

    W = Statistics.mean(vars)
    B = nhalf * Statistics.var(means)
    var_hat = (nhalf - 1) / nhalf * W + B / nhalf
    return W > 0.0 ? sqrt(var_hat / W) : NaN
end

"""
    _ess(x::AbstractMatrix{Float64}) → Float64

Effective sample size via Geyer's initial positive sequence estimator
with monotone truncation. Matches stan/analyze/mcmc/ess.hpp exactly:
biased (1/N) autocovariance, paired rho starting at lag 0,
monotone sequence enforcement, and `1/log10(N*M)` safety floor.
"""
function _ess(x::AbstractMatrix{Float64})
    n, m = size(x)
    n < 4 && return Float64(n * m)

    # Per-chain biased autocovariance (1/n normalization, matching Stan/Geyer 1992)
    # and within-chain variance (Bessel-corrected)
    inv_n = 1.0 / n
    chain_means = Vector{Float64}(undef, m)
    chain_var = Vector{Float64}(undef, m)

    # Demean into contiguous buffer
    xc = Matrix{Float64}(undef, n, m)
    @inbounds for j in 1:m
        s = 0.0
        @simd for i in 1:n
            s += x[i, j]
        end
        μ = s * inv_n
        chain_means[j] = μ
        @simd for i in 1:n
            xc[i, j] = x[i, j] - μ
        end
        # biased variance * n/(n-1) = unbiased variance
        v = 0.0
        @simd for i in 1:n
            v += xc[i, j] * xc[i, j]
        end
        chain_var[j] = v * inv_n * (n / (n - 1.0))
    end

    W = 0.0
    @inbounds @simd for j in 1:m
        W += chain_var[j]
    end
    W /= m

    # var_plus = W*(n-1)/n + B  where B = var(chain_means)
    var_plus = if m > 1
        overall = 0.0
        @inbounds @simd for j in 1:m
            overall += chain_means[j]
        end
        overall /= m
        B = 0.0
        @inbounds @simd for j in 1:m
            δ = chain_means[j] - overall; B += δ * δ
        end
        B /= (m - 1)
        W * (n - 1.0) / n + B
    else
        W  # single chain: no between-chain term
    end
    var_plus <= 0.0 && return Float64(n * m)

    # Geyer's initial positive sequence with monotone truncation
    # Starts at lag 0, processes pairs (0,1), (2,3), (4,5), ...
    # τ = -1 + 2 * Σ ρ̂_t  (including ρ̂_0)
    max_lag = n - 3
    max_lag < 1 && return Float64(n * m)

    τ = -1.0
    prev_pair_sum = Inf
    t = 0
    while t < max_lag
        # Mean biased autocovariance at lag t across chains
        acov_t = 0.0
        len_t = n - t
        @inbounds for j in 1:m
            s = 0.0
            @simd for i in 1:len_t
                s += xc[i, j] * xc[i + t, j]
            end
            acov_t += s * inv_n
        end
        acov_t /= m
        ρ_t = 1.0 - (W - acov_t) / var_plus

        if t + 1 >= n
            τ += 2.0 * ρ_t
            break
        end

        acov_t1 = 0.0
        len_t1 = n - t - 1
        @inbounds for j in 1:m
            s = 0.0
            @simd for i in 1:len_t1
                s += xc[i, j] * xc[i + t + 1, j]
            end
            acov_t1 += s * inv_n
        end
        acov_t1 /= m
        ρ_t1 = 1.0 - (W - acov_t1) / var_plus

        pair = ρ_t + ρ_t1
        pair < 0.0 && break

        # Monotone sequence: enforce pair_sum ≤ prev_pair_sum
        if pair > prev_pair_sum
            ρ_t  = prev_pair_sum / 2.0
            ρ_t1 = prev_pair_sum / 2.0
            pair = prev_pair_sum
        end
        prev_pair_sum = pair

        τ += 2.0 * (ρ_t + ρ_t1)
        t += 2
    end

    # Safety floor (matching Stan)
    total = Float64(n * m)
    floor_val = 1.0 / log10(total)
    τ = max(τ, floor_val)

    return total / τ
end

"""
    _rank_normalize(x::AbstractMatrix{Float64})

Replace values with rank-based normal scores (pooled across chains).
Matches Stan: average ranks for ties, Blom fractional offset `(r - 0.375)/(N + 0.25)`.
"""
function _rank_normalize(x::AbstractMatrix{Float64})
    n, m = size(x)
    N = n * m
    pooled = vec(x)
    order = sortperm(pooled)

    # Assign ranks with averaging for ties (matching Stan)
    ranks = Vector{Float64}(undef, N)
    k = 1
    @inbounds while k <= N
        k_start = k
        while k < N && pooled[order[k + 1]] == pooled[order[k]]
            k += 1
        end
        avg_rank = 0.5 * (k_start + k)  # 1-based average rank
        for t in k_start:k
            ranks[order[t]] = avg_rank
        end
        k += 1
    end

    # Inverse normal CDF via Blom offset
    inv_denom = 1.0 / (N + 0.25)
    sqrt2 = sqrt(2.0)
    z = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        p = (ranks[i] - 0.375) * inv_denom
        z[i] = sqrt2 * erfinv(2.0 * p - 1.0)
    end
    return reshape(z, n, m)
end

## ── Per-column diagnostics (all three in one call) ──────────────────────────

"""
    _split_chains(x::AbstractMatrix{Float64}) → Matrix{Float64}

Split each chain in half → (n÷2, 2m) matrix. Matching Stan: odd middle
draw discarded. Second half starts at `(n+1)÷2 + 1` (1-based).
"""
function _split_chains(x::AbstractMatrix{Float64})
    n, m = size(x)
    nhalf = n ÷ 2
    start2 = (n + 1) ÷ 2 + 1  # 1-based start of second half

    split = Matrix{Float64}(undef, nhalf, 2m)
    @inbounds for j in 1:m
        @simd for i in 1:nhalf
            split[i, 2j - 1] = x[i, j]
            split[i, 2j]     = x[start2 + i - 1, j]
        end
    end
    return split
end

"""
    _col_diagnostics(x::AbstractMatrix{Float64}) → (rhat, ess_bulk, ess_tail)

Matches Stan's `split_rank_normalized_ess`: splits chains, then computes
bulk ESS on rank-normalized data and tail ESS on indicator variables.
"""
function _col_diagnostics(x::AbstractMatrix{Float64})
    n, m = size(x)
    rhat = _split_rhat(x)

    # Need at least 4 draws per chain to split meaningfully
    if n < 4
        return (rhat, Float64(n * m), Float64(n * m))
    end

    # Split chains into halves (matching Stan)
    split = _split_chains(x)

    # ESS bulk: rank-normalize the split chains, then compute ESS
    essb = _ess(_rank_normalize(split))

    # ESS tail: indicator variables at 5th/95th percentiles of split chains
    pooled = vec(split)
    q05 = quantile(pooled, 0.05)
    q95 = quantile(pooled, 0.95)
    ns, ms = size(split)
    lo = Matrix{Float64}(undef, ns, ms)
    hi = Matrix{Float64}(undef, ns, ms)
    @inbounds for j in 1:ms
        @simd for i in 1:ns
            lo[i, j] = ifelse(split[i, j] <= q05, 1.0, 0.0)
            hi[i, j] = ifelse(split[i, j] >= q95, 1.0, 0.0)
        end
    end
    esst = min(_ess(lo), _ess(hi))

    return (rhat, essb, esst)
end

## ── Pretty-print ────────────────────────────────────────────────────────────

const _RHAT_WARN = 1.01
const _ESS_WARN  = 400.0

"""Build element label: `mu[1,2]` for matrices, `theta[1]` for vectors, `sigma` for scalars."""
function _elem_label(name::Symbol, shape::Tuple, flat_idx::Int)
    isempty(shape) && return String(name)
    ci = CartesianIndices(shape)[flat_idx]
    idxs = join(Tuple(ci), ",")
    return "$(name)[$idxs]"
end

const _RowSummary = @NamedTuple{label::String, mean::Float64, std::Float64,
                                q025::Float64, q975::Float64, rhat::Float64,
                                essb::Float64, esst::Float64, warn::Bool}

function Base.show(io::IO, ::MIME"text/plain", c::Chains)
    ns = size(c.data, 1)
    nc = size(c.data, 3)
    np = length(c.params)
    N  = ns * nc
    println(io, "Chains: $ns samples × $np parameters × $nc chains\n")

    # ── compute per-element rows ─────────────────────────────────────────
    rows = _RowSummary[]
    for p in c.params
        for (k, col) in enumerate(p.cols)
            x = @view c.data[:, col, :]

            s = 0.0; s2 = 0.0
            @inbounds for j in 1:nc
                @simd for i in 1:ns
                    v = x[i, j]; s += v; s2 += v * v
                end
            end
            μ  = s / N
            σ  = sqrt(max(s2 / N - μ * μ, 0.0))
            pooled = vec(x)
            lo = quantile(pooled, 0.025)
            hi = quantile(pooled, 0.975)

            rhat, essb, esst = _col_diagnostics(x)
            warn = rhat > _RHAT_WARN || essb < _ESS_WARN || esst < _ESS_WARN
            label = _elem_label(p.name, p.shape, k)

            push!(rows, (label=label, mean=μ, std=σ, q025=lo, q975=hi,
                         rhat=rhat, essb=essb, esst=esst, warn=warn))
        end
    end

    # sort: warnings first, then original order
    order = sortperm(rows; by = r -> r.warn ? 0 : 1)

    hdr = @sprintf("  %-20s %10s %10s %10s %10s %8s %8s %8s",
                   "Parameter", "Mean", "Std", "2.5%", "97.5%",
                   "R̂", "ESS_bulk", "ESS_tail")
    println(io, hdr)
    println(io, "  " * "─"^(length(hdr) - 2))

    n_total = length(order)
    n_show = min(10, n_total)
    for i in 1:n_show
        r = rows[order[i]]
        row = @sprintf("  %-20s %10.4f %10.4f %10.4f %10.4f %8.4f %8.1f %8.1f",
                       r.label, r.mean, r.std, r.q025, r.q975,
                       r.rhat, r.essb, r.esst)
        if r.warn
            printstyled(io, row, "\n"; color=:yellow)
        else
            println(io, row)
        end
    end
    if n_total > n_show
        printstyled(io, "  ... $(n_total - n_show) more rows\n"; color=:light_black)
    end

    n_warn = count(r -> r.warn, rows)
    if n_warn > 0
        println(io)
        printstyled(io, "  ⚠ $n_warn element(s) with warnings ",
                    "(R̂ > $(_RHAT_WARN) or ESS < $(Int(_ESS_WARN)))\n"; color=:yellow)
    end
end
