# benchmarks/bench_phaseskate.jl
# Benchmark PhaseSkate on the JointALM model using shared data from generate_data.jl.
#
# Usage: julia benchmarks/bench_phaseskate.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PhaseSkate
using Printf

# ── Minimal JSON reader (no external deps) ──────────────────────────────────

function read_json(path)
    txt = read(path, String)
    return _parse_json(txt, Ref(1))
end

function _skip_ws(s, pos)
    while pos[] <= length(s) && s[pos[]] in (' ', '\t', '\n', '\r')
        pos[] += 1
    end
end

function _parse_json(s, pos)
    _skip_ws(s, pos)
    c = s[pos[]]
    if c == '{'
        return _parse_obj(s, pos)
    elseif c == '['
        return _parse_arr(s, pos)
    elseif c == '"'
        return _parse_str(s, pos)
    elseif c == 't'
        pos[] += 4; return true
    elseif c == 'f'
        pos[] += 5; return false
    elseif c == 'n'
        pos[] += 4; return nothing
    else
        return _parse_num(s, pos)
    end
end

function _parse_str(s, pos)
    pos[] += 1  # skip opening "
    start = pos[]
    while s[pos[]] != '"'
        if s[pos[]] == '\\'
            pos[] += 1
        end
        pos[] += 1
    end
    result = s[start:pos[]-1]
    pos[] += 1  # skip closing "
    return result
end

function _parse_num(s, pos)
    start = pos[]
    while pos[] <= length(s) && s[pos[]] in ('-', '+', '.', 'e', 'E', '0':'9'...)
        pos[] += 1
    end
    numstr = s[start:pos[]-1]
    if occursin('.', numstr) || occursin('e', numstr) || occursin('E', numstr)
        return parse(Float64, numstr)
    else
        return parse(Int, numstr)
    end
end

function _parse_arr(s, pos)
    pos[] += 1  # skip [
    result = Any[]
    _skip_ws(s, pos)
    if s[pos[]] == ']'
        pos[] += 1
        return result
    end
    while true
        push!(result, _parse_json(s, pos))
        _skip_ws(s, pos)
        if s[pos[]] == ','
            pos[] += 1
        else
            break
        end
    end
    pos[] += 1  # skip ]
    return result
end

function _parse_obj(s, pos)
    pos[] += 1  # skip {
    result = Dict{String, Any}()
    _skip_ws(s, pos)
    if s[pos[]] == '}'
        pos[] += 1
        return result
    end
    while true
        _skip_ws(s, pos)
        key = _parse_str(s, pos)
        _skip_ws(s, pos)
        pos[] += 1  # skip :
        val = _parse_json(s, pos)
        result[key] = val
        _skip_ws(s, pos)
        if s[pos[]] == ','
            pos[] += 1
        else
            break
        end
    end
    pos[] += 1  # skip }
    return result
end

# JSON writing
function write_json(path, obj)
    open(path, "w") do io
        _write_json(io, obj)
        println(io)
    end
end

function _write_json(io, x::AbstractDict)
    print(io, "{")
    first = true
    for (k, v) in x
        first || print(io, ",")
        first = false
        print(io, "\n  \"$k\": ")
        _write_json(io, v)
    end
    print(io, "\n}")
end

_write_json(io, x::Real) = isinteger(x) && abs(x) < 1e15 ? print(io, Int(x)) : print(io, x)
_write_json(io, x::Integer) = print(io, x)
_write_json(io, x::AbstractString) = print(io, "\"$x\"")

function _write_json(io, x::AbstractVector)
    print(io, "[")
    for (i, v) in enumerate(x)
        i > 1 && print(io, ", ")
        _write_json(io, v)
    end
    print(io, "]")
end

# ── Configuration ────────────────────────────────────────────────────────────

const NUM_CHAINS  = 4
const NUM_WARMUP  = 1000
const NUM_SAMPLES = 2000
const SEED        = 42

bench_dir   = @__DIR__
data_path   = joinpath(bench_dir, "data", "joint_alm_data.json")
result_path = joinpath(bench_dir, "results", "phaseskate_results.json")

# ── Model definition ────────────────────────────────────────────────────────

@skate JointALM begin
    @constants begin
        n1::Int
        n2::Int
        p::Int
        n_countries::Int
        MRC_MAX::Int
        tier1_times::Vector{Float64}
        tier1_X::Matrix{Float64}
        tier1_country_ids::Vector{Int}
        tier1_obs_idx::Vector{Int}
        tier1_cens_idx::Vector{Int}
        tier2_times::Vector{Float64}
        tier2_X::Matrix{Float64}
        tier2_country_ids::Vector{Int}
        tier2_obs_idx::Vector{Int}
        tier2_cens_idx::Vector{Int}
        total_mrc_obs::Int
        mrc_scores_flat::Vector{Int}
        mrc_times_flat::Vector{Float64}
        mrc_patient_ids::Vector{Int}
    end

    @params begin
        log_shape::Float64
        log_scale::Float64
        beta_s     = param(Vector{Float64}, p)
        beta_k     = param(Vector{Float64}, p)
        sigma_country_k = param(Float64; lower=0.0)
        sigma_country_s = param(Float64; lower=0.0)
        mu_country_k = param(Vector{Float64}, n_countries)
        mu_country_s = param(Vector{Float64}, n_countries)
        mu_k::Float64
        omega_k    = param(Float64; lower=0.0)
        gamma_k::Float64
        gamma_hill = param(Float64; lower=1.0)
        EC50       = param(Float64; lower=0.0, upper=1.0)
        log_phi::Float64
        P0         = param(Float64; lower=0.0, upper=1.0)
        z_k        = param(Vector{Float64}, n2)
    end

    @logjoint begin
        shape = exp(log_shape)
        inv_shape = 1.0 / shape
        phi = exp(log_phi)
        log_P0_ratio = log1p(-P0) - log(P0)
        log_EC50g = gamma_hill * log(EC50)

        target += normal_lpdf(log_shape, 0.2, 0.5)
        target += normal_lpdf(log_scale, 2.5, 1.0)
        target += multi_normal_diag_lpdf(beta_s, 0.0, 2.0)
        target += multi_normal_diag_lpdf(beta_k, 0.0, 0.5)
        target += cauchy_lpdf(sigma_country_k, 0.0, 0.5)
        target += cauchy_lpdf(sigma_country_s, 0.0, 1.0)
        target += multi_normal_diag_lpdf(mu_country_k, 0.0, sigma_country_k)
        target += multi_normal_diag_lpdf(mu_country_s, 0.0, sigma_country_s)
        target += normal_lpdf(mu_k, log(0.08), 0.5)
        target += cauchy_lpdf(omega_k, 0.0, 0.5)
        target += normal_lpdf(gamma_k, 1.0, 0.5)
        target += normal_lpdf(gamma_hill, 3.0, 1.0)
        target += beta_lpdf(EC50, 4.0, 6.0)
        target += normal_lpdf(log_phi, log(15.0), 0.5)
        target += beta_lpdf(P0, 2.0, 8.0)
        target += multi_normal_diag_lpdf(z_k, 0.0, 1.0)

        @for begin
            log_k_2 = mu_k .+ (tier2_X * beta_k) .+ mu_country_k[tier2_country_ids] .+ (omega_k .* z_k)
            log_eff_scale_2 = log_scale .- ((tier2_X * beta_s) .+ mu_country_s[tier2_country_ids] .+ gamma_k .* log_k_2) .* inv_shape
        end

        for idx in tier2_obs_idx
            target += weibull_logsigma_lpdf(tier2_times[idx], shape, log_eff_scale_2[idx])
        end
        for idx in tier2_cens_idx
            target += weibull_logsigma_lccdf(tier2_times[idx], shape, log_eff_scale_2[idx])
        end

        @for begin
            log_k_1 = mu_k .+ (tier1_X * beta_k) .+ mu_country_k[tier1_country_ids]
            log_eff_scale_1 = log_scale .- ((tier1_X * beta_s) .+ mu_country_s[tier1_country_ids] .+ gamma_k .* log_k_1) .* inv_shape
        end

        for idx in tier1_obs_idx
            target += weibull_logsigma_lpdf(tier1_times[idx], shape, log_eff_scale_1[idx])
        end
        for idx in tier1_cens_idx
            target += weibull_logsigma_lccdf(tier1_times[idx], shape, log_eff_scale_1[idx])
        end

        for i in 1:total_mrc_obs
            k_i = exp(log_k_2[mrc_patient_ids[i]])
            P_t_i = 1.0 / (1.0 + exp(log_P0_ratio - k_i * mrc_times_flat[i]))
            log_Pg_i = gamma_hill * log(max(P_t_i, 1e-9))
            mu_i = clamp(1.0 / (1.0 + exp(log_Pg_i - log_EC50g)), 1e-6, 1.0 - 1e-6)
            a_mrc = mu_i * phi
            b_mrc = (1.0 - mu_i) * phi
            target += beta_binomial_lpdf(mrc_scores_flat[i], MRC_MAX, a_mrc, b_mrc)
        end
    end
end

# ── Load data from JSON ─────────────────────────────────────────────────────

println("Loading data from: $data_path")
raw = read_json(data_path)

# Convert JSON arrays to proper Julia types
to_vec(x) = Float64.(x)
to_ivec(x) = Int.(x)
to_mat(x) = Float64.(reduce(hcat, [r for r in x])')  # row-major JSON → Julia matrix

d = JointALMData(
    n1               = raw["n1"],
    n2               = raw["n2"],
    p                = raw["p"],
    n_countries      = raw["n_countries"],
    MRC_MAX          = raw["MRC_MAX"],
    tier1_times      = to_vec(raw["tier1_times"]),
    tier1_X          = to_mat(raw["tier1_X"]),
    tier1_country_ids = to_ivec(raw["tier1_country_ids"]),
    tier1_obs_idx    = to_ivec(raw["tier1_obs_idx"]),
    tier1_cens_idx   = to_ivec(raw["tier1_cens_idx"]),
    tier2_times      = to_vec(raw["tier2_times"]),
    tier2_X          = to_mat(raw["tier2_X"]),
    tier2_country_ids = to_ivec(raw["tier2_country_ids"]),
    tier2_obs_idx    = to_ivec(raw["tier2_obs_idx"]),
    tier2_cens_idx   = to_ivec(raw["tier2_cens_idx"]),
    total_mrc_obs    = raw["total_mrc_obs"],
    mrc_scores_flat  = to_ivec(raw["mrc_scores_flat"]),
    mrc_times_flat   = to_vec(raw["mrc_times_flat"]),
    mrc_patient_ids  = to_ivec(raw["mrc_patient_ids"]),
)

m = make(d)
println("JointALM Model — dim=$(m.dim)")

# ── Sample ───────────────────────────────────────────────────────────────────

println("\nSampling: $NUM_WARMUP warmup, $NUM_SAMPLES samples, $NUM_CHAINS chains")

t_start = time()
ch = sample_adjusted_mclmc(m, NUM_SAMPLES; warmup=NUM_WARMUP, chains=NUM_CHAINS, seed=SEED)
t_total = time() - t_start

# ── Diagnostics ──────────────────────────────────────────────────────────────

display(ch)

ess_min = min_ess(ch)
sampling_time = t_total  # PhaseSkate includes warmup in the total

println("\n", "── Results ", "─"^39)
@printf("  Total wall time:  %.1f s\n", t_total)
@printf("  Min ESS (bulk):   %.0f\n", ess_min)
@printf("  ESS/s:            %.1f\n", ess_min / t_total)

# ── Save results ─────────────────────────────────────────────────────────────

mkpath(joinpath(bench_dir, "results"))

results = Dict{String, Any}(
    "backend"         => "PhaseSkate",
    "num_chains"      => NUM_CHAINS,
    "num_warmup"      => NUM_WARMUP,
    "num_samples"     => NUM_SAMPLES,
    "total_time_s"    => round(t_total; digits=2),
    "sampling_time_s" => round(t_total; digits=2),
    "min_ess_bulk"    => round(ess_min; digits=1),
    "ess_per_sec"     => round(ess_min / t_total; digits=1),
    "divergences"     => 0,
)

write_json(result_path, results)
println("\nResults saved to: $result_path")
