# benchmarks/compare.jl
# Read all benchmark results and print a comparison table.
#
# Usage: julia benchmarks/compare.jl

using Printf

# ── Minimal JSON reader ──────────────────────────────────────────────────────

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
    pos[] += 1
    start = pos[]
    while s[pos[]] != '"'
        s[pos[]] == '\\' && (pos[] += 1)
        pos[] += 1
    end
    result = s[start:pos[]-1]
    pos[] += 1
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
    pos[] += 1
    result = Any[]
    _skip_ws(s, pos)
    s[pos[]] == ']' && (pos[] += 1; return result)
    while true
        push!(result, _parse_json(s, pos))
        _skip_ws(s, pos)
        s[pos[]] == ',' ? (pos[] += 1) : break
    end
    pos[] += 1
    return result
end

function _parse_obj(s, pos)
    pos[] += 1
    result = Dict{String, Any}()
    _skip_ws(s, pos)
    s[pos[]] == '}' && (pos[] += 1; return result)
    while true
        _skip_ws(s, pos)
        key = _parse_str(s, pos)
        _skip_ws(s, pos)
        pos[] += 1  # skip :
        val = _parse_json(s, pos)
        result[key] = val
        _skip_ws(s, pos)
        s[pos[]] == ',' ? (pos[] += 1) : break
    end
    pos[] += 1
    return result
end

# ── Load results ─────────────────────────────────────────────────────────────

results_dir = joinpath(@__DIR__, "results")

result_files = Dict(
    "PhaseSkate" => joinpath(results_dir, "phaseskate_results.json"),
    "CmdStan"    => joinpath(results_dir, "stan_results.json"),
    "BlackJAX"   => joinpath(results_dir, "blackjax_results.json"),
)

results = Dict{String, Dict{String, Any}}()
for (name, path) in result_files
    if isfile(path)
        results[name] = read_json(path)
    end
end

if isempty(results)
    println("No result files found in $results_dir")
    println("Run the individual benchmark scripts first:")
    println("  julia benchmarks/bench_phaseskate.jl")
    println("  JAX_PLATFORMS=cpu python benchmarks/bench_blackjax.py")
    println("  Rscript benchmarks/bench_stan.R")
    exit(1)
end

# ── Print comparison table ───────────────────────────────────────────────────

println()
header = @sprintf("%-14s %10s %10s %10s %10s %12s",
                  "Backend", "Time(s)", "Min ESS", "ESS/s", "Max Rhat", "Divergences")
println(header)
println("─"^length(header))

# Sort by ESS/s descending
ordered = sort(collect(keys(results));
    by = k -> get(results[k], "ess_per_sec", 0.0),
    rev = true
)

for name in ordered
    r = results[name]
    t = get(r, "sampling_time_s", get(r, "total_time_s", NaN))
    ess = get(r, "min_ess_bulk", NaN)
    ess_s = get(r, "ess_per_sec", NaN)
    rhat = get(r, "max_rhat", NaN)
    divs = get(r, "divergences", 0)

    rhat_str = isnan(rhat) ? "N/A" : @sprintf("%.4f", rhat)
    @printf("%-14s %10.1f %10.0f %10.1f %10s %12d\n",
            name, t, ess, ess_s, rhat_str, divs)
end

println()
println("Missing backends: ", join(setdiff(keys(result_files), keys(results)), ", "))
