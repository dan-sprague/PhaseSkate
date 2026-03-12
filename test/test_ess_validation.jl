"""
Validate PhaseSkate's ESS computation against Stan's algorithm.
Generates test chains, feeds them to both PhaseSkate and standalone
C++/Python reference implementations, and checks agreement.
"""

using Test
using Random
using Statistics
using Printf

# ── Build C++ reference if needed ──
const TEST_DIR = @__DIR__
const CPP_SRC  = joinpath(TEST_DIR, "stan_ess.cpp")
const CPP_BIN  = joinpath(tempdir(), "stan_ess")
const PY_SRC   = joinpath(TEST_DIR, "stan_ess.py")

function ensure_cpp_built()
    if !isfile(CPP_BIN) || mtime(CPP_SRC) > mtime(CPP_BIN)
        run(`g++ -std=c++17 -O2 -o $CPP_BIN $CPP_SRC`)
    end
end

# ── Write chains to file in column-major format ──
function write_chains(path, chains::Matrix{Float64})
    N, M = size(chains)
    open(path, "w") do f
        println(f, "$N $M")
        for j in 1:M
            for i in 1:N
                @printf(f, "%.17e\n", chains[i, j])
            end
        end
    end
end

# ── Run C++ reference ──
function run_cpp_ess(chains::Matrix{Float64})
    ensure_cpp_built()
    tmpfile = tempname()
    write_chains(tmpfile, chains)
    out = read(pipeline(`/bin/cat $tmpfile`, `$CPP_BIN`), String)
    rm(tmpfile)
    vals = parse.(Float64, split(strip(out)))
    return (ess_raw=vals[1], ess_bulk=vals[2], ess_tail=vals[3])
end

# ── Run Python reference ──
function run_py_ess(chains::Matrix{Float64})
    tmpfile = tempname()
    write_chains(tmpfile, chains)
    out = read(`python3 $(PY_SRC) $(tmpfile)`, String)
    rm(tmpfile)
    vals = parse.(Float64, split(strip(out)))
    return (ess_raw=vals[1], ess_bulk=vals[2], ess_tail=vals[3])
end

# ── PhaseSkate's internal ESS functions ──
# Import the internal functions we want to test
using PhaseSkate
const PS = PhaseSkate

# ── Generate AR(1) test chains ──
function make_ar1_chains(N, M; rho=0.7, seed=42)
    rng = Random.Xoshiro(seed)
    chains = zeros(N, M)
    for j in 1:M
        chains[1, j] = randn(rng)
        for i in 2:N
            chains[i, j] = rho * chains[i-1, j] + randn(rng) + (j-1) * 0.1
        end
    end
    chains
end

# ── Test cases ──
@testset "ESS validation against Stan (C++ & Python)" begin

    @testset "AR(1) chains, N=$N, M=$M" for (N, M) in [(200, 4), (500, 2), (100, 8), (1000, 1)]
        chains = make_ar1_chains(N, M)

        cpp = run_cpp_ess(chains)
        py  = run_py_ess(chains)

        # C++ and Python must agree with each other (< 1e-6 relative)
        @test isapprox(cpp.ess_raw,  py.ess_raw;  rtol=1e-6)
        @test isapprox(cpp.ess_bulk, py.ess_bulk; rtol=1e-6)
        @test isapprox(cpp.ess_tail, py.ess_tail; rtol=1e-6)

        # PhaseSkate bulk ESS (from _col_diagnostics)
        _, ps_bulk, ps_tail = PS._col_diagnostics(chains)

        @printf("  N=%4d M=%d: cpp_bulk=%7.1f  py_bulk=%7.1f  ps_bulk=%7.1f  |  cpp_tail=%7.1f  ps_tail=%7.1f\n",
                N, M, cpp.ess_bulk, py.ess_bulk, ps_bulk, cpp.ess_tail, ps_tail)

        # PhaseSkate should match Stan within 5% relative tolerance
        # (small differences expected due to rank normalization inverse CDF implementation)
        @test isapprox(ps_bulk, cpp.ess_bulk; rtol=0.05)
        @test isapprox(ps_tail, cpp.ess_tail; rtol=0.05)
    end

    @testset "IID chains (ESS should ≈ N*M)" begin
        rng = Random.Xoshiro(123)
        chains = randn(rng, 500, 4)
        cpp = run_cpp_ess(chains)
        _, ps_bulk, _ = PS._col_diagnostics(chains)

        @printf("  IID: cpp_bulk=%7.1f  ps_bulk=%7.1f  (expected ~%d)\n",
                cpp.ess_bulk, ps_bulk, 500*4)

        # For IID data, ESS should be close to total draws
        @test cpp.ess_bulk > 1500  # at least 75% of 2000
        @test ps_bulk > 1500
        @test isapprox(ps_bulk, cpp.ess_bulk; rtol=0.05)
    end

    @testset "Highly autocorrelated chains" begin
        chains = make_ar1_chains(300, 4; rho=0.95)
        cpp = run_cpp_ess(chains)
        _, ps_bulk, _ = PS._col_diagnostics(chains)

        @printf("  ρ=0.95: cpp_bulk=%7.1f  ps_bulk=%7.1f\n", cpp.ess_bulk, ps_bulk)

        @test isapprox(ps_bulk, cpp.ess_bulk; rtol=0.05)
    end

    @testset "Mean-shifted chains (between-chain variance)" begin
        rng = Random.Xoshiro(999)
        chains = randn(rng, 200, 4)
        # Add large mean shifts between chains
        for j in 1:4
            chains[:, j] .+= (j - 1) * 5.0
        end
        cpp = run_cpp_ess(chains)
        _, ps_bulk, _ = PS._col_diagnostics(chains)

        @printf("  Shifted: cpp_bulk=%7.1f  ps_bulk=%7.1f\n", cpp.ess_bulk, ps_bulk)

        @test isapprox(ps_bulk, cpp.ess_bulk; rtol=0.05)
    end
end
