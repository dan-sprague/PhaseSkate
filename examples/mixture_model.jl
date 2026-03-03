# examples/mixture_model.jl
# Simple K=2 Gaussian mixture model with ordered means and simplex weights.
#
# Usage: julia examples/mixture_model.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PhaseSkate
using Random

# ── Model ────────────────────────────────────────────────────────────────────

@skate MixtureModel begin
    @constants begin
        N::Int
        K::Int
        x::Vector{Float64}
    end

    @params begin
        theta = param(Vector{Float64}, K; simplex = true)
        mus = param(Vector{Float64}, K; ordered = true)
        sigma = param(Float64; lower = 0.0)
    end

    @logjoint begin
        target += dirichlet_lpdf(theta, 1.0)
        target += multi_normal_diag_lpdf(mus, 0.0, 10.0)
        target += normal_lpdf(sigma, 0.0, 5.0)

        for i in 1:N
            target += log_mix(theta) do j
                normal_lpdf(x[i], mus[j], sigma)
            end
        end
    end
end

# ── Generate data ────────────────────────────────────────────────────────────

Random.seed!(42)
K = 2
N = 100
true_mus = [-2.0, 2.0]
true_sigma = 0.8
x_data = vcat(
    randn(N ÷ 2) .* true_sigma .+ true_mus[1],
    randn(N ÷ 2) .* true_sigma .+ true_mus[2]
)

# ── Fit ──────────────────────────────────────────────────────────────────────

d = MixtureModelData(N=N, K=K, x=x_data)
m = make(d)
println("Mixture Model — dim=$(m.dim)")

@time ch = sample(m, 1000; warmup=500, chains=4, seed=42)

println("\nPosterior means:")
println("  mus:   ", round.(mean(ch, :mus); digits=3))
println("  sigma: ", round(mean(ch, :sigma); digits=3))
println("  theta: ", round.(mean(ch, :theta); digits=3))
println("\nTrue values:")
println("  mus:   ", true_mus)
println("  sigma: ", true_sigma)
println("  theta: [0.5, 0.5]")
