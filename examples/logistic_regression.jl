# examples/logistic_regression.jl
# Bayesian logistic regression with weakly informative priors.
#
# Usage: julia examples/logistic_regression.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PhaseSkate
using Random

# ── Model ────────────────────────────────────────────────────────────────────

@skate LogisticRegression begin
    @constants begin
        N::Int
        P::Int
        X::Matrix{Float64}
        y::Vector{Int}
    end

    @params begin
        alpha::Float64
        beta = param(Vector{Float64}, P)
    end

    @logjoint begin
        target += normal_lpdf(alpha, 0.0, 5.0)
        target += multi_normal_diag_lpdf(beta, 0.0, 2.5)

        for i in 1:N
            eta = alpha
            for j in 1:P
                eta += X[i, j] * beta[j]
            end
            target += bernoulli_logit_lpdf(y[i], eta)
        end
    end
end

# ── Generate data ────────────────────────────────────────────────────────────

Random.seed!(123)
N = 200
P = 3
true_alpha = -0.5
true_beta = [1.0, -0.5, 0.3]

X = randn(N, P)
logits = X * true_beta .+ true_alpha
probs = 1.0 ./ (1.0 .+ exp.(-logits))
y = Int.(rand(N) .< probs)

println("Generated data: $(sum(y)) positives out of $N observations")

# ── Fit ──────────────────────────────────────────────────────────────────────

d = LogisticRegressionData(N=N, P=P, X=X, y=y)
m = make(d)
println("Logistic Regression — dim=$(m.dim)")

@time ch = sample(m, 1000; warmup=500, chains=4, seed=42)

println("\nPosterior means:")
println("  alpha: ", round(mean(ch, :alpha); digits=3))
println("  beta:  ", round.(mean(ch, :beta); digits=3))
println("\nTrue values:")
println("  alpha: ", true_alpha)
println("  beta:  ", true_beta)
