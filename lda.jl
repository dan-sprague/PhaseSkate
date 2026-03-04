using PhaseSkate
using Random
using Distributions
using LinearAlgebra
using CairoMakie
using Distances, Clustering

# Stan naming: K=topics, V=vocab, M=docs, N=total word instances
function simulate_lda(; K=4, V=100, M=10, N_per_doc=500, beta_val=0.15, Σ=nothing)
    # topic-word distributions: each topic has sparse high words
    phi = hcat([rand(Dirichlet(fill(beta_val, V))) for _ in 1:K]...)  # V × K

    # non-diagonal covariance for correlated topic proportions
    if Σ === nothing
        A = randn(K, K)
        Σ = A'A / K + 0.5 * I
    end
    ln_dist = MvNormal(zeros(K), Σ)

    theta = zeros(M, K)
    counts = zeros(Int, M, V)
    for m in 1:M
        η = rand(ln_dist)
        theta[m, :] = exp.(η) ./ sum(exp.(η))
        for n in 1:N_per_doc
            z = rand(Distributions.Categorical(theta[m, :]))
            w = rand(Distributions.Categorical(phi[:, z]))
            counts[m, w] += 1
        end
    end
    return (; phi, theta, counts, Σ)
end

@skate LDA begin
    @constants begin
        K::Int               # num topics
        V::Int               # num words
        M::Int               # num docs
        x::Matrix{Float64}   # M × V count matrix
    end

    @params begin
        theta = param(Matrix{Float64}, M, K; simplex = true)  # topic dist for doc m
        phi = param(Matrix{Float64}, K, V; simplex = true, ordered = true)  # word dist, ordered by word 1
    end

    @logjoint begin
        for k in 1:K
            target += dirichlet_lpdf(@view(phi[k, :]), 0.3)
        end
        for m in 1:M
            target += dirichlet_lpdf(@view(theta[m, :]), 0.3)
            for v in 1:V
                target += x[m, v] * log_mix(@view(theta[m, :]), k -> log(phi[k, v]))
            end
        end
    end
end

sim = simulate_lda(K=4, V=100, M=10)
data = LDAData(K=4, M=size(sim.counts, 1), V=size(sim.counts, 2), x=Float64.(sim.counts))
model = make(data)
chain = PhaseSkate.sample(model, 2000; warmup=1000, max_depth=8)
