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

@skate CTM begin
    @constants begin
        K::Int                    # num topics
        V::Int                    # num words
        M::Int                    # num docs
        x::Matrix{Float64}       # M × V count matrix
        log_phi::Matrix{Float64} # K × V  (log of fixed word distributions)
        L::Matrix{Float64}      # K × K  Cholesky factor of Σ (lower triangular)
    end

    @params begin
        eta = param(Matrix{Float64}, M, K)  # unconstrained log-ratios
    end

    @logjoint begin
        for m in 1:M
            # prior: eta[m,:] ~ MvNormal(0, Σ) via Cholesky
            target += multi_normal_cholesky_lpdf(@view(eta[m, :]), 0.0, L)

            # softmax to get theta
            max_eta = eta[m, 1]
            for k in 2:K
                max_eta = max(max_eta, eta[m, k])
            end
            denom = 0.0
            for k in 1:K
                denom += exp(eta[m, k] - max_eta)
            end
            log_denom = max_eta + log(denom)

            for v in 1:V
                # log p(w_v | theta_m, phi) = log sum_k theta_mk * phi_kv
                #   = log_sum_exp_k (log_theta_mk + log_phi_kv)
                mix_max = eta[m, 1] - log_denom + log_phi[1, v]
                for k in 2:K
                    mix_max = max(mix_max, eta[m, k] - log_denom + log_phi[k, v])
                end
                mix_sum = 0.0
                for k in 1:K
                    mix_sum += exp(eta[m, k] - log_denom + log_phi[k, v] - mix_max)
                end
                target += x[m, v] * (mix_max + log(mix_sum))
            end
        end
    end
end

sim = simulate_lda(K=4, V=100, M=10)
log_phi = log.(sim.phi')  # sim.phi is V×K, we need K×V
L = Matrix(cholesky(sim.Σ).L)
data = CTMData(
    K=4, M=size(sim.counts, 1), V=size(sim.counts, 2),
    x=Float64.(sim.counts), log_phi=log_phi, L=L
)
model = make(data)
chain = PhaseSkate.sample(model, 2000; warmup=1000, max_depth=8)
