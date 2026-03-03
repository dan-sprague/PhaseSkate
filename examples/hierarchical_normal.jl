# examples/hierarchical_normal.jl
# Eight Schools hierarchical model (Rubin, 1981).
# A classic test model for MCMC samplers — tests ability to handle
# the funnel geometry in hierarchical models.
#
# Usage: julia examples/hierarchical_normal.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PhaseSkate

# ── Model ────────────────────────────────────────────────────────────────────

@skate EightSchools begin
    @constants begin
        J::Int
        y::Vector{Float64}
        sigma::Vector{Float64}
    end

    @params begin
        mu::Float64
        tau = param(Float64; lower=0.0)
        theta_raw = param(Vector{Float64}, J)
    end

    @logjoint begin
        # Hyperpriors
        target += normal_lpdf(mu, 0.0, 5.0)
        target += cauchy_lpdf(tau, 0.0, 5.0)

        # Non-centered parameterization: theta = mu + tau * theta_raw
        target += multi_normal_diag_lpdf(theta_raw, 0.0, 1.0)

        for j in 1:J
            theta_j = mu + tau * theta_raw[j]
            target += normal_lpdf(y[j], theta_j, sigma[j])
        end
    end
end

# ── Data (Rubin, 1981) ──────────────────────────────────────────────────────

y_obs     = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
sigma_obs = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]

d = EightSchoolsData(J=8, y=y_obs, sigma=sigma_obs)
m = make(d)
println("Eight Schools — dim=$(m.dim)")

# ── Fit ──────────────────────────────────────────────────────────────────────

@time ch = sample(m, 1000; warmup=500, chains=4, seed=42)

println("\nPosterior means:")
println("  mu:  ", round(mean(ch, :mu); digits=2))
println("  tau: ", round(mean(ch, :tau); digits=2))

theta_raw_mean = mean(ch, :theta_raw)
println("  theta (school effects):")
for j in 1:8
    theta_j = mean(ch, :mu) + mean(ch, :tau) * theta_raw_mean[j]
    println("    School $j: ", round(theta_j; digits=2))
end
