# PhaseSkate

*Scalable, high-performance Bayesian inference in native Julia.*

PhaseSkate is a Bayesian inference library built **for** [Enzyme](https://github.com/EnzymeAD/Enzyme.jl) LLVM autodifferentiation. It provides a Stan-like DSL for model definition and efficient MCMC samplers optimized for CPU.

## Design Principles

1. **Speed** — Pure Julia log-density functions with zero-allocation `@for` macro and Enzyme reverse-mode AD. No C++ backend, no FFI overhead.
2. **Clarity** — The `@skate` DSL defines constants, parameters, and the log-joint in a single cohesive block.
3. **Explicit accumulation** — `target += normal_lpdf(x, mu, sigma)` instead of tilde syntax. The log-joint accumulation is transparent.

## Quick Example

```julia
using PhaseSkate

@skate NormalModel begin
    @constants begin
        N::Int
        y::Vector{Float64}
    end
    @params begin
        mu::Float64
        sigma = param(Float64; lower=0.0)
    end
    @logjoint begin
        target += normal_lpdf(mu, 0.0, 10.0)
        target += exponential_lpdf(sigma, 1.0)
        for i in 1:N
            target += normal_lpdf(y[i], mu, sigma)
        end
    end
end

y_data = randn(100) .* 2.0 .+ 3.0
d = NormalModelData(N=100, y=y_data)
m = make(d)

ch = sample(m, 2000; warmup=1000, chains=4)
mean(ch, :mu)    # posterior mean
ci(ch, :mu)      # 95% credible interval
```

## Contents

```@contents
Pages = ["getting_started.md", "dsl.md", "samplers.md", "api.md"]
Depth = 2
```
