```@raw html
---
layout: home

hero:
  name: PhaseSkate
  text: Bayesian Inference in Native Julia
  tagline: Scalable, high-performance MCMC powered by Enzyme autodifferentiation
  actions:
    - theme: brand
      text: Getting Started
      link: /getting_started
    - theme: alt
      text: API Reference
      link: /api

features:
  - icon: ⚡
    title: Speed
    details: Pure Julia log-density functions with zero-allocation @for macro and Enzyme reverse-mode AD. No C++ backend, no FFI overhead.
  - icon: 🔍
    title: Clarity
    details: The @skate DSL defines constants, parameters, and the log-joint in a single cohesive block.
  - icon: 📐
    title: Explicit Accumulation
    details: "target += normal_lpdf(x, mu, sigma) instead of tilde syntax. The log-joint accumulation is transparent."
---
```

````@raw html
<div class="vp-doc" style="width:80%; margin:auto">

<h2> Quick Example </h2>

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

</div>
````
