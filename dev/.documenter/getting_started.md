
# Getting Started {#Getting-Started}

## Installation {#Installation}

```julia
using Pkg
Pkg.add(url="https://github.com/YOUR_USERNAME/PhaseSkate.jl")
```


Or for local development:

```julia
using Pkg
Pkg.develop(path="/path/to/PhaseSkate")
```


## Workflow {#Workflow}

Every PhaseSkate analysis follows three steps:

### 1. Define the model {#1.-Define-the-model}

Use the `@skate` macro to define your model in a single block:

```julia
using PhaseSkate

@skate MyModel begin
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
```


This generates:
- A data struct `MyModelData` with the fields from `@constants`
  
- A `make(data::MyModelData)` function that returns a compiled `ModelLogDensity`
  

### 2. Build and sample {#2.-Build-and-sample}

```julia
d = MyModelData(N=100, y=randn(100))
m = make(d)
ch = sample(m, 2000; warmup=1000, chains=4)
```


### 3. Inspect results {#3.-Inspect-results}

```julia
mean(ch, :mu)          # posterior mean
ci(ch, :mu)            # 95% credible interval
ci(ch, :mu; level=0.9) # 90% credible interval
samples(ch, :mu)       # raw samples (nsamples × nchains)
min_ess(ch)            # minimum bulk ESS across all parameters
```


Printing a `Chains` object displays a summary table with mean, std, credible intervals, R-hat, and ESS for each parameter.

## Choosing a Sampler {#Choosing-a-Sampler}

PhaseSkate provides three samplers:

|                  Function |             Algorithm |                                     Best for |
| -------------------------:| ---------------------:| --------------------------------------------:|
|                `sample()` |      NUTS (No-U-Turn) | General purpose, well-understood diagnostics |
|          `sample_mclmc()` |                 MCLMC |             High dimensions, biased but fast |
| `sample_adjusted_mclmc()` | Adjusted MCLMC (MAMS) |                    High dimensions, unbiased |


```julia
# NUTS
ch = sample(m, 2000; warmup=1000, chains=4)

# MCLMC
ch = sample_mclmc(m, 2000; warmup=1000, chains=4)

# Adjusted MCLMC
ch = sample_adjusted_mclmc(m, 2000; warmup=1000, chains=4)
```

