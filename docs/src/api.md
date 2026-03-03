# API Reference

## Model Definition

```@docs
PhaseSkate.@skate
```

## Core Types

```@docs
ModelLogDensity
log_prob
```

## Samplers

```@docs
sample
sample_mclmc
sample_adjusted_mclmc
```

## Chains & Diagnostics

```@docs
Chains
samples
mean(::Chains, ::Symbol)
ci
thin
min_ess
```

## Simulation-Based Calibration

```@docs
sbc
calibrated
```

## Constraint Transforms

```@docs
IdentityConstraint
LowerBounded
UpperBounded
Bounded
SimplexConstraint
OrderedConstraint
simplex_transform
ordered_transform
corr_cholesky_transform
```

## Log-Density Functions

### Continuous Univariate

```@docs
normal_lpdf
cauchy_lpdf
exponential_lpdf
gamma_lpdf
beta_lpdf
lognormal_lpdf
student_t_lpdf
uniform_lpdf
laplace_lpdf
logistic_lpdf
weibull_lpdf
weibull_logsigma_lpdf
weibull_logsigma_lccdf
```

### Discrete

```@docs
poisson_lpdf
binomial_lpdf
bernoulli_logit_lpdf
binomial_logit_lpdf
neg_binomial_2_lpdf
beta_binomial_lpdf
categorical_logit_lpdf
dirichlet_lpdf
```

### Multivariate

```@docs
multi_normal_diag_lpdf
multi_normal_cholesky_lpdf
multi_normal_cholesky_scaled_lpdf
lkj_corr_cholesky_lpdf
diag_pre_multiply
```

## Utilities

```@docs
log_sum_exp
log_mix
```
