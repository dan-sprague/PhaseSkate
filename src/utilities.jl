
"""
    log_sum_exp(x)

The 'Bare Metal' stability trick for Logit/Softmax math.
"""
function log_sum_exp(x)
    isempty(x) && throw(ArgumentError("log_sum_exp: input must be non-empty"))
    max_x = maximum(x)
    s = zero(eltype(x))
    for xi in x
        s += exp(xi - max_x)
    end
    return max_x + log(s)
end

"""
    log_mix(weights, f)

Zero-allocation log-mixture-density (branchless).
`f(j)` returns the log-density of component `j`.

Usage with `do` syntax:
    log_mix(theta) do j
        normal_lpdf(x[i], mus[j], sigma)
    end
"""
function log_mix(f, weights)
    K = length(weights)
    acc = log(weights[1]) + f(1)
    for j in 2:K
        lp_j = log(weights[j]) + f(j)
        m = max(acc, lp_j)
        acc = m + log(exp(acc - m) + exp(lp_j - m))
    end
    return acc
end

"""
    log_mix(a, b)

Log-sum-exp of elementwise `a .+ b`, zero-allocation.
`a` and `b` are vectors of log-values (e.g. log-weights and log-likelihoods).

    log_mix(log_theta, log_phi) == log(sum(exp.(log_theta .+ log_phi)))
"""
function log_mix(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    K = length(a)
    acc = a[1] + b[1]
    for j in 2:K
        lp = a[j] + b[j]
        mx = max(acc, lp)
        acc = mx + log(exp(acc - mx) + exp(lp - mx))
    end
    return acc
end

"""
    log_mix(a, b, offset)

`log_sum_exp(a .+ offset .+ b)`, zero-allocation.
"""
function log_mix(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}, offset::Real)
    K = length(a)
    acc = a[1] + offset + b[1]
    for j in 2:K
        lp = a[j] + offset + b[j]
        mx = max(acc, lp)
        acc = mx + log(exp(acc - mx) + exp(lp - mx))
    end
    return acc
end
