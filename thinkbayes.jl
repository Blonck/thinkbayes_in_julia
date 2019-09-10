module thinkbayes

export Pmf, create_pmf, create_pmf_power_law, prob, probs, total, mult!, normalize!
export Suite, update!, mean, percentile


Pmf{T} = Dict{T, Float64} where T <: Any

"""Print Pmf to command line."""
function Base.show(io::IO, pmf::Pmf{T}) where T <: Any
    print(io, "Pmf(")
    print(io, join(["$k=>$v" for (k, v) in pmf], ","))
    print(io, ")")
end

"""Print Pmf as MIME type "text/plain"."""
function Base.show(io::IO, ::MIME"text/plain", pmf::Pmf{T}) where T <: Any
    print(io, "Probability mass function:\n")
    for (k, v) in pmf
        print(io, " $k => $v\n")
    end
end


"""
    create_pmf(hypos::AbstractArray{T, 1})

Create a probability mass function from a given arrays of hypotheses, each hypothesis
with the same probability.
"""
function create_pmf(hypos::AbstractArray{T, 1}) where T <: Any
    pmf = Pmf{T}()
    for hypo in hypos
        pmf[hypo] = 1.0
    end
    normalize!(pmf)
end

"""
    create_pmf(hypos::AbstractArray{T, 1}; init_prob::Function = x -> 1.0)

Create a probability mass function from a given arrays of hypotheses, each hypothesis
is initiallized by calling init_prob(hypothesis::T).
"""
function create_pmf(hypos::AbstractArray{T, 1}; init_prob::Function = x -> 1.0) where T <: Number
    pmf = Pmf{T}()
    for hypo in hypos
        pmf[hypo] = init_prob(hypo)
    end
    normalize!(pmf)
end


"""
    create_pmf(hypos::AbstractArray{T, 1}; init_prob::Function = x -> 1.0)

Create a probability mass function from a given arrays of hypotheses, each hypothesis
is initiallized with a power law given by \$x^{- \\alpha}\$.
"""
function create_pmf_power_law(hypos::AbstractArray{T, 1}; α=1.0) where T <: Number
    power_law(x) = x^(-1.0 * α)
    create_pmf(hypos, init_prob = power_law)
end


"""
    prob(pmf::Pmf{T}, x::T, default::Float64=0.0)

Returns probability of hypothesis x from probability mass function pmf.
"""
function prob(pmf::Pmf{T}, x::T, default::Float64=0.0) where T <: Any
    get(pmf, x, default)
end


"""
    probs(pmf::Pmf, xs::AbstractArray, default::Float64=0.0)

Returns probabilities of all hypotheses in xs as list.
"""
function probs(pmf::Pmf, xs::AbstractArray, default::Float64=0.0)
    [prob(pmf, x, default) for x in xs]
end


"""
    total(pmf::Pmf)

Sums all probabilities of pmf.
"""
function total(pmf::Pmf)
    sum(values(pmf))
end


"""
    mult!(pmf::Pmf{T}, x::T, factor)

Scales the probability of hypothesis x with factor.
"""
function mult!(pmf::Pmf{T}, x::T, factor) where T <: Any
    pmf[x] = get(pmf, x, 0) * factor
    pmf
end

"""
    normalize!(pmf::Pmf; fraction::Float64=1.0)

Normalize probability mass function such that the sum of all probabilities
is fraction.
"""
function normalize!(pmf::Pmf; fraction::Float64=1.0)
    norm = total(pmf)

    if norm == 0.0
        error("Total probability is zero")
    end

    factor = fraction / norm
    for k in keys(pmf)
        pmf[k] = pmf[k] * factor
    end
    pmf
end


"""
    mean(pmf::Pmf{T})

Calculates the mean of pmf.
"""
function mean(pmf::Pmf{T}) where T <: Number
    mu = 0.0
    for (hypo, prob) in pmf
        mu += hypo * prob
    end
    mu
end


"""
    percentile(pmf::Pmf{T}, percentage::Number)

Calculates the percentile of a pmf for a given percentage.

# Arguments
- `pmf::Pmf`: Probability mass function
- `percentage::Number`: Percentage in [0, 100]
"""
function percentile(pmf::Pmf{T}, percentage::Number) where T<: Number
    if !(0.0 <= percentage <= 100.0)
        error("percentage must be in [0, 100]")
    end

    p = percentage / 100.0
    total = 0.0
    ret = 0.0
    # TODO: replacing Dict with SortedDict from DataStructures.jl
    # could be an option
    for (val, prob) in sort(collect(pmf))
        total += prob
        if total >= p
            ret = val
            break
        end
    end
    ret
end


struct Suite
    pmf :: Pmf
    likelihood

    function Suite(hypos::AbstractArray{T, 1}, likelihood) where T <: Any
        new(create_pmf(hypos), likelihood)
    end

    function Suite(pmf::Pmf{T}, likelihood) where T <: Any
        new(pmf, likelihood)
    end
end

function update!(suite::Suite, data)
    for hypo in keys(suite.pmf)
        like = suite.likelihood(suite.pmf, data, hypo)
        mult!(suite.pmf, hypo, like)
    end
    normalize!(suite.pmf)
end

mean(suite::Suite) = mean(suite.pmf)

percentile(suite::Suite, percentage::Number) = percentile(suite.pmf, percentage)

function Base.show(io::IO, suite::Suite)
    print(io, "Suite($(suite.pmf))")
end

function Base.show(io::IO, ::MIME"text/plain", suite::Suite)
    print(io, "Bayesian suite\n")
    print(io, " current pmf:\n")
    for (k, v) in suite.pmf
        print(io, "  $k => $v\n")
    end
    print(io, " likelihood: $(string(suite.likelihood))\n")
end


end # end of module
