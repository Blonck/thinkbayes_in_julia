module thinkbayes

export Pmf, create_pmf, create_pmf_power_law, prob, probs, total, mult!, normalize!
export maximumlikelihood, credibleinterval
export Suite, update!, mean, percentile
export Cdf, value



Pmf{T} = Dict{T, Float64} where T <: Any

"""Print Pmf to console."""
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

""" Computes a credible interval for a given percentage """
function credibleinterval(pmf::Pmf{T}, percentage::Number) where T <: Number
    cdf = Cdf(pmf)
    credibleinterval(cdf, percentage)
end

"""
    maximumlikelihood(pmf::Pmf{T}) where T <: Any

Return the value with the highest probability
"""
function maximumlikelihood(pmf::Pmf{T}) where T <: Any
    prob, val = findmax(pmf)
    val
end


"""
Encapsulates multiple hypotheses and their probabilities.
"""
struct Suite
    pmf :: Pmf
    likelihood

    """
        Suite(hypos::AbstractArray{T, 1}, likelihood)

    Construct a suite by a list of hypotheses and likelihood function.
    """
    function Suite(hypos::AbstractArray{T, 1}, likelihood) where T <: Any
        new(create_pmf(hypos), likelihood)
    end

    """
        Suite(pmf::Pmf{T}, likelihood)

    Constuct a suite by a Pmf and a likelihood function.
    """
    function Suite(pmf::Pmf{T}, likelihood) where T <: Any
        new(pmf, likelihood)
    end
end


""" Helper function for updating a suite """
function unnormalized_update!(suite::Suite, datum)
    for hypo in keys(suite.pmf)
        like = suite.likelihood(suite.pmf, datum, hypo)
        mult!(suite.pmf, hypo, like)
    end
end


"""
    update!(suite::Suite, datum)

Updates a suite with a datum
"""
function update!(suite::Suite, datum)
    unnormalized_update!(suite, datum)
    normalize!(suite.pmf)
end


"""
    update!(suite::Suite, data::AbstractArray{T, 1}) where T <: Any

Update a suite with multiple data samples
"""
function update!(suite::Suite, data::AbstractArray{T, 1}) where T <: Any
    for datum in data
        unnormalized_update!(suite, datum)
    end
    normalize!(suite.pmf)
end


""" Mean value of all probabilities in a suite. """
mean(suite::Suite) = mean(suite.pmf)

""" Percentile of a suite """
percentile(suite::Suite, percentage::Number) = percentile(suite.pmf, percentage)

""" Returns probability of hypothesis x from probability mass function in suite """
prob(suite::Suite, x::T, default::Float64=0.0) where T <: Number = prob(suite.pmf, x, default)

""" Return the value with the highest probability """
maximumlikelihood(suite::Suite) = maximumlikelihood(suite.pmf)

""" Computes a credible interval for a given percentage """
credibleinterval(suite::Suite, percentage) = credibleinterval(suite.pmf, percentage)


"""Print suite to console."""
function Base.show(io::IO, suite::Suite)
    print(io, "Suite($(suite.pmf))")
end

"""Print suite as MIME type "text/plain"."""
function Base.show(io::IO, ::MIME"text/plain", suite::Suite)
    print(io, "Bayesian suite\n")
    print(io, " current pmf:\n")
    for (k, v) in suite.pmf
        print(io, "  $k => $v\n")
    end
    print(io, " likelihood: $(string(suite.likelihood))\n")
end



"""
Represents a cumulative distribution function.
"""
struct Cdf
    """
    Sequence of values.
    """
    values :: AbstractArray{Number, 1}
    """
    Sequence of probabilities
    """
    probs :: AbstractArray{AbstractFloat, 1}
    """
    Name for this cdf
    """
    name

    """
        Cdf(values::AbstractArray{Number, 1}=[],probs::AbstractArray{Number, 1}=[], name="")

    Construct a cumulative distribution function by its values and corresponding probabilities.
    """
    function Cdf(values::AbstractArray{T1, 1}=[],
                 probs::AbstractArray{T2, 1}=[],
                 name="") where {T1<:Number, T2<:Number}
        new(values, probs, name)
    end

    """
        Cdf(items::Dict{T1, T2}, name="") where {T1<:Number, T2<:Number}

    Construct a cumulative distribution function by given a dictionary of values and
    their corresponding frequencies.
    """
    function Cdf(items::Dict{T1, T2}, name="") where {T1<:Number, T2<:Number}
        # all elements as sorted (by key) array of pairs
        sorted = sort(collect(items))
        values = first.(sorted)
        probs = float(cumsum(last.(sorted)))
        norm = sum(float(last.(sorted)))
        new(values, probs/norm, name)
    end

    """ Construct a cumulative distribution function by a given probability mass function. """
    Cdf(suite::Suite, name="") = Cdf(suite.pmf, name)
end


"""
    value(cdf::Cdf, p::AbstractFloat)

Return the value to a given probability of a cdf.

If the probability is not given in the discrete set of probabilities, the next larger
value is returned.
"""
function value(cdf::Cdf, p::AbstractFloat)
    if !(0.0 <= p <= 1.0)
        error("p must be in [0.0, 1.0]")
    end

    index = first(searchsorted(cdf.probs, p))
    cdf.values[index]
end


"""
    percentile(cdf::Cdf, percentage::Number)

Calculates the percentile of a cdf for a given percentage.

# Arguments
- `cdf::Cdf`: Cumulative distribution function
- `percentage::Number`: Percentage in [0, 100]
"""
function percentile(cdf::Cdf, percentage::Number)
    value(cdf, percentage / 100.0)
end


""" Computes a credible interval for a given percentage """
function credibleinterval(cdf::Cdf, percentage::Number)
    prob = (1.0 - percentage / 100.0) / 2.0
    interval = (value(cdf, prob), value(cdf, 1.0 - prob))
    return interval
end

"""Print Cdf to console."""
function Base.show(io::IO, cdf::Cdf) where T <: Any
    print(io, "Cdf(")
    print(io, join(["$k=>$v" for (k, v) in zip(cdf.values, cdf.probs)], ","))
    print(io, ")")
end

"""Print Pmf as MIME type "text/plain"."""
function Base.show(io::IO, ::MIME"text/plain", cdf::Cdf) where T <: Any
    print(io, "Probability mass function:\n")
    for (k, v) in zip(cdf.values, cdf.probs)
        print(io, " $k => $v\n")
    end
end


end # end of module
