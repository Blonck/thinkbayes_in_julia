module thinkbayes

export Pmf, create_pmf, prob, probs, total, mult!, normalize!
export maximumlikelihood, credibleinterval
export Suite, update!, mean, percentile
export Cdf, value

using SpecialFunctions

Pmf{T} = Dict{T, Float64} where T <: Any


"""
Encapsulates multiple hypotheses and their probabilities.
"""
struct Suite
    pmf :: Pmf
    likelihood


    """
        Suite(pmf::Pmf{T}, likelihood)

    Constuct a suite by a Pmf and a likelihood function.
    """
    function Suite(pmf::Pmf{T}, likelihood) where T <: Any
        new(pmf, likelihood)
    end


    """
        Suite(hypos::AbstractArray{T, 1}, likelihood)

    Construct a suite by a list of hypotheses and likelihood function.
    """
    function Suite(hypos::AbstractArray{T, 1}, likelihood) where T <: Any
        pmf = Pmf(hypos)
        new(pmf, likelihood)
    end
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
                 name="") where {T1<:Number, T2<:AbstractFloat}
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

end


"""
Represents a Beta distribution.

See http://en.wikipedia.org/wiki/Beta_distribution
"""
struct Beta
    α::Float64
    β::Float64

    function Beta(α::AbstractFloat = 1.0, β::AbstractFloat = 1.0)
        new(α, β)
    end
end


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
    Pmf(hypos::AbstractArray{T, 1}; prior::Function = x -> 1.0)

Create a probability mass function from a given arrays of hypotheses, each hypothesis
is initialized by calling init_prob(hypothesis::T).
"""
function Pmf(hypos::AbstractArray{T, 1}; prior::Function = x -> 1.0) where T <: Any
    pmf = Pmf{T}(hypo => prior(hypo) for hypo in hypos)
    normalize!(pmf)
end


"""
    create_pmf(hypos::AbstractArray{T, 1};
               prior::AbstractString,
               params::Dict{AbstractString, Any} = Dict{AbstractString, Any}()) where T <: Number

Create a probability mass function from a given arrays of hypotheses, each hypothesis
is initialized with a specific prior.

# Arguments
- `hypos::AbstractArray{T, 1}`: List of hypotheses.
- `mode::AbstractString`: Prior which should be used. Currently, only 'power_law' is supported.
- `params::Dict{AbstractString, Any}`: Parameter specific for each prior:
                                        * 'power_law': "alpha" (default: 1.0)
"""
function create_pmf(hypos::AbstractArray{T, 1};
                    prior::AbstractString,
                    params::Dict{AbstractString, Any} = Dict{AbstractString, Any}()) where T <: Number
    allowed_modes = ["power_law"]

    if !(prior in allowed_modes)
        error("mode must be one of $(allowed_modes)")
    end

    if prior == "power_law"
        α = get(params, "alpha", 1.0)
        power_law(x) = x^(-1.0 * α)
        return Pmf(hypos, prior = power_law)
    else
        return Pmf{T, Any}()
    end
end


"""
    Pmf(cdf::Cdf)

Constructs a probability mass function from a cumulative distribution function.
"""
function Pmf(cdf::Cdf)
    pmf = Pmf{Number, AbstractFloat}()

    prev = 0.0
    for (val, prob) in items(cdf)
        incr!(pmf, val, prob - prev)
        prev = prob
    end

    pmf
end


"""
    Pmf(beta::Beta, steps=101, name="")

Constructs a probability mass function from a beta distribution.:w

Note: Normally, we just evaluate the PDF at a sequence
of points and treat the probability density as a probability
mass.
But if alpha or beta is less than one, we have to be
more careful because the PDF goes to infinity at x=0
and x=1. In that case we evaluate the CDF and compute
differences.
"""
function Pmf(beta::Beta, steps=100.0, name="")
    if beta.α < 1.0 || beta.β < 1.0
        cdf = Cdf(beta)
        pmf = Pmf(cdf)
        return pmf
    end

    steps = float(steps)
    values = [i / (steps - 1.0) for i in (0.0:steps)]
    probs = [eval_pdf(x) for x in values]

    pmf
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
    incr!(self, x, term=1)

Increments the freq/prob associated with the value x.
"""
function incr!(pmf::Pmf{T}, x, term=1.0) where T <: Number
    pmf[x] = get(pmf, x, 0) + term
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


""" Construct a cumulative distribution function by a given probability mass function. """
Cdf(suite::Suite, name="") = Cdf(suite.pmf, name)


""" Construct a cumulative distribution function from a Beta distribution. """
function Cdf(beta::Beta, steps=100)
    steps = float(steps)
    values = [i / (steps - 1.0) for i in (0.0:steps)]
    probs = [beta_inc(beta.α, beta.β, x) for x in values]
    Cdf(values, probs)
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

items(cdf::Cdf) = zip(cdf.values, cdf.probs)


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


""" Print Cdf to console."""
function Base.show(io::IO, cdf::Cdf) where T <: Any
    print(io, "Cdf(")
    print(io, join(["$k=>$v" for (k, v) in zip(cdf.values, cdf.probs)], ","))
    print(io, ")")
end


""" Print Pmf as MIME type "text/plain"."""
function Base.show(io::IO, ::MIME"text/plain", cdf::Cdf) where T <: Any
    print(io, "Probability mass function:\n")
    for (k, v) in zip(cdf.values, cdf.probs)
        print(io, " $k => $v\n")
    end
end


"""Computes the mean of this distribution."""
mean(beta::Beta) = (beta.α) / (beta.α + beta.β)


"""Evaluates the PDF at x."""
eval_pdf(beta::Beta, x::AbstractFloat) = x^(beta.α - 1.0) * (1.0 - x)^(beta.β - 1.0)


function update!(beta::Beta, data::Tuple{Integer, Integer})
    heads, tails = data
    beta.β += heads
    beta.α += tails
end


end # end of module
