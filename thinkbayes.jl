module thinkbayes

export Pmf, create_pmf, create_pmf_power_law, prob, probs, total, mult!, normalize!
export Suite, update!, mean, percentile


Pmf{T} = Dict{T, Float64} where T <: Any


function Base.show(io::IO, pmf::Pmf{T}) where T <: Any
    print(io, "Pmf(")
    print(io, join(["$k=>$v" for (k, v) in pmf], ","))
    print(io, ")")
end


function Base.show(io::IO, ::MIME"text/plain", pmf::Pmf{T}) where T <: Any
    print(io, "Probability mass function:\n")
    for (k, v) in pmf
        print(io, " $k => $v\n")
    end
end


function create_pmf(hypos::AbstractArray{T, 1}) where T <: Any
    pmf = Pmf{T}()
    for hypo in hypos
        pmf[hypo] = 1.0
    end
    normalize!(pmf)
end


function create_pmf(hypos::AbstractArray{T, 1}; init_prob::Function = x -> 1.0) where T <: Number
    pmf = Pmf{T}()
    for hypo in hypos
        pmf[hypo] = init_prob(hypo)
    end
    normalize!(pmf)
end


function create_pmf_power_law(hypos::AbstractArray{T, 1}; alpha=1.0) where T <: Number
    power_law(x) = x^(-1.0 * alpha)
    create_pmf(hypos, init_prob = power_law)
end


function prob(pmf::Pmf{T}, x::T, default::Float64=0.0) where T <: Any
    get(pmf, x, default)
end


function probs(pmf::Pmf, xs::AbstractArray, default::Float64=0.0)
    [prob(pmf, x, default) for x in xs]
end


function total(pmf::Pmf)
    sum(values(pmf))
end


function mult!(pmf::Pmf{T}, x::T, factor) where T <: Any
    pmf[x] = get(pmf, x, 0) * factor
    pmf
end


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


function mean(pmf::Pmf{T}) where T <: Number
    mu = 0.0
    for (hypo, prob) in pmf
        mu += hypo * prob
    end
    mu
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
