module thinkbayes

export Pmf, create_pmf, prob, probs, total, mult!, normalize!
export Suite, update!


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

struct Suite
    pmf :: Pmf
    likelihood

    function Suite(hypos::AbstractArray{T, 1}, likelihood) where T <: Any
        new(create_pmf(hypos), likelihood)
    end
end

function update!(suite::Suite, data)
    for hypo in keys(suite.pmf)
        like = suite.likelihood(suite.pmf, data, hypo)
        mult!(suite.pmf, hypo, like)
    end
    normalize!(suite.pmf)
end

function Base.show(io::IO, suite::Suite)
    print(io, "Suite($(suite.pmf))")
end

function Base.show(io::IO, ::MIME"text/plain", suite)
    print(io, "Bayesian suite\n")
    print(io, " current pmf:\n")
    for (k, v) in suite.pmf
        print(io, "  $k => $v\n")
    end
    print(io, " likelihood: $(string(suite.likelihood))\n")
end



end # end of module
