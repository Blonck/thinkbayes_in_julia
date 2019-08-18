module thinkbayes

export Pmf, create_pmf, prob, probs, total, mult!, normalize!


Pmf{T} = Dict{T, Float64} where T <: Any


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

end
