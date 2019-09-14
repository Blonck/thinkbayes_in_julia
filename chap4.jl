push!(LOAD_PATH, pwd())

using thinkbayes
using UnicodePlots

function likelihood_euro(pmf::Pmf, data, hypo)
    x = hypo
    if data == 'H'
        return x/100.0
    else
        return 1.0 - x/100.0
    end
end

euro = Suite(collect(0:100), likelihood_euro)
data = vcat(repeat(['H'], 140), repeat(['T'], 110))

update!(euro, data)

plt = scatterplot(collect(keys(euro.pmf)), collect(values(euro.pmf)),
                  name = "uniform")
println("Euro suite after seen data, mean: $(mean(euro))")
println("Euro suite after seen data, max. likelihood: $(maximumlikelihood(euro))")
println(plt)
