push!(LOAD_PATH, pwd())

using thinkbayes
using Printf

pmf = Pmf{Any}()
for x in [1 2 3 4 5 6]
    pmf[x] = 1.0/6.0
end

println("pmf of six sided dice: $pmf")

# the cookie problem
pmf = Pmf{String}()
pmf["Bowl 1"] = 0.5
pmf["Bowl 2"] = 0.5

mult!(pmf, "Bowl 1", 0.75)
mult!(pmf, "Bowl 2", 0.5)
normalize!(pmf)

println("posterior distribution of the cookie problem $pmf")
println("posterior probability of Bowl 1 $( @sprintf("%.2f", prob(pmf, "Bowl 1")))")
println("")

# The Bayesian framework

cookie = create_pmf(["Bowl 1", "Bowl 2"])

println("initial pmf of the cookie problem $cookie")

mixes = Dict("Bowl 1" => Dict(:vanilla => 0.75, :chocolate => 0.5),
             "Bowl 2" => Dict(:vanilla => 0.5, :chocolate => 0.5))

function likelihood_cookie(pmf::Pmf, data, hypo)
    mix = mixes[hypo]
    mix[data]
end

function update_cookie!(pmf::Pmf, data)
    for hypo in keys(pmf)
        like = likelihood_cookie(pmf, data, hypo)
        mult!(pmf, hypo, like)
    end
    normalize!(pmf)
end

update_cookie!(cookie, :vanilla)
println("cookie probabilities after first update: $cookie")

for data in [:vanilla, :chocolate, :vanilla]
    update_cookie!(cookie, data)
end
println("cookie probabilities after all updates: $cookie")

println("")

# Monty Hall Problem

Monty = Pmf{Char}

function init_monty(hypos::String)
    monty = Monty()
    for hypo in hypos
        monty[hypo] = 1.0
    end
    normalize!(monty)
end

pmf = init_monty("ABC")
println("pmf of Monty Hall prolem $pmf")


function likelihood_monty(pmf::Pmf, data, hypo)
    if hypo == data
        0.0
    elseif hypo == 'A'
        0.5
    else
        1.0
    end
end

function update_monty!(pmf::Pmf, data)
    for hypo in keys(pmf)
        like = likelihood_monty(pmf, data, hypo)
        mult!(pmf, hypo, like)
    end
    normalize!(pmf)
end

update_monty!(pmf, 'B')

println("pmf of Monty Hall after seen data: $pmf")
