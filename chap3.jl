push!(LOAD_PATH, pwd())

using thinkbayes

# dice problem

function likelihood_dice(pmf::Pmf, data, hypo)
    if hypo < data
        0.0
    else
        1.0/hypo
    end
end

dice = Suite([4, 6, 8, 12, 20], likelihood_dice)

println("Dice suite: $dice")

update!(dice, 6)

println("Dice suite after seen '6' data: $dice")

for roll in [6,8, 7, 7, 5, 4]
    update!(dice, roll)
end

println("Dice suite after seen all data: $dice")

# locomotive problem

function likelihood_train(pmf::Pmf, data, hypo)
    if hypo < data
        0.0
    else
        1.0/hypo
    end
end

train = Suite(collect(1:1000), likelihood_train)

println("Train suite, mean value: $(mean(train))")

update!(train, 60)

println("Train suite, mean value after seend data: $(mean(train))")


pmf_train_power_law = create_pmf_power_law(collect(1:2000))
train_pow_law = Suite(pmf_train_power_law, likelihood_train)

println("Train suite (power law posterior), mean value: $(mean(train_pow_law))")

for data in [30, 60, 90]
    update!(train_pow_law, data)
end

println("Train suite (power law posterior), mean value after seen data: "
        * "$(mean(train_pow_law))")
println("Credible intervall for train suite: "
        * "($(percentile(train_pow_law, 5)), $(percentile(train_pow_law, 95)))")
