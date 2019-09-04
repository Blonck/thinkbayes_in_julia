push!(LOAD_PATH, pwd())

using thinkbayes

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
