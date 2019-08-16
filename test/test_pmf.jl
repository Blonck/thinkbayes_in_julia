using Test
using thinkbayes

@testset "Pmf test" begin
    pmf = Pmf{Any}()

    pmf["Set A"] = 0.5
    pmf["Set B"] = 0.5

    # test prob
    @test prob(pmf, "Set A") == 0.5
    @test prob(pmf, "Set B") == 0.5
    @test prob(pmf, "Set C") == 0.0
    @test prob(pmf, "Set C", 1.0) == 1.0

    # test probs
    @test probs(pmf, ["Set A", "Set B"]) == [0.5, 0.5]

    # test total
    @test total(pmf) == 1.0

    # test normalize
    pmf["Set A"] = 1.0
    pmf["Set B"] = 1.0

    @test prob(pmf, "Set A") == 1.0
    @test prob(pmf, "Set B") == 1.0

    normalize!(pmf)
    @test prob(pmf, "Set A") == 0.5
    @test prob(pmf, "Set B") == 0.5

    normalize!(pmf, fraction=2.0)
    @test prob(pmf, "Set A") == 1.0
    @test prob(pmf, "Set B") == 1.0

    # test mult
    mult!(pmf, "Set A", 2.0)
    mult!(pmf, "Set B", 0.5)
    @test prob(pmf, "Set A") == 1.0 * 2.0
    @test prob(pmf, "Set B") == 1.0 * 0.5
end
