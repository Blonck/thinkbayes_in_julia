push!(LOAD_PATH, "../")

using thinkbayes

# check if all chapters run
include("../chap2.jl")
include("../chap3.jl")

include("test_pmf.jl")
