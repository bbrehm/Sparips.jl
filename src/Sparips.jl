module Sparips
using StaticArrays, Distances, DataStructures

const RIPSER_BIN = joinpath(dirname(pathof(Sparips)), "../deps/ripser/ripser")
const RIPSER_COEFF_BIN = joinpath(dirname(pathof(Sparips)), "../deps/ripser/ripser-coeff")

export Ctree, sparsify_ctree, APriori, SemiAPosteriori, APosteriori, runrips, plotpers

include("ctree.jl")
include("precision_func.jl")
include("sparse_assembly.jl")
include("ripswrap.jl")
include("plot_pers.jl")
end 