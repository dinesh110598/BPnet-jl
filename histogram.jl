include("MonteCarlo.jl")
using Flux, Statistics
using .MonteCarloMethods, .Ising, .BPnetModel
##
function MultipleHistogramMethod(ener::Array{Float64,2}, 
                β::Array{Float64,1}, q::Array{Float64,2}=ener,
                lnZ_init::Array{Float64,1}=ones(size(β)))
    
    m = size(β)[1]
    n = size(ener)[1]
    lnZ = lnZ_init
    Ẽ = reshape(ener, 1,n*m)
    for i in 1:100
        for k in 1:m
            lnZ[k] = logsumexp(-logsumexp(
                log(n) .- lnZ .- ((β[k].-β).*Ẽ), dims=1))
        end
    end
    βs = β[1]-0.005:0.002:β[end]+0.005
    lnZs = Array{Float64}(undef,size(βs)...)
    for k in 1:size(lnZs)[1]
        lnZs[k] = logsumexp(-logsumexp(
                log(n) .- lnZ .- ((βs[k].-β).*Ẽ), dims=1))
    end
    lnQ = Array{Float64}(undef, size(βs)...)
    q̃ = reshape(q, 1,n*m)
    flag = 0
    if all(q̃.<0)
        flag = 1
        q̃ = -q̃
    end
    
    for k in 1:size(lnQ)[1]
        lnQ[k] = -lnZs[k] + logsumexp(log.(q̃) .- logsumexp(
            log(n) .- lnZ .- ((βs[k].-β).*Ẽ), dims=1))
    end
    if flag==1
        Q = -exp.(lnQ)
    else
        Q = exp.(lnQ)
    end
    return βs, Q
end
##
prms = IsingParams(32)
eq_swp = 5000
tot_swp = 25000
mag = Array{Float64}(undef, tot_swp-eq_swp, 0)
ener = Array{Float64}(undef, tot_swp-eq_swp, 0)
init_conf = rand([1,-1], prms.L, prms.L);
##
betas = [0.43, 0.4407, 0.45]
for β in betas
    qs = WolffCluster(prms, eq_swp, tot_swp, init_conf, beta=β)
    mag = cat(mag, qs[1], dims=2)
    ener = cat(ener, qs[2], dims=2)
end
##
βs, E = MultipleHistogramMethod(ener, betas);
##
function MonteCarloRG(L=32, model::BPnet, beta::Float64)
    prms1 = IsingParams(L)
    mag, ener = neural_mc(model, prms, 1000, beta)
end