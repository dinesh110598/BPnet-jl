include("BPnet.jl")

module Trainer
using ..BPnetModel
using CUDA
using Flux, BSON, Plots
using Flux: @nograd
using ProgressMeter, Statistics
using ParameterSchedulers
using ParameterSchedulers:Stateful, Exp

export L, batch_size, opt, lr_schedule, train_loop, params
#Need to use custom training loop for the unsupervised learning problem
#It's convenient to specify the loss as a function here

struct hyperparams
    beta_anneal::Real
    L::Integer
    batch_size::Integer
end

"""
    function params(L=32, batch_size=100; beta_anneal=0.9)

    Object that holds the values of some hyperparameters for training

    L: Lattice length

    batch_size: size of training batch

    beta_anneal(optional): Factor for simulated annealing during training
"""
function hyperparams(;L=32, batch_size=100,
    beta_anneal=0.9)
    
    return hyperparams(beta_anneal, L, batch_size)
end

J = -1f0

include("ising.jl")

loss(lp, beta, E) = (lp .+ beta.*E)
@nograd loss #gradient tape will treat this as constant

lr_schedule = Stateful(Exp(1e-3, 0.8))
opt = Flux.Optimise.ADAM(1e-3, (0.9, 0.999))

function train_loop(model::BPnet, iter::Integer, beta::Real, 
            par::hyperparams; anneal=false)
    if anneal
        beta_conv = beta
    end
    
    p = Progress(iter; showspeed=true)
    for step in 1:iter
        if anneal
            beta = beta_conv*(1-par.beta_anneal^step)
        end
        x = sample(model, par.L, par.batch_size)
        E = H(x)
        local free_ener_av
        ps = params(model) #Extract trainable parameters
        gs = gradient(ps) do #Gradient of reinforced loss wrt parameters
            lp = log_prob(model, x)
            F = loss(lp, beta, E)./(par.L^2)
            free_ener_av = mean(F) / beta
            return mean((F .- mean(F, dims=1)) .* lp)
        end
        Flux.update!(opt, ps, gs)
        if step%250 == 0
            opt.eta = ParameterSchedulers.next!(lr_schedule)
        end
        ProgressMeter.next!(p; showvalues = [(:Steps,step), (:Avg_F, free_ener_av), (:lr, opt.eta)])
        if step%500 == 0
            cpu_model = cpu(model) 
            #There are complications in saving CuArray-based objects
            BSON.@save "Saves/Chkpts/BP_s$(step).bson" cpu_model
        end
    end
end

end
##
