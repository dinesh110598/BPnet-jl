include("BPnet.jl")

module Trainer
using ..BPnetModel
using CUDA, Flux, BSON
using Flux: @nograd
using ProgressMeter, Statistics
using ParameterSchedulers, LsqFit
using ParameterSchedulers:Stateful, Exp

export train_loop, hyperparams
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

include("ising.jl")

loss(lp, beta, E) = (lp .+ beta.*E)
@nograd loss #gradient tape will treat this as constant

lr_schedule = Stateful(Exp(5e-6, 0.8))
opt = Flux.Optimise.ADAM(1e-5, (0.9, 0.999))

function train_reinforce(model, beta, par)
    x = sample(model, par.L, par.batch_size)
    E = H(x, "square", false)
    local free_ener_av
    ps = params(model) #Extract trainable parameters
    gs = gradient(ps) do #Gradient of reinforced loss wrt parameters
        lp = log_prob(model, x)
        F = loss(lp, beta, E)
        free_ener_av = mean(F) / beta
        return mean((F .- mean(F, dims=1)) .* lp)
    end
    return free_ener_av
end

function train_regression(model, beta, par, c)
    x = sample(model, par.L, par.batch_size)
    E = H(x, "square", false)
    local free_ener_av
    ps = params(model)
    gs = gradient(ps) do 
        lp = log_prob(model, x)
        Flux.Losses.mse(lp, (c[1]*beta).*E .+ c[2])  
    end
end

function train_loop(model::BPnet, iter::Integer, beta::Real, 
            par::hyperparams; anneal=false)
    if anneal
        beta_conv = beta
    end

    @. lin_reg(e, c) = -0.45*e - c[1]
    x = sample(model, par.L, par.batch_size)
    E = H(x, "square", false)
    lp = log_prob(model, x)
    fit = curve_fit(lin_reg, cpu(E), cpu(lp), [1000.])
    c = Float32.(coef(fit))
    
    p = Progress(iter; showspeed=true)
    for step in 1:iter
        if anneal
            beta = beta_conv*(1-par.beta_anneal^step)
        end
        x = sample(model, par.L, par.batch_size)
        E = H(x, "square", false)
        local F, lp
        ps = params(model) #Extract trainable parameters
        gs = gradient(ps) do #Gradient of reinforced loss wrt parameters
            lp = log_prob(model, x)
            #F = loss(lp, beta, E)
            return Flux.Losses.mse(lp, -0.45 .*E .- c[1])
            #return mean((F .- mean(F, dims=1)) .* lp)
        end
        F = loss(lp, beta, E)
        free_ener_av = mean(F) / beta
        free_ener_std = std(F) / beta
        Flux.update!(opt, ps, gs)
        if step%250 == 0
            opt.eta = ParameterSchedulers.next!(lr_schedule)
            fit = curve_fit(lin_reg, cpu(E), cpu(lp), Float64.(c))
            c = Float32.(coef(fit))
        end
        ProgressMeter.next!(p; showvalues = [(:Steps,step), (:Avg_F, free_ener_av), (:lr, opt.eta),
                                        (:F_std, free_ener_std)])
        if step%500 == 0
            cpu_model = cpu(model) 
            #There are complications in saving CuArray-based objects
            BSON.@save "Saves/Chkpts/BP_lreg_s$(step).bson" cpu_model
        end
    end
end

end
##
