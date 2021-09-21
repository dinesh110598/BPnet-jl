include("BPnet.jl")
include("ising.jl")

module Trainer
import ..Ising: IsingParams, H
using ..BPnetModel
using CUDA, Flux, BSON
using Flux: @nograd
using ProgressMeter, Statistics
using ParameterSchedulers, LsqFit
using ParameterSchedulers:Stateful, Exp

export train_loop

include("ising.jl")

loss(lp, beta, E) = (lp .+ beta.*E)
@nograd loss #gradient tape will treat this as constant

function train_loop(model::BPnet, iter::Integer, beta::Real,
            pars::IsingParams;batch_size=100, anneal=false,
            init_eta=0.005, beta_anneal=0.9)
    
    opt = Flux.Optimise.ADAM(init_eta, (0.9, 0.999))
    lr_schedule = Stateful(Exp(0.75*init_eta, 0.75))

    if anneal
        beta_conv = beta
    end

    @. lin_reg(e, c) = -beta*e - c[1]
    x = sample(model, pars.L, batch_size)
    E = H(x, pars)
    lp = log_prob(model, x)
    fit = curve_fit(lin_reg, cpu(E), cpu(lp), [1000.])
    c = Float32.(coef(fit))
    
    p = Progress(iter; showspeed=true)
    for step in 1:iter
        if anneal
            beta = beta_conv*(1-beta_anneal^step)
        end
        x = sample(model, pars.L, batch_size)
        E = H(x, pars)
        local F, lp
        ps = params(model) #Extract trainable parameters
        gs = gradient(ps) do #Gradient of reinforced loss wrt parameters
            lp = log_prob(model, x)
            #F = loss(lp, beta, E)
            return Flux.Losses.mse(lp, -beta.*E .- c[1])
            #return mean((F .- mean(F, dims=1)) .* lp)
        end
        F = loss(lp, beta, E)
        free_ener_av = mean(F) / beta
        free_ener_std = std(F) / beta
        Flux.update!(opt, ps, gs)
        if step%100 == 0
            fit = curve_fit(lin_reg, cpu(E), cpu(lp), Float64.(c))
            c = Float32.(coef(fit))
        end
        if step%250 == 0
            opt.eta = ParameterSchedulers.next!(lr_schedule)
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
