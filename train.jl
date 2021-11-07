include("BPnet.jl")
include("ising.jl")

module Trainer
import ..Ising: IsingParams, H
using ..BPnetModel
using CUDA, Flux, BSON
using Flux: @nograd
using ProgressMeter, Statistics, Plots
using ParameterSchedulers, LsqFit
using ParameterSchedulers:Stateful, Exp

export train_loop, training_regimen, train_reg2

include("ising.jl")

loss(lp, beta, E) = (lp .+ beta.*E)
@nograd loss #gradient tape will treat this as constant

function train_loop(model::BPnet, iter::Integer, beta::Real,
            pars::IsingParams;batch_size=100, anneal=false,
            init_eta=0.005, beta_anneal=0.95)
    
    opt = Flux.Optimise.ADAM(init_eta, (0.9, 0.999))
    lr_schedule = Stateful(Exp(0.9*init_eta, 0.9))

    if anneal
        beta_conv = beta
    end

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
            F = loss(lp, beta, E)
            return mean((F .- mean(F, dims=1)) .* lp)
        end
        free_ener_av = mean(F) / beta
        free_ener_std = std(F) / beta
        Flux.update!(opt, ps, gs)
        if step%250 == 0
            opt.eta = ParameterSchedulers.next!(lr_schedule)
        end
        ProgressMeter.next!(p; showvalues = [(:Steps,step), (:Avg_F, free_ener_av), 
                            (:lr, opt.eta), (:F_std, free_ener_std)])
        if step%500 == 0
            cpu_model = cpu(model) 
            #There are complications in saving CuArray-based objects
            BSON.@save "Saves/Chkpts/BP_lreg_s$(step).bson" cpu_model
        end
    end
end

function train_reg(model::BPnet, iter::Integer, beta::Real,
            pars::IsingParams;batch_size=100, init_eta=0.005)
    opt = Flux.Optimise.ADAM(init_eta, (0.9, 0.999))
    lr_schedule = Stateful(Exp(0.9*init_eta, 0.9))

    x = sample(model, pars.L, 1000)
    E = H(x, pars)
    lp = log_prob(model, x)
    @. lin_reg(e, c) = -beta*e - c[1]
    fit = curve_fit(lin_reg, cpu(E), cpu(lp), [1000.])
    c = coef(fit)

    p = Progress(iter; showspeed=true)
    for step in 1:iter
        x = sample(model, pars.L, batch_size)
        E = H(x, pars)
        local F, lp
        ps = params(model) #Extract trainable parameters
        gs = gradient(ps) do #Gradient of reinforced loss wrt parameters
            lp = log_prob(model, x)
            F = Flux.Losses.mse(lp, lin_reg(E, c))
            return F
        end
        Flux.update!(opt, ps, gs)
        if step%50 == 0
            x = sample(model, pars.L, 1000)
            E = H(x, pars)
            lp = log_prob(model, x)
            fit = curve_fit(lin_reg, cpu(E), cpu(lp), Float64.(c))
            c = Float32.(coef(fit))
        end
        if step%250 == 0
            opt.eta = ParameterSchedulers.next!(lr_schedule)
        end
        ProgressMeter.next!(p; showvalues = [(:Steps,step), (:Coef,c), (:Loss, F), 
                            (:lr, opt.eta)])
        if step%500 == 0
            cpu_model = cpu(model) 
            #There are complications in saving CuArray-based objects
            BSON.@save "Saves/Chkpts/BP_lreg_s$(step).bson" cpu_model
        end
    end
end

function train_reg2(model::BPnet, iter::Integer, beta::Real,
            pars::IsingParams, c::Float64=nothing;batch_size=100, init_eta=0.005)
    opt = Flux.Optimise.ADAM(init_eta, (0.9, 0.999))
    lr_schedule = Stateful(Exp(0.9*init_eta, 0.9))

    if isnothing(c)
        x = sample(model, pars.L, 1000)
        E = H(x, pars)
        lp = log_prob(model, x)
        c = mean(-lp .- (beta.*E))
    end

    p = Progress(iter; showspeed=true)
    for step in 1:iter
        x = sample(model, pars.L, batch_size)
        E = H(x, pars)
        local F, lp
        ps = params(model) #Extract trainable parameters
        gs = gradient(ps) do #Gradient of reinforced loss wrt parameters
            lp = log_prob(model, x)
            F = Flux.Losses.mse(lp, -beta.*E .- c)
            return F
        end
        Flux.update!(opt, ps, gs)
        if step%250 == 0
            opt.eta = ParameterSchedulers.next!(lr_schedule)
        end
        ProgressMeter.next!(p; showvalues = [(:Steps,step), (:Coef,c), (:Loss, F), 
                            (:lr, opt.eta)])
        if step%500 == 0
            cpu_model = cpu(model) 
            #There are complications in saving CuArray-based objects
            BSON.@save "Saves/Chkpts/BP_lreg_s$(step).bson" cpu_model
        end
    end
end

function training_regimen(model, iter, beta, pars; init_eta=0.005)
    train_loop(model, iter÷2, beta, pars, init_eta=init_eta, anneal=true)
    train_reg(model, iter÷2, beta, pars, init_eta=0.7*init_eta)
end

end
##
