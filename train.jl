using Base: Integer, Real
##
module Trainer

include("BPnet.jl")
using BPnetModel
using CUDA, LsqFit
using Flux, BSON, Plots
using Flux: @nograd
using ProgressMeter, Statistics
using ParameterSchedulers
using ParameterSchedulers:Stateful, Exp

export L, batch_size, opt, lr_schedule, train_loop
#Need to use custom training loop for the unsupervised learning problem
#It's convenient to specify the loss as a function here

beta_anneal=0.99
L = 36
batch_size = 100
J = -1f0

include("ising.jl")

loss(lp, beta, E) = (lp .+ beta.*E)./(L^2)
@nograd loss #gradient

function Plateau(init_loss, patience=100; min_dist=0.01)
    is_plateau = let last_loss = init_loss
        L -> begin
            curr_loss = L
            Δ = last_loss - curr_loss
            last_loss = curr_loss
            return Δ < min_dist
        end
    end
    perseverence = let count=0
        L -> begin
            check = is_plateau(L)
            #If check true increase count
            count = check ? count+1 : 0
            cond = count == patience
            #Set count to zero if reached patience
            #steps without improvement in loss
            count = cond ? 0 : count
            return cond
        end
    end
    return perseverence
end

trigger = Plateau(-2.0, 100, min_dist=1e-3)
lr_schedule = Stateful(Exp(1e-3, 0.8))
opt = Flux.Optimise.ADAM(1e-3, (0.9, 0.999))

function train_loop(model::BPnet, iter::Integer, beta::Float32; anneal=true)
    if anneal
        beta_conv = beta
    end
    
    p = Progress(iter; showspeed=true)
    for step in 1:iter
        if anneal
            beta = beta_conv*(1-beta_anneal^step)
        end
        x = sample(model, L, batch_size)
        E = H(x)
        local free_ener_av
        ps = params(model) #Extract trainable parameters
        gs = gradient(ps) do #Gradient of reinforced loss wrt parameters
            lp = log_prob(model, x)
            F = loss(lp, beta, E)
            free_ener_av = mean(F) / beta
            return mean((F .- mean(F, dims=1)) .* lp)
        end
        Flux.update!(opt, ps, gs)
        if step%250 == 0
            opt.eta = ParameterSchedulers.next!(lr_schedule)
        end
        ProgressMeter.next!(p; showvalues = [(:Steps,step), (:Avg_F, free_ener_av), (:lr, opt.eta)])
        if step%500 == 0
            cpu_model = cpu(model) #Somehow, there are complications in saving CuArray-based objects
            BSON.@save "Saves/Chkpts/BP_s$(step).bson" cpu_model
        end
    end
end

end
##
