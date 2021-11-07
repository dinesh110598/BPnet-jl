##
include("MonteCarlo.jl")
using .BPnetModel
using .Ising: IsingParams, H
using .Trainer: train_loop, train_reg2, training_regimen
using .MonteCarloMethods
using CUDA, Flux, BSON, Statistics, LsqFit
using Plots, DataFrames, CSV, PyCall, StatsBase
##
L = 32
prmts = IsingParams(L)
beta = 0.5
BSON.@load "Saves/mish_(9,16,2)/L$(L)_b_$(beta).bson" cpu_model
model = gpu(cpu_model);
##
r(x) = reverse(x, dims=(1, 2))
f(x) = PyReverseDims(x)
g(x) = PyReverseDims(permutedims(r(x), (2,1,3,4)));

py"""
import torch
import sys
sys.path.insert(1, '/home/dinesh/Documents/Repos/BPnet-jl')
from BPnetModel import BPnet
from collections import OrderedDict

def tensor_conv(d: dict, beta):
    dev = torch.device("cuda:0")
    for i, val in d.items():
        d[i] = torch.from_numpy(val).to(dev)
    od = []
    for i in range(2):
        od.extend([(f"blocks.{i}.l_conv.0.weight", 
                    d[f"blocks.{i}.l_conv.0.weight"]),
                   (f"blocks.{i}.l_conv.0.bias",
                    d[f"blocks.{i}.l_conv.0.bias"]),
                   (f"blocks.{i}.l_conv.1.weight",
                    d[f"blocks.{i}.l_conv.1.weight"]),
                   (f"blocks.{i}.l_conv.1.bias",
                    d[f"blocks.{i}.l_conv.1.bias"]),
                   (f"blocks.{i}.r_conv.0.weight",
                    d[f"blocks.{i}.r_conv.0.weight"]),
                   (f"blocks.{i}.r_conv.0.bias",
                    d[f"blocks.{i}.r_conv.0.bias"]),
                   (f"blocks.{i}.r_conv.1.weight",
                    d[f"blocks.{i}.r_conv.1.weight"]),
                   (f"blocks.{i}.r_conv.1.bias",
                    d[f"blocks.{i}.r_conv.1.bias"]),
                   (f"blocks.{i}.m_conv.0.weight",
                    d[f"blocks.{i}.m_conv.0.weight"]),
                   (f"blocks.{i}.m_conv.1.weight",
                    d[f"blocks.{i}.m_conv.1.weight"]),
                   (f"blocks.{i}.m_conv.1.bias",
                    d[f"blocks.{i}.m_conv.1.bias"]),
                   (f"blocks.{i}.m_conv.2.weight",
                    d[f"blocks.{i}.m_conv.2.weight"])
                   ])
    od.extend([("final_conv.0.weight", d["final_conv.0.weight"]),
               ("final_conv.0.bias", d["final_conv.0.bias"])
               ])
    od = OrderedDict(od)
    model = BPnet(9,16,2).to(dev)
    model.load_state_dict(od)
    torch.save(model.state_dict(),
               f'Saves/mish_(9,16,2)/L32_b_{beta}.pt')
"""
##
function exp_to_torch(model, beta)
    state_dict = Dict{String, Array{Float64}}()
    for i in 1:size(model.layers)[1]
        get!(state_dict, "blocks.$(i-1).l_conv.0.weight", g(model.layers[i].l_conv[1].weight))
        get!(state_dict, "blocks.$(i-1).l_conv.0.bias", f(model.layers[i].l_conv[1].bias))
        get!(state_dict, "blocks.$(i-1).l_conv.1.weight", g(model.layers[i].l_conv[2].weight))
        get!(state_dict, "blocks.$(i-1).l_conv.1.bias", f(model.layers[i].l_conv[2].bias))
        get!(state_dict, "blocks.$(i-1).r_conv.0.weight", g(model.layers[i].r_conv[1].weight))
        get!(state_dict, "blocks.$(i-1).r_conv.0.bias", f(model.layers[i].r_conv[1].bias))
        get!(state_dict, "blocks.$(i-1).r_conv.1.weight", g(model.layers[i].r_conv[2].weight))
        get!(state_dict, "blocks.$(i-1).r_conv.1.bias", f(model.layers[i].r_conv[2].bias))
        get!(state_dict, "blocks.$(i-1).m_conv.0.weight", g(model.layers[i].m_conv[1].weight))
        get!(state_dict, "blocks.$(i-1).m_conv.1.weight", g(model.layers[i].m_conv[2].weight))
        get!(state_dict, "blocks.$(i-1).m_conv.1.bias", f(model.layers[i].m_conv[2].bias))
        get!(state_dict, "blocks.$(i-1).m_conv.2.weight", g(model.layers[i].m_conv[3].weight))
    end
    get!(state_dict, "final_conv.0.weight", g(model.final_conv.weight))
    get!(state_dict, "final_conv.0.bias", f(model.final_conv.bias))
    py"tensor_conv"(state_dict, beta)
end
##
L = 32
prmts = IsingParams(L)
betas = [0.4, 0.4407, 0.46, 0.5]
hist = Array{Float64, 2}(undef, 10, 4)
F_range = Array{Float64, 2}(undef, 11, 4)
i = 1
for beta in betas
    BSON.@load "Saves/mish_(9,16,2)/L$(L)_b_$(beta).bson" cpu_model
    model = gpu(cpu_model)
    x = BPnetModel.sample(model, L, 2000)
    lp = log_prob(model, x);
    E = H(x, prmts);
    F = (lp./beta .+ E)./(L^2) |> cpu;
    F_range[:, i] = range(minimum(F), maximum(F), length=11)
    hist[:, i] = fit(Histogram, F, F_range[:, i]).weights
    i += 1
end
##
L = 32
prmts = IsingParams(L)
betas = [0.4, 0.4407, 0.46, 0.5]
for beta in betas
    model = BPnet(9, 16, 2) |> gpu
    train_loop(model, 500, beta, prmts, init_eta=0.008, anneal=true)
    cpu_model = cpu(model)
    BSON.@save "Saves/mish_(9,16,2)/L32_b_$(beta).bson" cpu_model
end
##
x = sample(model, L, 2000)
lp = log_prob(model, x);
E = H(x, prmts);
F = (lp./beta .+ E)./1024;
##
@. reg(e, p) = -0.5*p[2]*e - p[1]
fit = curve_fit(reg, cpu(E), cpu(lp), [1000., 1.]);
c = coef(fit)
confidence_interval(fit)
##
Flux.Losses.mse(lp, reg(E, c))
##
plot(cpu(E), reg(cpu(E), coef(fit)),
    xlabel="Total Energy(E)", ylabel="Log Probability(lp)")
scatter!(cpu(E), cpu(lp), markersize=2, markercolor="black", label="Actual lp")
##
betas = [0.4, 0.4407, 0.46, 0.5]
L = 32
prms = IsingParams(L)
for beta in betas 
    BSON.@load "Saves/mish_(9,16,2)/L$(L)_b_$(beta).bson" cpu_model
    model = gpu(cpu_model)
    mag, ener = neural_mc(model, prms, 1000, beta, n=20)    
    BSON.@save "Saves/MC_Data/NeuralMC2_L$(L)_b_$(beta).bson" mag ener
end
##
betas = [0.4, 0.4407, 0.46, 0.5]
NeuralMC2_L32 = DataFrame(beta=Float64[], m=Float64[], δm=Float64[], 
                        E=Float64[], δE=Float64[], χ=Float64[], δχ=Float64[],
                        C=Float64[], δC=Float64[])
for beta in betas
    BSON.@load "Saves/MC_Data/NeuralMC2_L$(L)_b_$(beta).bson" mag ener
    results = PhysicalObservables(mag, ener, beta, L)
    push!(NeuralMC2_L32, (beta, results.m, results.δm, results.E, results.δE,
                    results.χ, results.δχ, results.C, results.δC))
end

CSV.write("Saves/MC_Data/NeuralMC2_L32.csv", NeuralMC2_L32)
##
betas = [0.4, 0.4407, 0.46, 0.5]
L = 64
prms = IsingParams(L)
for beta in betas
    init_conf = rand([1,-1], L, L)
    mag, ener = WolffCluster(prms, 10000, 30000, init_conf, beta=beta)
    BSON.@save "Saves/MC_Data/WolffMC_L$(L)_b_$(beta).bson" mag ener
end
##
WolffMC_L64 = DataFrame(beta=Float64[], m=Float64[], δm=Float64[], 
                        E=Float64[], δE=Float64[], χ=Float64[], δχ=Float64[],
                        C=Float64[], δC=Float64[])
for beta in betas
    BSON.@load "Saves/MC_Data/WolffMC_L$(L)_b_$(beta).bson" mag ener
    results = PhysicalObservables(mag, ener, beta, L)
    push!(WolffMC_L64, (beta, results.m, results.δm, results.E, results.δE,
                    results.χ, results.δχ, results.C, results.δC))
end

CSV.write("Saves/MC_Data/WolffMC_L64.csv", WolffMC_L64)
##
prms = IsingParams(L)
init_conf = rand([1,-1], prms.L, prms.L)
mag3, ener3 = WolffCluster(prms, 10000, 30000, init_conf, beta=beta)
##
results3 = PhysicalObservables(mag3, ener3, beta, L);
##
BSON.@load "Saves/MC_Data/Smart_L64_b0.43.bson" mag ener
BSON.@load "Saves/MC_Data/Wolff_L64_b0.43.bson" mag3 ener3
##
WolffMC_data = DataFrame(L=Int[], beta=Float64[], m=Float64[], δm=Float64[], 
                        E=Float64[], δE=Float64[], χ=Float64[], δχ=Float64[],
                        C=Float64[], δC=Float64[])
##
push!(WolffMC_data, (L, beta, results3.m, results3.δm, results3.E, results3.δE,
                    results3.χ, results3.δχ, results3.C, results3.δC))
##
CSV.write("Saves/MC_Data/WolffMC_L32.csv", WolffMC_L32)