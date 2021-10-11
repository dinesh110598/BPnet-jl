##
include("MonteCarlo.jl")
using .BPnetModel
using .Ising: IsingParams, H
using .Trainer: train_loop, train_reg2, training_regimen
using .MonteCarloMethods
using CUDA, Flux, BSON, Statistics, LsqFit
using Plots, DataFrames, CSV
##
L = 32
prmts = IsingParams(L)
beta = 0.5
BSON.@load "Saves/mish_(9,16,2)/L$(L)_b_$(beta).bson" cpu_model
model = gpu(cpu_model)
##
train_reg2(model, 500, beta, prmts, c, init_eta=0.01)
##
betas = [0.5]
for β in betas
    #model = BPnet(9, 16, 2) |> gpu
    #BSON.@load "Saves/mish_(9,16,2)/L32_b_$β.bson" cpu_model
    #model = gpu(cpu_model)
    train_reg(model, 500, β, prmts, init_eta=0.008)
    cpu_model = cpu(model)
    BSON.@save "Saves/mish_(9,16,2)/L32_b_$β.bson" cpu_model
end
##
betas = [0.5]
for β in betas
    model = BPnet(18, 16, 2) |> gpu
    training_regimen(model, 1000, β, prmts, init_eta=0.008)
    cpu_model = cpu(model)
    BSON.@save "Saves/mish_(18,16,2)/L32_b_$β.bson" cpu_model
end
##
train_loop(model, 250, beta, prmts, regression=true, init_eta=0.008, batch_size=50)
#training_regimen(model, 1000, beta, prmts, init_eta=0.005)
##
BSON.@load "Saves/mish_linreg_L32_b_0.5.bson" cpu_model
model = gpu(cpu_model)
##
cpu_model = cpu(model)
BSON.@save "Saves/mish_L64_b_0.5.bson" cpu_model
##
x = sample(model, L, 2000)
lp = log_prob(model, x);
E = H(x, prmts);
F = (lp./beta) .+ E;
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
betas = [0.5]
L = 32
prms = IsingParams(L)
for beta in betas 
    #BSON.@load "Saves/mish_(18,16,2)/L$(L)_b_$(beta).bson" cpu_model
    #model = gpu(cpu_model)
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