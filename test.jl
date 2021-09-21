##
include("MonteCarlo.jl")
using .BPnetModel
using .Ising: IsingParams, H
using .Trainer: train_loop
using .MonteCarloMethods
using CUDA, Flux, BSON, Statistics, LsqFit
using Plots
##
model = BPnet(18, 16, 2) |> gpu
##
L = 32
prmts = IsingParams(L)
beta = 0.43
train_loop(model, 1000, beta, prmts, init_eta=0.0066)
##
BSON.@load "Saves/mish_linreg_L32_b_0.46.bson" cpu_model c
model = gpu(cpu_model)
##
L = 64
prmts = IsingParams(L)
beta = 0.45
##
cpu_model = cpu(model)
BSON.@save "Saves/mish_linreg_L32_b_0.46.bson" cpu_model c
##
x = sample(model, L, 2000)
lp = log_prob(model, x);
E = H(x, prmts);
#F = (lp./beta) .+ E;
##
@. reg(e, p) = -p[1]*beta*e - p[2]
fit = curve_fit(reg, cpu(E), cpu(lp), [1.,1000.]);
c = coef(fit);
##
plot(cpu(E), reg(cpu(E), coef(fit)), label="lp=-0.996*0.45*E-949.5",
    xlabel="Total Energy(E)", ylabel="Log Probability(lp)")
scatter!(cpu(E), cpu(lp), markersize=2, markercolor="black", label="Actual lp")
##
log(1+sqrt(2f0))/2f0

##
BSON.@load "Saves/mish_linreg_L32_b_0.43.bson" cpu_model c
model = gpu(cpu_model)
##
beta = 0.45
L=64
##
prms = IsingParams(L)
mag, ener = neural_mc(model, prms, 1000, beta, n=20)
##
results = PhysicalObservables(mag, ener, beta, L);
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
CSV.write("Saves/MC_Data/WolffMCData.csv", WolffMC_data)