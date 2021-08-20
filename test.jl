##
include("train.jl")
using .BPnetModel
using .Trainer: hyperparams, train_loop
using CUDA, Flux, BSON, Statistics, LsqFit
using Plots
##
model = BPnet(9, 16, 2) |> gpu#values of kernel_size, net_width
#and net_depth
##
par = hyperparams(L=32, batch_size=100)
train_loop(model, 750, 0.45, par)
##
BSON.@load "Saves/mish_linreg_(9,16,2).bson" cpu_model
model = gpu(cpu_model)
##
cpu_model = cpu(model)
BSON.@save "Saves/mish_linreg_(9,16,2).bson" cpu_model
##
include("ising.jl")
x = sample(model, 32, 2000)
E = H(x)
lp = log_prob(model, x);
E = H(x, "square", false);
#F = (lp./0.45) .+ E;
##
@. reg(e, p) = -p[1]*0.45*e - p[2]
fit = curve_fit(reg, cpu(E), cpu(lp), [1.,1000.]);
##
plot(cpu(E), reg(cpu(E), coef(fit)))
scatter!(cpu(E), cpu(lp), markersize=2, markercolor="black")
##