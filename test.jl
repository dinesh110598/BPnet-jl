##
include("train.jl")
using .BPnetModel
using .Trainer: hyperparams, train_loop
using CUDA, Flux

model = BPnet(9, 16, 2) |> gpu#values of kernel_size, net_width
#and net_depth
par = hyperparams(L=32, batch_size=100)
##
train_loop(model, 10, 0.45, par)
##
x = sample(model, 32, 50)
lp = log_prob(model, x)