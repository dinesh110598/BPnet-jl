# BPnet-jl
## Introduction
BPnet stands for Boltzmann-PixelCNNetwork which is a generative neural network that (approximately) samples configurations from the Boltzmann distribution for the classical spin system known as the [Ising model](https://en.wikipedia.org/wiki/Ising_model) on various 2d lattices. It's based on the NN architecture called [PixelCNN](https://arxiv.org/abs/1606.05328). We tweak the original design intended for image generation to account for the symmetries of the system and improve the sampling/generation speed as well as scalability to larger system sizes, a development over a previous such [attempt](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.080602). As a result, a neural network trained at smaller system size can sample bigger systems without further training. This makes it a computationally viable candidate to approximately calcualate quantities like Helmholtz free energy as well as speed up Monte Carlo simulations near the critical temperature.

We've included some Jupyter notebooks to illustrate the meaning behind the code in .jl files and perform some experiments with the models and methods defined inside them.

## Neural Monte Carlo - goals
We are trying to propose a new method based on autoregressive neural networks, for simulating the Ising model at equilibrium. We already have well-established Monte Carlo methods like the super simple **Single Spin Flip Metropolis** algorithm and the **Wolff cluster-update** algorithm (which is known to be faster near phase transition).

There's a caveat though- Since periodic boundary conditions is much trickier to implement for autoregressive networks, we'll work with an "open" boundary of zero-valued spins. We then evaluate a Monte Carlo *time series* (using either Metropolis or Wolff) of two qunatities at different temperatures- absolute magnetization and energy. From here, we can calculate routinely computed quantities like average magnetization, average energy, magnetic susceptibility and specific heat and energy as well as the errors in them. Apparently, reproducing these quantities *accurately* and *quickly* is sufficient to show that there's merit in any alternative new method we propose.

## Note on programming language and ML library
We use the Flux package in Julia to build, train and deploy neural networks. Though not as popular as Tensorflow or PyTorch in the Python language, Flux provides an automatic differentiation framework over native Julia types (like Arrays and CuArrays) and operations as well as slightly better speeds owing to Julia being a compiled language. Tensorflow or PyTorch works with only objects and ops defined within the respective modules. This flexibility of Flux has already proven advantageous to implement latest innovations in NN architecture like [Neural ODE](https://julialang.org/blog/2019/01/fluxdiffeq/). So in the broader vision of developing DL models that aid study of physics, we would like not be limited by the functionality available in tf or torch, hence Flux.

### Notes:
- torch conv2d weight shape: (out_channels, in_channels, kernel_rows, kernel_cols), bias shape: (out_channels,)
- Flux conv2d weight shape: (kernel_rows, kernel_cols, in_channels, out_channels), bias shape: (out_channels,)
### Debugging a possible bug in torch implementation:
1. Succesfully used PyCall to transfer the state_dic