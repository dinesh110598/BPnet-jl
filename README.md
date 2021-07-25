# BPnet-jl
BPnet stands for Boltzmann-PixelCNNetwork which is a generative neural network that (approximately) samples configurations from the Boltzmann distribution for the 
classical spin system known as the Ising model on various 2d lattices. It's based on the NN architecture called [PixelCNN](https://arxiv.org/abs/1606.05328). We 
tweak the original design intended for image generation to account for the symmetries of the system and improve the sampling/generation speed as well as scalability
to larger system sizes, a development over a previous such attempt [here](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.080602). As a result, the
a neural network trained at smaller system size can sample bigger systems without further training.

## Note on programming language and ML library
We use the Flux package in Julia to build, train and deploy neural networks. Though not as popular as Tensorflow or PyTorch in the Python language, Flux provides an
automatic differentiation framework over native Julia types (like Arrays and CuArrays) and operations as well as slightly better speeds owing to Julia being a
compiled language. Tensorflow or PyTorch works with only objects and ops defined within the respective modules. This flexibility of Flux has already proven 
advantageous to implement latest innovations in NN architecture like [Neural ODE](https://julialang.org/blog/2019/01/fluxdiffeq/). So in the broader vision of 
developing DL models that aid study of physics, we would like not be limited by the functionality available in tf or torch and hence Flux.
