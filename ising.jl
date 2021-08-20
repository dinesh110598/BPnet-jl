"""
Computes and returns the Hamiltonian/energy of the
input configuration of Ising model, x

__x__ ::AbstractArray - Input configuration of the Ising model
    with dims=1,2 for the lattice position and dims=3,4 for
    the channels and batch position respectively

__shape__ ::String - shape of the lattice, either "square" or
    "triangle" allowed.

__pbc__ ::Bool - specify whether periodic boundary conditon is 
    imposed
"""
J = -1f0

function H(x::AbstractArray, shape="square", pbc=true)
    if pbc
        #x_n stores the sum of neighbors at every position
        x_n = circshift(x, (1,0,0,0))
        x_n = x_n .+ circshift(x, (0,1,0,0))
        if shape=="tri"
            x_n = x_n .+ circshift(x, (1,1,0,0))
        end
        energy = J .* x.*(x_n)
        energy = sum(energy, dims=(1,2,3))
    else
        #n_sum stores the sum of neighbors, reduced over dims=1,2,3
        n_sum = sum(x[2:end,:,:,:].*x[1:end-1,:,:,:], dims=(1,2,3))
        n_sum = n_sum .+ sum(x[:,2:end,:,:].*x[:,1:end-1,:,:], dims=(1,2,3))
        if shape=="tri"
            n_sum = n_sum .+ sum(x[1:end-1,1:end-1,:,:].*x[2:end,2:end,:,:], 
                                dims=(1,2,3))
        end
        energy = J .* (n_sum)
    end
    energy = reshape(energy, size(x)[4])
    return energy
end