module Ising

export IsingParams, H

struct IsingParams
    L::Integer
    pbc::Bool
    lattice::String
    J::Float64
end

function IsingParams(L=32;pbc=false, lattice="square", J=1e0)
    return IsingParams(L, pbc, lattice, J)
end

function H(x::AbstractArray{T,4}, params::IsingParams) where T
    if params.pbc
        #x_n stores the sum of neighbors at every position
        x_n = circshift(x, (1,0,0,0))
        x_n = x_n .+ circshift(x, (0,1,0,0))
        if params.lattice=="tri"
            x_n = x_n .+ circshift(x, (1,1,0,0))
        end
        energy = -params.J .* x.*(x_n)
        energy = sum(energy, dims=(1,2,3))
    else
        #n_sum stores the sum of neighbors, reduced over dims=1,2,3
        n_sum = sum(x[2:end,:,:,:].*x[1:end-1,:,:,:], dims=(1,2,3))
        n_sum = n_sum .+ sum(x[:,2:end,:,:].*x[:,1:end-1,:,:], dims=(1,2,3))
        if params.lattice=="tri"
            n_sum = n_sum .+ sum(x[1:end-1,1:end-1,:,:].*x[2:end,2:end,:,:], 
                                dims=(1,2,3))
        end
        energy = -params.J .* (n_sum)
    end
    energy = reshape(energy, size(x)[4])
    return energy
end

function H(x::AbstractArray{T,2}, params::IsingParams) where T
    if params.pbc
        #x_n stores the sum of neighbors at every position
        x_n = circshift(x, (1,0))
        x_n = x_n .+ circshift(x, (0,1))
        if params.lattice=="tri"
            x_n = x_n .+ circshift(x, (1,1))
        end
        energy = -params.J .* x.*(x_n)
        energy = sum(energy)
    else
        #n_sum stores the sum of neighbors, reduced over dims=1,2,3
        n_sum = sum(x[2:end,:].*x[1:end-1,:,])
        n_sum = n_sum + sum(x[:,2:end].*x[:,1:end-1])
        if params.lattice=="tri"
            n_sum = n_sum + sum(x[1:end-1,1:end-1].*x[2:end,2:end])
        end
        energy = -params.J * n_sum
    end
    return energy
end

end