using Statistics

struct IsingParams
    L::Integer
    pbc::Bool
    lattice::String
    J::Float64
end

function IsingParams(L=32;pbc=false, lattice="square", J=1e0)
    return IsingParams(L, pbc, lattice, J)
end

function H(x::AbstractArray, params::IsingParams)
    if params.pbc
        #x_n stores the sum of neighbors at every position
        x_n = circshift(x, (1,0,0,0))
        x_n = x_n .+ circshift(x, (0,1,0,0))
        if params.lattice=="tri"
            x_n = x_n .+ circshift(x, (1,1,0,0))
        end
        energy = J .* x.*(x_n)
        energy = sum(energy, dims=(1,2,3))
    else
        #n_sum stores the sum of neighbors, reduced over dims=1,2,3
        n_sum = sum(x[2:end,:,:,:].*x[1:end-1,:,:,:], dims=(1,2,3))
        n_sum = n_sum .+ sum(x[:,2:end,:,:].*x[:,1:end-1,:,:], dims=(1,2,3))
        if params.lattice=="tri"
            n_sum = n_sum .+ sum(x[1:end-1,1:end-1,:,:].*x[2:end,2:end,:,:], 
                                dims=(1,2,3))
        end
        energy = J .* (n_sum)
    end
    energy = reshape(energy, size(x)[4])
    return energy
end

function sum_nn(spin::Array, pos::Tuple, params::IsingParams)

    function pbc(x; L=size(spin)[1])
        return x==L ? 1 : (x==0 ? L : x)
    end

    p1, p2 = pos
    if params.pbc
        sum = (spin[pbc(p1+1), p2] + spin[p1, pbc(p2+1)] +
               spin[pbc(p1-1), p2] + spin[p1, pbc(p2-1)])
        if params.lattice=="tri"
            sum += spin[pbc(p1+1), pbc(p2+1)] + spin[pbc(p1-1), pbc(p2-1)]
        end
    else
        sum = 0
        if p1<size(spin)[1]
            sum += spin[p1+1, p2]
        end
        if p2<size(spin)[2]
            sum += spin[p1, p2+1]
        end
        if p1>1
            sum += spin[p1-1, p2]
        end
        if p2>1
            sum += spin[p1, p2-1]
        end

        if params.lattice=="tri"
            if p1<size(spin)[1] && p2<size(spin)[2]
                sum += spin[p1+1, p2+1]
            end
            if p1>1 && p2>1
                sum += spin[p1-1, p2-1]
            end
        end
    end
    return convert(Float64, sum)
end

function SingleSpinFlip(params::IsingParams, eq_swp::Integer, init_conf;
                        beta=0.44)

    L = params.L
    spin = init_conf

    autocorr = Array{Float64}(undef, eq_swp+1)
    autocorr[1] = 1
    mag = Array{Float64}(undef, eq_swp+1)
    mag[1] = mean(spin)
    init_lat = copy(spin)

    for j = 1:eq_swp
        for k = 1:L^2
            p1 = rand(1:L)
            p2 = rand(1:L)
            arg = -2*beta*(spin[p1, p2]*
                    sum_nn(spin, (p1,p2), params))
            if rand() < exp(arg)
                spin[p1,p2] *= -1
            end
        end
        autocorr[j+1] = mean(spin.*init_lat)
        mag[j+1] = mean(spin)
    end
    return autocorr, mag
end

function WolffCluster(params::IsingParams, eq_swp::Integer, init_conf)
    
end