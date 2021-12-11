include("train.jl")

module MonteCarloMethods

using Statistics, DataStructures, DataFrames, CSV, Flux
using ..Ising, ..BPnetModel
export SingleSpinFlip, WolffCluster, neural_mc, PhysicalObservables, display_res, autocor_time

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

function SingleSpinFlip(params::IsingParams, eq_swp, tot_swp, init_conf;
                        beta=0.44)

    L = params.L
    spin = init_conf

    ener = Array{Float64}(undef, tot_swp-eq_swp)
    mag = Array{Float64}(undef, tot_swp-eq_swp)

    for j = 1:tot_swp
        for k = 1:L^2
            p1 = rand(1:L)
            p2 = rand(1:L)
            arg = -2*beta*(spin[p1, p2]*
                    sum_nn(spin, (p1,p2), params))
            if rand() < exp(arg)
                spin[p1,p2] *= -1
            end
        end
        if j>eq_swp
            mag[j-eq_swp] = mean(spin)
            ener[j-eq_swp] = H(spin, params)
        end
    end
    return abs.(mag), ener
end

function WolffCluster(params::IsingParams, eq_swp, tot_swp, init_conf; beta=0.44)

    function nn_list(x1, x2)
        function pbc(x; L=params.L)
            return x==L+1 ? 1 : (x==0 ? L : x)
        end

        if params.pbc
            l = [(pbc(x1+1), x2)
                 (x1, pbc(x2+1))
                 (pbc(x1-1), x2)
                 (x1, pbc(x2-1))]
        else
            l = Tuple{Int,Int}[]
            if x1<params.L
                push!(l, (x1+1, x2))
            end
            if x2<params.L
                push!(l, (x1, x2+1))
            end
            if x1>1
                push!(l, (x1-1, x2))
            end
            if x2>1
                push!(l, (x1, x2-1))
            end
        end
        return l
    end

    spin = init_conf
    L = params.L

    mag = Array{Float64}(undef, tot_swp-eq_swp)
    ener = Array{Float64}(undef, tot_swp-eq_swp)
    cluster = Tuple{Int,Int}[]
    buffer = Queue{Tuple{Int,Int}}()

    for j = 1:tot_swp
        p1, p2 = rand(1:L, 2)
        push!(cluster, (p1,p2))
        enqueue!(buffer, (p1,p2))
        while !isempty(buffer)
            x1, x2 = dequeue!(buffer)
            val = spin[x1, x2]
            for nn in nn_list(x1, x2)
                if !(nn in cluster) && spin[nn...]==val
                    if rand() < 1 - exp(-2*beta)
                        push!(cluster, nn)
                        enqueue!(buffer, nn)
                    end
                end
            end
        end
        for pos in cluster
            spin[pos...] *= -1
        end
        empty!(cluster)
        if j>eq_swp
            mag[j-eq_swp] = mean(spin)
            ener[j-eq_swp] = H(spin, params)
        end
    end
    return abs.(mag), ener
end

struct PhysicalObservables
    m::Float64
    δm::Float64
    E::Float64
    δE::Float64
    χ::Float64
    δχ::Float64
    C::Float64
    δC::Float64
end

function PhysicalObservables(mag, ener, beta, L)
    function autocor_time(x, max=500)
        r = zeros(max+1)
        v = var(x)
        m = mean(x)
        for t in 0:max
            r[t+1] = (mean(x[1:end-max].*x[1+t:end-max+t]) - m^2) / v
            if r[t+1] < 0
                r[t+1] = 0
                break
            end
        end
        return sum(r)
    end
    
    n = size(mag)[1]
    m = mean(mag)
    E = mean(ener)
    τ_m = autocor_time(mag, 200)
    τ_E = autocor_time(ener, 200)
    @info "Autocorrelation times" τ_m τ_E
    δm = sqrt((1+2τ_m)/(n-1))*std(mag)
    δE = sqrt((1+2τ_E)/(n-1))*std(ener)
    #Implementing Jackknife method for error estimation
    n_m = ceil(Int64, n/ceil(2τ_m))
    n_E = ceil(Int64, n/ceil(2τ_E))
    ind_mag = [mag[ceil(Int64, 2τ_m)*i+1] for i in 0:n_m-1]
    ind_ener = [ener[ceil(Int64, 2τ_E)*i+1] for i in 0:n_E-1]
    JK_χ = Array{Float64}(undef, n_m)
    JK_C = Array{Float64}(undef, n_E)
    χ = (beta*L^2)*var(ind_mag)
    C = (beta^2/L^2)*var(ind_ener)
    for i in 1:n_m
        JK_χ[i] = (beta*L^2)*var(ind_mag[1:end .!=i])
    end
    for i in 1:n_E
        JK_C[i] = (beta^2/L^2)*var(ind_ener[1:end .!=i])
    end
    δχ = sqrt(sum((JK_χ.-χ).^2))
    δC = sqrt(sum((JK_C.-C).^2))
    return PhysicalObservables(m, δm, E, δE,
            χ, δχ, C, δC)
end

function autocor_time(x, max=500)
    r = zeros(max+1)
    v = var(x)
    m = mean(x)
    for t in 0:max
        r[t+1] = (mean(x[1:end-max].*x[1+t:end-max+t]) - m^2) / v
        if r[t+1] < 0
            r[t+1] = 0
            break
        end
    end
    return sum(r)
end

function neural_mc(model::BPnet, params, batch_size, beta; n=1)
    L = params.L
    lp = Array{Float64}(undef, 0)
    E = Array{Float64}(undef, 0)
    m = Array{Float64}(undef, 0)
    for l in 1:n
        x2 = sample(model, L, batch_size)
        #Symmetry operations on the sample
        for k in 1:batch_size
            y = x2[:, :, :, k]
            i, j = rand(1:L, 2)
            y = circshift(y, (i, j, 0))
            if rand(Bool)
                y = reverse(y, dims=1)
            end
            if rand(Bool)
                y = reverse(y, dims=2)
            end
            if rand(Bool)
                y = permutedims(y, (2, 1, 3))
            end
            x2[:, :, :, k] = y
        end
        lp2 = log_prob(model, x2) |> cpu
        e2 = H(x2, params) |> cpu
        m2 = dropdims(mean(x2, dims=(1,2,3)), dims=(1,2,3)) |> cpu
        m = cat(m, m2, dims=1)
        lp = cat(lp, lp2, dims=1)
        E = cat(E, e2, dims=1)
    end

    ptr = 1
    acc = 0
    mag = Array{Float64}(undef, size(E)...)
    ener = Array{Float64}(undef, size(E)...)
    mag[1] = m[1]
    ener[1] = E[1]
    for j in 2:size(E)[1]
        arg = beta*(E[ptr]-E[j]) + (lp[ptr]-lp[j])
        if rand() < exp(arg)
            ptr = j
            acc += 1
        end
        mag[j] = abs(m[ptr])
        ener[j] = E[ptr]
    end
    @info "Acceptance Ratio" acc/(size(E)[1]-1.) size(E)[1]
    return mag, ener
end

function display_res(results)
    println("m: $(results.m)")
    println("δm: $(results.δm)")
    println("E: $(results.E)")
    println("δE: $(results.δE)")
    println("χ: $(results.χ)")
    println("δχ: $(results.δχ)")
    println("C: $(results.C)")
    println("δC: $(results.δC)")
end

end
##
