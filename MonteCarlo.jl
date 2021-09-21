include("train.jl")

module MonteCarloMethods

using Statistics, DataStructures, DataFrames, CSV
using ..Ising, ..BPnetModel
export SingleSpinFlip, WolffCluster, neural_mc, PhysicalObservables, display_res

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
    function autocor_time(x, max)
        a = Array{Float64}(undef, max)
        for j in 1:max
            a[j] = cor(x[1:end-max], x[1+j:end-max+j])
        end
        return sum(a)+1
    end
    
    n = size(mag)[1]
    m = mean(mag)
    E = mean(ener)
    τ_m = autocor_time(mag, 200)
    τ_E = autocor_time(ener, 200)
    @info "Autocor" τ_m τ_E
    δm = sqrt((1+2τ_m)/(n-1))*std(mag)
    δE = sqrt((1+2τ_E)/(n-1))*std(ener)
    #χ = (beta*L^2)*var(mag)
    #C = (beta^2/L^2)*var(ener)

    bootstrap_χ = Array{Float64}(undef, n)
    bootstrap_C = Array{Float64}(undef, n)
    trial_mag = Array{Float64}(undef, n)
    trial_ener = Array{Float64}(undef, n)
    for i in 1:n
        for j in 1:n
            trial_mag[j] = mag[rand(1:n)]
            trial_ener[j] = ener[rand(1:n)]
        end
        bootstrap_χ[i] = (beta*L^2)*var(trial_mag)
        bootstrap_C[i] = (beta^2/L^2)*var(trial_ener)
    end
    χ = mean(bootstrap_χ)
    C = mean(bootstrap_C)
    δχ = std(bootstrap_χ)
    δC = std(bootstrap_C)
    return PhysicalObservables(m, δm, E, δE,
            χ, δχ, C, δC)
end

function neural_mc(model::BPnet, params, batch_size, beta; n=1, c=missing)
    L = params.L

    function RegressionCoef() #Or, I can save these coefficients as BSON
        x = sample(model, L, 1000) |> cpu #along with the model
        lp = log_prob(model, x) |> cpu
        E = H(x, params) |> cpu
        @. reg(e, p) = -p[1]*beta*e - p[2]
        fit = curve_fit(reg, E, lp, [1.,1000.])
        return coef(fit)
    end

    lp = Array{Float64}(undef, 0)
    E = Array{Float64}(undef, 0)
    m = Array{Float64}(undef, 0)
    #if ismissing(c)
        #c = RegressionCoef()
    #end
    for i in 1:n
        x2 = sample(model, L, batch_size)
        lp2 = log_prob(model, x2) |> cpu
        e2 = H(x2, params) |> cpu
        m2 = dropdims(mean(x2, dims=(1,2,3)), dims=(1,2,3)) |> cpu
        #Removing samples that are outlying the straight line fit
        #ind = Tuple([j for j=1:batch_size if lp2[j]+beta*c[1]*e2[j]<-c[2]])
        #ind = Tuple([])
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
